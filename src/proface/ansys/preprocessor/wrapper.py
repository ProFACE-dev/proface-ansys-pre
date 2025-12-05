# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

import logging
import re
import uuid
from collections.abc import Generator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Self

import awkward as ak
import numpy as np
import numpy.typing as npt
from ansys.dpf import core as dpf
from ansys.dpf.core.elements import element_types

logger = logging.getLogger(__name__)

_ety_re = re.compile(r"(?P<type>\w+?)(?P<num>\d*)")

topo_ninc = {
    i: int(j.group("num"))
    for i, j in ((x, _ety_re.fullmatch(x.name)) for x in element_types)
    if j and j.group("num")
}

topo_code = {
    element_types.Tet10: "C3D10",
    element_types.Hex20: "C3D20",
    element_types.Wedge15: "C3D15",
    element_types.Pyramid13: "C3D13",
    element_types.Tet4: "C3D4",
    element_types.Hex8: "C3D8",
    element_types.Wedge6: "C3D6",
    element_types.Pyramid5: "C3D5",
}

# type aliases
DPFIdsType = npt.NDArray[np.int32]  # FIXME: check if this is always true
# ansys.dpf.core and awkward are untyped, add only for documentation
DPFServerType = Any
DPFResultsType = Any
DPFOperatorType = Any
DPFMetadataType = Any
AwkwardArrayType = Any


class Model(AbstractContextManager["Model"]):
    """
    Wrapper for ansys.dpf.core.Model, with added utility methods
    and a close method for releasing resources;
    can be used as a context manager.
    """

    def __str__(self) -> str:
        return self._model.__str__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __init__(self, server: DPFServerType, pth: Path):
        logger.debug(
            "Creating model wrapper on server '%s' from '%s'", repr(server), pth
        )

        self.server = server
        if server.local_server:
            self.path = pth.resolve()
        else:
            logger.debug("Upload data source to remote server")
            self.path = dpf.upload_file_in_tmp_folder(
                file_path=pth,
                new_file_name=f"{uuid.uuid4()!s}.rst",
                server=server,
            )
        self._data_sources = dpf.DataSources(
            result_path=self.path, server=self.server
        )
        logger.debug(
            "Data sources result files: %s",
            ", ".join(self._data_sources.result_files),
        )
        self._model = dpf.Model(data_sources=self._data_sources)

        # cache some properties and scopings
        meshed_region = self._model.metadata.meshed_region
        self.eltype = meshed_region.property_field("eltype").data
        self.apdlty = meshed_region.property_field("apdl_element_type").data
        self.elids = meshed_region.elements.scoping.ids
        self.nodids = meshed_region.nodes.scoping.ids
        self.nodcoords = meshed_region.nodes.coordinates_field.data

        # unique eltypes
        self._etypes = [element_types(i) for i in np.unique(self.eltype)]
        self.connectivity = self._make_connectivity()

        # check for consistent shapes
        assert (
            len(self.eltype)
            == len(self.apdlty)
            == len(self.elids)
            == len(self.connectivity)
        )
        assert len(self.nodids) == len(self.nodcoords)

        # flag for finalizer
        self.closed = False

    def _make_connectivity(self) -> AwkwardArrayType:
        flat_conn = self._model.metadata.meshed_region.property_field(
            "connectivity"
        ).data
        cnt = np.zeros(self.eltype.shape, dtype=np.uint32)
        for i in self._etypes:
            cnt[self.eltype == i.value] = topo_ninc[i]
        assert np.all(cnt != 0)
        assert np.sum(cnt) == len(flat_conn)
        return ak.unflatten(flat_conn, cnt)

    def close(self) -> None:
        if self.closed:
            return

        del self._model
        del self._data_sources
        self.closed = True

    def get_3del(
        self,
    ) -> Generator[tuple[str, DPFIdsType, DPFIdsType]]:
        for i in self._etypes:
            try:
                et = topo_code[i]
            except KeyError:
                continue
            mask = self.eltype == i.value
            yield (
                et,
                self.elids[mask],
                self.nodids[self.connectivity[mask]],
            )

    def get_by_apdl_type(
        self, apdl_eltypes: Sequence[int]
    ) -> Generator[tuple[DPFIdsType, DPFIdsType]]:
        """get connectivity filtering elements by apdl eltype"""
        mask = np.isin(self.apdlty, apdl_eltypes)
        elids = self.elids[mask]
        conn = self.connectivity[mask]
        cnt = ak.num(conn)
        for n in np.unique(cnt):
            mask_n = cnt == n
            elids_n = elids[mask_n]
            conn_n = conn[mask_n]
            yield elids_n, self.nodids[conn_n]

    def equal_mesh(self, other: Self) -> bool:
        """check if others mesh is 'equal' to ours'"""
        # differences in ids are tolerated
        for attr in ["eltype", "apdlty", "nodcoords", "connectivity"]:
            if not ak.array_equal(getattr(self, attr), getattr(other, attr)):
                logger.debug("Mesh attribute '%s' mismatch", attr)
                return False
        return True

    @property
    def metadata(self) -> DPFMetadataType:
        if self.closed:
            msg = "Operation on closed model"
            raise ValueError(msg)
        return self._model.metadata

    @property
    def operator(self) -> DPFOperatorType:
        if self.closed:
            msg = "Operation on closed model"
            raise ValueError(msg)
        return self._model.operator

    @property
    def results(self) -> DPFResultsType:
        if self.closed:
            msg = "Operation on closed model"
            raise ValueError(msg)
        return self._model.results

    @property
    def time_ids(self) -> DPFIdsType:
        """list of available time ids"""
        ret: DPFIdsType
        if self.closed:
            msg = "Operation on closed model"
            raise ValueError(msg)
        ret = (
            self._model.metadata.time_freq_support.time_frequencies.scoping.ids
        )
        # FIXME: why code below can give different results?
        # ret = (
        #     self._model.metadata.time_freq_support
        #     .prop_field_support_by_property("time_freqs_cumulative_ids").data
        # )

        return ret
