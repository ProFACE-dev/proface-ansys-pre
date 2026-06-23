# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

import datetime
import logging
import os
import tempfile
import textwrap
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
from ansys.dpf import core as dpf

from proface.ansys.preprocessor.wrapper import DPFServerType, Model
from proface.preprocessor import DIM, VOIGT_NOTATION, PreprocessorError

from ._version import __version__

if TYPE_CHECKING:
    from types import EllipsisType

logger = logging.getLogger(__name__)

# save results in single precision
RES_DTYPE = np.float32
COMPLEX_RES_DTYPE = np.complex64
# intermediate computations in double precision
TMP_DTYPE = np.float64
COMPLEX_TMP_DTYPE = np.complex128
# relative tolerance for equality testing
RTOL = np.finfo(RES_DTYPE).eps.item()
# absolute tolerance for equality testing
ATOL = 0.0
ATOL_INFTY = 2**-30

# Ansys Voigt Notation
VOIGT_NOTATION_A = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
# permutation from Ansys to target Voigt notation
V_PERM = [VOIGT_NOTATION_A.index(i) for i in VOIGT_NOTATION]
assert all(
    VOIGT_NOTATION[i] == VOIGT_NOTATION_A[V_PERM[i]] for i in range(len(V_PERM))
)

# type aliases
type H5Container = Any
type AnsysCumulativeTimeID = int


class AnsysTranslatorError(PreprocessorError):
    """Generic Ansys translator error"""


def main(*, job: dict[str, Any], job_path: Path, h5: H5Container) -> None:
    logger.info(
        "\U0001f680 START Ansys to ProFACE translator, ver. %s",
        __version__,
    )  # 🚀

    #
    # load DPF server
    #
    server = _load_server()

    logger.debug("Job %s", job)
    job_dir = job_path.parent

    with tempfile.TemporaryDirectory(prefix="wbpz_") as tmpdir:
        logger.debug("Temp directory %s", tmpdir)

        #
        # .wbpz file or .rst direct access
        #
        rst = _get_rst(
            pth=job["input"]["rst"],
            job_dir=job_dir,
            work_dir=tmpdir,
            wbpz=job["input"].get("wbpz"),
        )
        logger.debug("rst '%s'", rst)
        if "rst_ip" in job["input"]:
            rst_ip = _get_rst(
                pth=job["input"]["rst_ip"],
                job_dir=job_dir,
                work_dir=tmpdir,
                wbpz=job["input"].get("wbpz"),
            )
            logger.debug("rst_ip '%s'", rst_ip)
        else:
            rst_ip = None

        try:
            #
            # create DPF Model wrappers
            #
            logger.debug("Wrap DPF models")
            with (
                Model(server, rst) as model,
                Model(server, rst_ip) if rst_ip else nullcontext() as model_ip,
            ):
                logger.info(
                    "Main model: '%s'\n%s",
                    job["input"]["rst"],
                    textwrap.indent(str(model).strip(), prefix=" " * 6),
                )
                #
                # save results
                #
                _main(model=model, model_ip=model_ip, job=job, h5=h5)
        except dpf.errors.DpfVersionNotSupported as err:
            msg = (
                f"Current DPF Server (v{server.info['server_version']} "
                f"does not support required capabilities: {err}"
            )
            raise AnsysTranslatorError(msg) from err

    logger.info("\U0001f3c1 END Ansys to ProFACE translator")  # 🏁


def _load_server() -> DPFServerType:
    "return DPF server instance"

    #
    # prepare environment for suppressing noisy message from
    # ansys-dpf-core and grpcio
    #
    # ansys-dpf-core:
    noisylog = logging.getLogger("ansys.dpf.core.server_types")
    noisylog.setLevel(logging.NOTSET)
    # grpcio:
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    # tqdm:
    os.environ["TQDM_DISABLE"] = "1"

    try:
        server = dpf.server.get_or_create_server(None)
    except (OSError, ValueError) as err:
        logger.error("Unable to get DPF server:\n%s", err)
        msg = "Unable to load DPF server"
        raise AnsysTranslatorError(msg) from err
    if server.local_server:
        logger.info(
            "Local DPF server: '%s' (v. %s)",
            server.info["path"],
            server.info["server_version"],
        )
    else:
        logger.info(
            "Remote DPF server: '%s:%s' (v. %s)",
            server.info["server_ip"],
            server.info["server_port"],
            server.info["server_version"],
        )
    return server


def _get_rst(
    pth: str, job_dir: Path, work_dir: str, wbpz: str | None = None
) -> Path:
    if wbpz is not None:
        with ZipFile(job_dir / wbpz, "r") as zip_file:
            rst = Path(zip_file.extract(pth, path=work_dir))
    else:
        rst = job_dir / pth
    return rst


def _main(
    *,
    model: Model,
    model_ip: Model | None,
    job: dict[str, Any],
    h5: H5Container,
) -> None:
    """helper function to reduce indentation in main"""

    # collect available cumulative time_ids
    ts = model.metadata.time_freq_support.n_sets
    logger.debug("Detected cumulative time ids: %s", ts)

    if model_ip:
        logger.info("IP model: '%s'", job["input"]["rst_ip"])

        logger.debug("Check functional mesh equality")
        if not model.equal_mesh(model_ip):
            msg = "Base and IP meshes are not compatible"
            raise AnsysTranslatorError(msg)

        # logger.debug("Check strict mesh equality via DPF")
        # if not _equal_mesh(
        #     model_ip.metadata.meshed_region,
        #     model.metadata.meshed_region,
        # ):
        #     logger.warning(
        #         "Mesh for results at IPs not identical to base mesh"
        #     )

        ts_ip = model_ip.metadata.time_freq_support.n_sets
        logger.debug("Detected ip model cumulative time ids: %s", ts_ip)
        if not np.all(ts == ts_ip):
            msg = "Base and IP results do not share identical time support"
            logger.warning(msg)
    else:
        logger.warning("NO IP model: evaluating results at element centroids")

    #
    # save meta, mesh, sets info
    #
    _meta(model, h5)
    _mesh(model, h5)
    _sets(model, h5)

    #
    # save load cases
    #
    for load_case in job["results"]:
        cumulative_time_id = job["results"][load_case]["id"]

        if not isinstance(cumulative_time_id, int):
            msg = "'id' must be an int"
            raise TypeError(msg)

        logger.info("Saving results %s", load_case)
        if cumulative_time_id not in ts:
            logger.error(
                "Results %s: cumulative time id %d not found in results",
                load_case,
                cumulative_time_id,
            )

        h5gr = h5.create_group(f"results/{load_case}")
        # save nodal stresses
        _s_nod(model, h5gr, cumulative_time_id)
        # save IP/centroid stresses and volumes
        if model_ip:
            if not _identical_disp(model, model_ip, cumulative_time_id):
                msg = "Base and IP results not equal"
                raise AnsysTranslatorError(msg)
            _svol_ip(model_ip, h5gr, cumulative_time_id)
        else:
            _svol_ip(model, h5gr, cumulative_time_id, centroid=True)


def _equal_mesh(ma: Model, mb: Model) -> bool:
    """check mesh equality via DPF operator

    This operation is very costly and strict"""
    op = dpf.operators.logic.identical_meshes(
        meshA=ma,
        meshB=mb,
        small_value=ATOL,  # optional
        tolerance=RTOL,  # optional
        compare_auxiliary=True,  # optional
    )
    return bool(op.outputs.are_identical())


def _meta(model: Model, h5: H5Container) -> None:
    #
    # h5 metadata
    #
    logger.debug("Save results metadata")
    h5.attrs["program"] = "Ansys"
    h5.attrs["version"] = model.metadata.result_info.solver_version
    h5.attrs["run_datetime"] = datetime.datetime(
        *_decode_decimal(model.metadata.result_info.solver_date),
        *_decode_decimal(model.metadata.result_info.solver_time),
    ).isoformat()
    h5.attrs["title"] = model.metadata.result_info.main_title


def _mesh(model: Model, h5: H5Container) -> None:
    logger.debug("Save mesh data")
    # nodes
    h5.create_dataset("nodes/coordinates", data=model.nodcoords)
    h5.create_dataset("nodes/numbers", data=model.nodids)

    # elements
    for s, ids, conn in model.get_3del():
        h5gr = h5.create_group(f"elements/{s}")
        h5gr.create_dataset("incidences", data=conn)
        h5gr.create_dataset("numbers", data=ids)
        h5gr.create_dataset("nodes", data=np.unique(conn))


def _sets(model: Model, h5: H5Container) -> None:
    logger.debug("Save sets")
    for name in model.metadata.available_named_selections:
        op = model.operator("scoping_provider_by_ns")
        op.inputs.named_selection_name.connect(name)
        scoping = op.outputs.mesh_scoping()
        match scoping.location:
            case "Nodal":
                h5.create_dataset(
                    f"sets/node/{name}", data=np.sort(scoping.ids)
                )
            case "Elemental":
                h5.create_dataset(
                    f"sets/element/{name}", data=np.sort(scoping.ids)
                )
            case mistery:
                logger.error("%s: unknown scoping '%s'", name, mistery)

    ## internal sets, associated with TARGET (170) and CONTA (174) elements
    elset = np.array([], dtype=np.uint32)
    nset = np.array([], dtype=np.uint32)
    for elids, conn in model.get_by_apdl_type((170, 174)):
        logger.debug("Found %s elements/shape in internal set", conn.shape)
        elset = np.union1d(elset, elids)
        nset = np.union1d(nset, conn)
    if np.size(nset):
        # FIXME, use better name for special 'internal' set
        h5.create_dataset("sets/node/internal", data=nset)


def _s_nod(
    model: Model, h5: H5Container, time_id: AnsysCumulativeTimeID
) -> None:
    """save S averaged at nodes, from ElementalNodal"""
    logger.debug("Save S/SP results at cumulative time_id '%s'", time_id)

    for code, ids, _conn in model.get_3del():
        scoping = dpf.Scoping(location=dpf.locations.elemental_nodal, ids=ids)
        st, st_imag = _stress_fields(model, time_id, scoping)
        if st is None:
            logger.error("Unable to get stresses for time id %d", time_id)
            continue

        assert st is not None
        st = st.to_nodal()
        if st_imag is not None:
            st_imag = st_imag.to_nodal()
            _assert_same_field_layout(st, st_imag)

        # FIXME FIXME: do *not* get node ids from h5
        node_ids_h5 = h5.file[f"elements/{code}/nodes"][()]
        assert st.elementary_data_count == len(node_ids_h5)
        perm = np.searchsorted(node_ids_h5, st.scoping.ids)
        assert np.all(node_ids_h5[perm] == st.scoping.ids)

        data = _field_data(st, st_imag)[..., V_PERM]
        out = np.empty(shape=st.shape, dtype=data.dtype)
        out[perm] = data
        h5.create_dataset(f"S/nodal_averaged/{code}", data=out)
        h5.create_dataset(f"SP/nodal_averaged/{code}", data=_sp(out))


def _identical_disp(
    model_a: Model, model_b: Model, time_id: AnsysCumulativeTimeID
) -> bool:
    ua = (
        model_a.results.displacement(time_scoping=time_id)
        .outputs.fields_container()
        .get_field_by_time_id(time_id)
        .data
    )
    ub = (
        model_b.results.displacement(time_scoping=time_id)
        .outputs.fields_container()
        .get_field_by_time_id(time_id)
        .data
    )

    assert ua.shape == ub.shape
    if np.all(ua == ub):
        return True
    logger.warning(
        "Solution for IP model is not identical to main model: "
        "error \N{INFINITY}-norm %.3g (units of length)",
        np.linalg.norm(ua.ravel() - ub.ravel(), ord=np.inf),
    )
    atol = ATOL_INFTY * np.linalg.norm(ub, ord=np.inf)
    return np.allclose(ua, ub, rtol=RTOL, atol=atol)


def _svol_ip(
    model: Model,
    h5: H5Container,
    time_id: AnsysCumulativeTimeID,
    *,
    centroid: bool = False,
) -> None:
    """save results at IPs"""
    logger.debug("Save IP results at cumulative time_id '%s'", time_id)

    for code, ids, _conn in model.get_3del():
        # FIXME: do not read h5 root
        assert np.all(ids == h5.file[f"elements/{code}/numbers"][()])

        #
        # S, SP
        #
        scoping = dpf.Scoping(location=dpf.locations.elemental_nodal, ids=ids)
        st, st_imag = _stress_fields(model, time_id, scoping)
        if st is None:
            logger.error("Unable to get stresses for time id %d", time_id)
            continue

        assert st is not None
        assert st.location == "ElementalNodal"
        assert np.all(st.scoping.ids == ids)
        if st_imag is not None:
            _assert_same_field_layout(st, st_imag)

        if centroid:
            st = dpf.operators.averaging.elemental_mean(
                field=st
            ).outputs.field()
            assert st.location == "Elemental"
            if st_imag is not None:
                st_imag = dpf.operators.averaging.elemental_mean(
                    field=st_imag
                ).outputs.field()
                assert st_imag.location == "Elemental"
                _assert_same_field_layout(st, st_imag)

        assert np.all(st.scoping.ids == ids)

        n_ip = st.elementary_data_count // len(ids)
        assert (st.elementary_data_count % n_ip) == 0
        assert not (centroid and n_ip != 1)

        _, n_comp = st.elementary_data_shape
        assert _ == 1

        out = _field_data(st, st_imag).reshape((-1, n_ip, n_comp))[
            ..., V_PERM
        ]
        assert out.shape == (len(ids), n_ip, n_comp)

        h5.create_dataset(f"S/integration_point/{code}", data=out)
        h5.create_dataset(f"SP/integration_point/{code}", data=_sp(out))

        #
        # IVOL
        #
        scoping = dpf.Scoping(location=dpf.locations.elemental, ids=ids)
        ivol = (
            model.results.elemental_volume(
                time_scoping=time_id, mesh_scoping=scoping
            )
            .outputs.fields_container()
            .get_field_by_time_id(time_id)
        )
        assert ivol is not None

        assert np.all(ivol.scoping.ids == ids)
        assert ivol.elementary_data_count == len(ids)
        assert ivol.elementary_data_shape == 1

        # if centroid is False:
        #   crude assumption: elemental vol equally distributed to ips
        # else if centroid is True:
        #   n_ip is 1, so we are just adding an axis
        out = np.broadcast_to(
            ivol.data[..., np.newaxis] / n_ip,
            (len(ids), n_ip),
        )
        h5.create_dataset(
            f"IVOL/integration_point/{code}", data=np.astype(out, RES_DTYPE)
        )


def _stress_fields(
    model: Model, time_id: AnsysCumulativeTimeID, scoping: dpf.Scoping
) -> tuple[Any | None, Any | None]:
    """Return real and optional imaginary stress fields."""
    fc = model.results.stress(
        time_scoping=time_id, mesh_scoping=scoping
    ).outputs.fields_container()
    real = fc.get_field_by_time_id(time_id)
    if real is None:
        return None, None

    if not fc.has_label("complex"):
        return real, None

    imag = fc.get_imaginary_field(time_id)
    if imag is None:
        msg = f"Imaginary stresses unavailable for time id {time_id}"
        raise AnsysTranslatorError(msg)
    _assert_same_field_layout(real, imag)
    return real, imag


def _assert_same_field_layout(real: Any, imag: Any) -> None:
    assert real.location == imag.location
    assert real.elementary_data_count == imag.elementary_data_count
    assert real.elementary_data_shape == imag.elementary_data_shape
    assert np.all(real.scoping.ids == imag.scoping.ids)


def _field_data(real: Any, imag: Any | None = None) -> npt.NDArray[Any]:
    real_data = np.astype(real.data, RES_DTYPE)
    if imag is None:
        return real_data

    imag_data = np.astype(imag.data, RES_DTYPE)
    out = np.empty(real_data.shape, dtype=COMPLEX_RES_DTYPE)
    out.real = real_data
    out.imag = imag_data
    return out


def _decode_decimal(i: int) -> tuple[int, int, int]:
    """solve for a, b, c from
    i = (a * 100 + b) * 100 + c"""

    a, b = divmod(i, 100_00)
    b, c = divmod(b, 100)
    assert i == (a * 100 + b) * 100 + c

    return a, b, c


def _sp(s: np.ndarray) -> npt.NDArray[Any]:
    """compute principal stresses from array 's'"""
    logger.debug("Compute principal stresses SP")

    # s is a (*sdim)-stack of data in Voigt notation
    *sdim, _ = s.shape
    assert _ == len(VOIGT_NOTATION)

    # st is a (*sdim)-stack of 2-tensors in a DIM dimensional space
    tmp_dtype = COMPLEX_TMP_DTYPE if np.iscomplexobj(s) else TMP_DTYPE
    st = np.zeros(shape=(*sdim, DIM, DIM), dtype=tmp_dtype)

    # tperm is tensor to Voigt indices permutation, UPLO=U for real tensors

    tperm: tuple[EllipsisType, tuple[int, ...], tuple[int, ...]] = (
        ...,
        *zip(*VOIGT_NOTATION, strict=True),
    )  # type: ignore[assignment]
    st[tperm] = s
    st[(..., tperm[2], tperm[1])] = s

    # return eigenvalues, i.e. principal stresses
    if np.iscomplexobj(st):
        return np.linalg.eigvals(st).astype(COMPLEX_RES_DTYPE)
    return np.linalg.eigvalsh(st, UPLO="U").astype(RES_DTYPE)

    # here a rhs permutation
    # rperm = (..., [[0,3,4],[3,1,5],[4,5,2]])
    # st = s[rperm]
