<!--
SPDX-FileCopyrightText: 2025 ProFACE developers

SPDX-License-Identifier: MIT
-->

# ProFACE-Ansys-Pre

ProFACE-Ansys-Pre is a python package that provides the `proface.preprocessor.ansys` plugin for use with the [`proface-pre`](https://github.com/ProFACE-dev/proface-pre) command-line interface (CLI).

This plugin enables the conversion of Ansys `.rst` results files to ProFACE `.h5` FEA input files.

<!--
## Installation

Install from <https://pypi.org> with

```
pip install proface-ansys-pre
```
-->

## Usage

Running

```
proface-pre example.toml
```

will produce `example.h5` from `example.rst`.

`example.toml` has the following format:

```toml
fea_software = "Ansys"

[Ansys.input]
rst = "example.rst"

[Ansys.results.ref_load]
id = 1
```

## License

ProFACE-Ansys-Pre is licensed under the MIT license.
ProFACE-Ansys-Pre makes no claims about [ProFACE](https://proface.polimi.it) which is a distinct program with different licensing requirements.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Ansys, Inc., The HDF Group, or any of their respective products.

- Ansys is a registered trademark of Ansys, Inc.
- HDF5 is a trademark of The HDF Group.

All trademarks and registered trademarks are the property of their respective owners.

## Funding & Disclaimer

The research leading to this software has been co-funded by the European Union under Grant Agreement No [101103504](https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/projects-details/44181033/101103504/EDF).

<img src="assets/EN_Co-fundedbytheEU_RGB_POS.png" alt="Co-Funded by the EU" style="width:20em; height:auto;" />

> **Disclaimer:**
> Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or the granting authority. Neither the European Union nor the granting authority can be held responsible for them.
