# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

"""ProFACE Preprocessor package for Ansys FEA"""

from ._version import __version__
from .translator import AnsysTranslatorError, main

__all__ = ["AnsysTranslatorError", "__version__", "main"]
