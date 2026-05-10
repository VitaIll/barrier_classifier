"""Feature family modules.

Each family file (e.g. volatility.py) declares Feature subclasses for one
logical group of columns. Importing this package auto-imports every .py
file in this directory, which fires Feature.__init_subclass__ and populates
the registry.

Empty at step 1; grows one family per migration step.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def _autoimport() -> None:
    pkg_path = Path(__file__).parent
    for mod_info in pkgutil.iter_modules([str(pkg_path)]):
        if mod_info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{mod_info.name}")


_autoimport()
