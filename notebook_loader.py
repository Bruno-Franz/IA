"""Utilities for importing Jupyter notebooks as modules."""
from __future__ import annotations

from pathlib import Path
import types

import nbformat
from nbconvert import PythonExporter


def load_notebook_as_module(nb_path: str) -> types.ModuleType:
    """Load an IPython notebook as a Python module."""
    nb = nbformat.read(nb_path, as_version=4)
    code, _ = PythonExporter().from_notebook_node(nb)
    module = types.ModuleType(Path(nb_path).stem)
    exec(code, module.__dict__)
    return module
