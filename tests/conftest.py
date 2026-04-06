"""Root pytest configuration.

TensorFlow must be imported before numpy/pandas on Windows to avoid a
DLL initialisation race in the native runtime.  Importing it here in
the top-level conftest ensures it is loaded first for every test
subdirectory during collection.
"""

from __future__ import annotations

try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    pass
