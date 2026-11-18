"""hwoutils -- Shared utilities for the HWO simulation suite.

Constants, conversions and ``Epoch`` need only NumPy. The JAX modules need
the ``hwoutils[jax]`` extra and import JAX themselves.
"""

from importlib import import_module as _import_module

try:
    from hwoutils._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "enable_x64",
    "set_host_device_count",
    "set_platform",
]


def __getattr__(name):
    """Load the JAX configuration helpers only when requested."""
    if name in __all__ and name != "__version__":
        value = getattr(_import_module("hwoutils.jax_config"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
