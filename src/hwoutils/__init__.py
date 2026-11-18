"""hwoutils -- Shared JAX-based utilities for the HWO simulation suite."""

from importlib import import_module as _import_module

try:
    from hwoutils._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

_GRID_EXPORTS = ("CellOverlap", "Measure", "footprint_overlap", "integrate_footprints")

__all__ = [
    *_GRID_EXPORTS,
    "__version__",
    "enable_x64",
    "radial_profile",
    "set_host_device_count",
    "set_platform",
]


def __getattr__(name):
    """Load JAX-backed exports only when requested."""
    if name in __all__ and name != "__version__":
        module = "radial" if name == "radial_profile" else "jax_config"
        if name in _GRID_EXPORTS:
            module = "grids"
        value = getattr(_import_module(f"hwoutils.{module}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
