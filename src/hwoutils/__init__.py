"""hwoutils -- Shared utilities for the HWO simulation suite.

Constants, conversions and ``Epoch`` need only NumPy. The JAX modules, including
the Equinox-based ``blocks`` and ``model_paths``, need the ``hwoutils[jax]`` extra and
import JAX themselves.
"""

from importlib import import_module as _import_module

try:
    from hwoutils._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

_EXPORTS = {
    "PrecisionError": "precision",
    "complex_of": "precision",
    "elapsed_seconds": "epoch",
    "enable_x64": "precision",
    "float_dtype": "precision",
    "interp_linear": "interpolation",
    "like": "precision",
    "precision_of": "precision",
    "provenance": "provenance",
    "require_float64": "precision",
    "set_host_device_count": "jax_config",
    "set_platform": "jax_config",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    """Load the JAX helpers only when requested.

    Args:
        name: Attribute name looked up on the package.

    Returns:
        The requested public attribute.

    Raises:
        AttributeError: If ``name`` is not a lazy public export.
    """
    if name in _EXPORTS:
        module = _import_module(f"hwoutils.{_EXPORTS[name]}")
        value = module if name == "provenance" else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy public exports in interactive discovery.

    Returns:
        Sorted names of the module globals and the lazy public exports.
    """
    return sorted(set(globals()) | set(__all__))
