"""hwoutils -- Shared JAX-based utilities for the HWO simulation suite."""

try:
    from hwoutils._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

from hwoutils.jax_config import enable_x64, set_host_device_count, set_platform

__all__ = [
    "__version__",
    "enable_x64",
    "set_host_device_count",
    "set_platform",
]
