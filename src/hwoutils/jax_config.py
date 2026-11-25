"""JAX runtime configuration helpers.

Centralizes JAX platform, precision, and device-count settings so that
every library in the workspace uses the same API and stays current with
upstream deprecations.
"""

import os
import re

import jax

from hwoutils.precision import dtype_tag as dtype_tag
from hwoutils.precision import enable_x64 as enable_x64
from hwoutils.precision import float_dtype as float_dtype


def set_platform(platform: str | None = None) -> None:
    """Select the JAX compute platform (cpu, gpu, or tpu).

    Must be called before any JAX computation.

    Args:
        platform: One of ``'cpu'``, ``'gpu'``, or ``'tpu'``.
            Defaults to the ``JAX_PLATFORMS`` environment variable,
            or ``'cpu'`` if unset.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_platforms", platform)


def set_host_device_count(n: int) -> None:
    """Expose *n* CPU cores as separate XLA devices.

    By default XLA treats all CPU cores as one device. This function
    sets the ``XLA_FLAGS`` environment variable so that
    :func:`jax.pmap` can distribute work across *n* host devices.

    Must be called before any JAX computation.

    Args:
        n: Number of CPU devices to expose.
    """
    xla_flags_str = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags_str
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={n}", *xla_flags]
    )
