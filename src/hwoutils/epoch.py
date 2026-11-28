"""A reference epoch: the day zero that time offsets are measured from.

Traced code carries times as plain arrays of days from an epoch (``time_d``).
The epoch itself is host-side metadata. Subtracting it from a Julian date
happens once, in NumPy float64, before any array reaches a device, so that
float32 code keeps sub-second resolution over decades instead of the quarter
day that an absolute Julian date resolves in float32.

The epoch is a two-part Julian date (``jd1 + jd2``), the convention of the
SOFA/ERFA libraries, which keeps the subtraction exact well below a
microsecond. ``scale`` labels the time scale; no conversion between scales
is applied, and mixing scales raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hwoutils.constants import d2s

if TYPE_CHECKING:
    import jax

_MJD_OFFSET_D = 2400000.5  # JD - MJD [d]
_SCALES = ("tdb", "tt", "tcb", "tai", "utc")


@dataclass(frozen=True)
class Epoch:
    """Day zero for time offsets, as a two-part Julian date on one time scale.

    Frozen and hashable, so it can be a static (non-traced) field of a model.

    Attributes:
        jd1: First part of the Julian date [d].
        jd2: Second part of the Julian date [d].
        scale: Time scale label, one of ``tdb``, ``tt``, ``tcb``, ``tai``, ``utc``.
    """

    jd1: float = 0.0
    jd2: float = 0.0
    scale: str = "tdb"

    def __post_init__(self):
        """Store host floats and validate the scale label."""
        object.__setattr__(self, "jd1", float(self.jd1))
        object.__setattr__(self, "jd2", float(self.jd2))
        if self.scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {self.scale!r}")

    def _check_scale(self, time):
        scale = getattr(time, "scale", None)
        if isinstance(scale, str) and scale != self.scale:
            raise ValueError(
                f"time is on scale {scale!r} but the epoch is on {self.scale!r}"
            )

    def days(self, time=None, *, jd=None, mjd=None) -> np.ndarray:
        """Offset of a date from the epoch [d], NumPy float64 on the host.

        Exactly one of ``time``, ``jd`` and ``mjd`` is given. A bare number is
        never accepted positionally, because a Julian date and a modified
        Julian date differ by 2400000.5 d and the two cannot be told apart.
        Name the dialect: ``jd=`` for a Julian date, ``mjd=`` for a modified
        Julian date (JD - 2400000.5). The ``mjd=`` offset subtracts the epoch
        parts from the date in an order that keeps float64 precision: the
        shift ``jd1 - 2400000.5`` is exact for any ``jd1`` that is a multiple
        of 0.5 d, so the only rounding is that of the final differences.

        Args:
            time: Object exposing ``jd1`` and ``jd2`` [d] (for example
                ``astropy.time.Time``), which keeps sub-day precision. An
                object with a ``scale`` attribute that differs from the
                epoch's raises ``ValueError``.
            jd: Julian date [d] as a number or array.
            mjd: Modified Julian date [d] (JD - 2400000.5) as a number or array.

        Returns:
            Offset from the epoch [d], float64, with the shape of the date.

        Raises:
            ValueError: If not exactly one of ``time``, ``jd`` and ``mjd`` is
                given, or if the time scales differ.
            TypeError: If ``time`` is a bare number or array.
        """
        if sum(x is not None for x in (time, jd, mjd)) != 1:
            raise ValueError("give exactly one of time, jd= or mjd=")
        if time is not None:
            if not (hasattr(time, "jd1") and hasattr(time, "jd2")):
                raise TypeError(
                    "a bare number is ambiguous between Julian and modified Julian "
                    "dates; name the dialect with jd= or mjd=, or pass an object "
                    "with jd1 and jd2 (for example astropy Time)"
                )
            self._check_scale(time)
            return (np.asarray(time.jd1, dtype=np.float64) - self.jd1) + (
                np.asarray(time.jd2, dtype=np.float64) - self.jd2
            )
        if jd is not None:
            return (np.asarray(jd, dtype=np.float64) - self.jd1) - self.jd2
        shift = self.jd1 - _MJD_OFFSET_D
        return (np.asarray(mjd, dtype=np.float64) - shift) - self.jd2

    def jd(self, time_d) -> np.ndarray:
        """Julian date of an offset in days, NumPy float64 on the host.

        Args:
            time_d: Offset from the epoch [d], scalar or array.

        Returns:
            Julian date [d], float64, with the shape of ``time_d``.
        """
        return (np.asarray(time_d, dtype=np.float64) + self.jd2) + self.jd1

    def days_between(self, other: Epoch) -> float:
        """Days from ``other`` to this epoch, so ``t_other = t_self + result``.

        Args:
            other: The epoch to measure from. Its scale must match this epoch's.

        Returns:
            Difference between the two epochs [d].
        """
        self._check_scale(other)
        return (self.jd1 - other.jd1) + (self.jd2 - other.jd2)


def days_to_seconds(time_d):
    """Convert an offset in days to seconds (works on traced arrays).

    Args:
        time_d: Offset [d].

    Returns:
        The offset [s].
    """
    return time_d * d2s


def seconds_to_days(time_s):
    """Convert an offset in seconds to days (works on traced arrays).

    Args:
        time_s: Offset [s].

    Returns:
        The offset [d].
    """
    return time_s / d2s


def elapsed_seconds(
    time_d, *, origin_d: float, reference=None, what: str = "elapsed_seconds"
) -> jax.Array:
    """Convert days to seconds from a static origin, subtracting in binary64.

    Args:
        time_d: Scalar or array of days on the caller's clock.
        origin_d: Origin [d], a Python float fixed during tracing.
        reference: Optional array or pytree supplying precision for weak times
            when the origin is zero.
        what: Name of the elapsed-seconds input in error messages.

    Returns:
        Strong JAX array of seconds from the origin. At zero origin, explicit
        float32 and float64 inputs keep their available precision. For a host
        subtraction, the active precision determines the output dtype.

    Raises:
        TypeError: If the origin is not a Python float.
        PrecisionError: If a nonzero origin receives a narrow floating time,
            or if a traced subtraction cannot use float64.
        ValueError: If a nonzero origin receives an unsupported time dtype.
    """
    import jax
    import jax.numpy as jnp

    from hwoutils.precision import float_dtype, require_float64

    if type(origin_d) is not float:
        name = "epoch_d" if what == "elapsed_s" else "origin_d"
        raise TypeError(f"{name} must be a static host float, not a leaf")
    if origin_d == 0.0:
        if isinstance(time_d, (tuple, list)):
            time_d = jnp.asarray(time_d)
        dtype = getattr(getattr(time_d, "dtype", None), "type", None)
        if dtype in (np.float32, np.float64) and not getattr(
            time_d, "weak_type", False
        ):
            dtype = jax.dtypes.canonicalize_dtype(dtype)
        elif reference is None:
            dtype = float_dtype()
        else:
            leaves = (
                leaf
                for leaf in jax.tree.leaves(reference)
                if isinstance(leaf, jax.Array)
                and not getattr(leaf, "weak_type", False)
                and jnp.issubdtype(leaf.dtype, jnp.inexact)
            )
            dtype = jax.dtypes.canonicalize_dtype(jnp.result_type(*leaves, time_d, 1.0))
        return days_to_seconds(jnp.asarray(time_d, dtype=dtype))

    dtype = float_dtype()
    if isinstance(time_d, jax.Array):
        if time_d.dtype != np.float64:
            _reject_time_dtype(time_d.dtype, origin_d, what)
        if isinstance(time_d, jax.core.Tracer) or dtype == np.float64:
            require_float64(
                f"Traced epoch subtraction (or pass {what} formed on the host)"
            )
            return days_to_seconds(jnp.asarray(time_d, dtype=dtype) - origin_d)
    host = np.asarray(time_d)
    kind, size = host.dtype.kind, host.dtype.itemsize
    if not (kind in "iu" or (kind == "f" and size == 8)):
        _reject_time_dtype(host.dtype, origin_d, what)
    return jnp.asarray(days_to_seconds(host.astype(np.float64) - origin_d), dtype=dtype)


def _reject_time_dtype(dtype: np.dtype, origin_d: float, what: str):
    from jax.numpy import floating, issubdtype

    from hwoutils.precision import PrecisionError

    name = "epoch_d" if what == "elapsed_s" else "origin_d"
    if dtype.itemsize < 8 and issubdtype(dtype, floating):
        raise PrecisionError(
            f"time_d of dtype {dtype} may have lost its offset from "
            f"{name}={origin_d!r}. Pass {what} formed on the host from "
            "the original times, or rebuild time_d in float64 from those times."
        )
    raise ValueError(f"unsupported time_d dtype {dtype} for {name}={origin_d!r}")
