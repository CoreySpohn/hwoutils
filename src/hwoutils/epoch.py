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

import numpy as np

from hwoutils.constants import d2s

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
