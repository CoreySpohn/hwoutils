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

    def days(self, time_jd) -> np.ndarray:
        """Offset of a Julian date from the epoch [d], NumPy float64 on the host.

        ``time_jd`` is a numeric Julian date (scalar or array) or an object
        exposing ``jd1`` and ``jd2`` (for example ``astropy.time.Time``),
        which keeps sub-day precision. An object with a ``scale`` attribute
        that differs from the epoch's raises ``ValueError``.
        """
        self._check_scale(time_jd)
        if hasattr(time_jd, "jd1") and hasattr(time_jd, "jd2"):
            return (np.asarray(time_jd.jd1, dtype=np.float64) - self.jd1) + (
                np.asarray(time_jd.jd2, dtype=np.float64) - self.jd2
            )
        return (np.asarray(time_jd, dtype=np.float64) - self.jd1) - self.jd2

    def jd(self, time_d) -> np.ndarray:
        """Julian date of an offset in days, NumPy float64 on the host."""
        return (np.asarray(time_d, dtype=np.float64) + self.jd2) + self.jd1

    def days_between(self, other: Epoch) -> float:
        """Days from ``other`` to this epoch, so ``t_other = t_self + result``."""
        self._check_scale(other)
        return (self.jd1 - other.jd1) + (self.jd2 - other.jd2)


def days_to_seconds(time_d):
    """Convert an offset in days to seconds (works on traced arrays)."""
    return time_d * d2s


def seconds_to_days(time_s):
    """Convert an offset in seconds to days (works on traced arrays)."""
    return time_s / d2s
