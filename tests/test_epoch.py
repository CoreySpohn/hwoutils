"""Tests for the reference epoch and day offsets."""

import dataclasses

import numpy as np
import pytest

from hwoutils.epoch import Epoch, days_to_seconds, seconds_to_days


class _TwoPart:
    def __init__(self, jd1, jd2, scale="tdb"):
        self.jd1, self.jd2, self.scale = jd1, jd2, scale


def test_offsets_keep_sub_microsecond_precision_and_round_trip():
    """Fails if the epoch is subtracted after the two parts are summed.

    Truth: exact rational arithmetic on the two parts. One microsecond is
    1.16e-11 d; a single float64 Julian date near 2.46e6 resolves only
    4.7e-10 d (40 microseconds), so the two-part path must beat 1e-12 d
    (floating-point basis: 0.125 and the small offset are exact in binary
    to that level).
    """
    epoch = Epoch(2460000.0, 0.125)
    small = 3.0e-9
    t = _TwoPart(np.array([2460000.0, 2460010.0]), np.array([0.125 + small, 0.5]))
    np.testing.assert_allclose(epoch.days(t), [small, 10.375], rtol=0, atol=1e-12)
    jd = np.array([2460000.125, 2467305.375])
    out = epoch.days(jd)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0.0, 7305.25])
    np.testing.assert_array_equal(epoch.jd(out), jd)


def test_float32_offsets_resolve_seconds_where_absolute_dates_do_not():
    """Fails if offsets lose the resolution that motivates them.

    Truth: float32 spacing. Near 2.46e6 d it is 0.25 d; near 7305 d
    (20 years) it is 4.9e-4 d, or 42 s. Basis: floating-point.
    """
    epoch = Epoch(2460000.0)
    jd = 2460000.0 + 7305.0 + 100.0 / 86400.0
    assert abs(float(np.float32(jd)) - jd) > 1e-3
    offset = epoch.days(jd)
    assert abs(float(np.float32(offset)) - offset) < 4.9e-4


def test_static_metadata_scale_checks_and_unit_changes():
    """Fails if the epoch is mutable, unhashable, or lets time scales mix."""
    epoch = Epoch(2460000, 0.5, "tt")
    assert type(epoch.jd1) is float
    assert hash(epoch) == hash(Epoch(2460000.0, 0.5, "tt"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        epoch.jd1 = 0.0
    with pytest.raises(ValueError, match="scale"):
        Epoch(scale="gps")
    with pytest.raises(ValueError, match="scale"):
        epoch.days(_TwoPart(2460000.0, 0.5, "utc"))
    with pytest.raises(ValueError, match="scale"):
        epoch.days_between(Epoch(scale="tdb"))
    later = Epoch(2460010.0, 0.75, "tt")
    assert later.days_between(epoch) == 10.25
    t_later = 3.0
    assert epoch.days(later.jd(t_later)) == t_later + later.days_between(epoch)
    assert days_to_seconds(0.5) == 43200.0
    assert seconds_to_days(43200.0) == 0.5
