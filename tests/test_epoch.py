"""Tests for the reference epoch and day offsets."""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import PrecisionError, elapsed_seconds
from hwoutils.constants import d2s
from hwoutils.epoch import Epoch, days_to_seconds, seconds_to_days
from hwoutils.testing import ulps


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
    out = epoch.days(jd=jd)
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
    offset = epoch.days(jd=jd)
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
    assert epoch.days(jd=later.jd(t_later)) == t_later + later.days_between(epoch)
    assert days_to_seconds(0.5) == 43200.0
    assert seconds_to_days(43200.0) == 0.5


@pytest.mark.parametrize(
    ("epoch", "mjd", "expected"),
    [
        (Epoch(2461000.5), 61000.0, 0.0),
        (Epoch(2461000.0, 0.5), 61037.0, 37.0),
        (Epoch(2461000.5, 0.25), 61000.75, 0.5),
    ],
)
def test_mjd_and_jd_name_the_same_instant_exactly(epoch, mjd, expected):
    """Fails if the MJD dialect is offset by 2400000.5 d or loses bits.

    Truth: exact binary arithmetic. Every value is a multiple of 0.25 d, so
    both dialects must agree with the expected offset to exactly 0.0.
    """
    assert epoch.days(mjd=mjd) == expected
    assert epoch.days(jd=mjd + 2400000.5) == expected


def test_mjd_offset_keeps_astropy_precision():
    """Fails if the MJD path subtracts the epoch after summing the parts.

    Truth: astropy Time two-part subtraction. The epoch jd2 and the date carry
    a fractional day; the MJD spacing near 6.1e4 d is 7.3e-12 d, so 1e-9 d
    (86 microseconds) is far above the float64 floor of the date itself.
    """
    time = pytest.importorskip("astropy.time")
    epoch = Epoch(2461000.5, 0.3125, "tdb")
    mjd = 61012.3456789012
    stamp = time.Time(mjd, format="mjd", scale="tdb")
    truth = (stamp.jd1 - epoch.jd1) + (stamp.jd2 - epoch.jd2)
    np.testing.assert_allclose(epoch.days(mjd=mjd), truth, rtol=0, atol=1e-9)
    np.testing.assert_allclose(epoch.days(stamp), truth, rtol=0, atol=1e-12)
    out = epoch.days(mjd=np.array([mjd, mjd + 1.0]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(np.diff(out), 1.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("bare", [61000.0, np.array([61000.0, 61001.0]), [2461000.5]])
def test_bare_number_is_refused_positionally(bare):
    """Fails if a bare number is accepted without naming JD or MJD."""
    with pytest.raises(TypeError, match="jd= or mjd="):
        Epoch(2461000.5).days(bare)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"jd": 2461000.5, "mjd": 61000.0}],
)
def test_exactly_one_date_argument(kwargs):
    """Fails if zero or two date arguments are accepted."""
    with pytest.raises(ValueError, match="exactly one"):
        Epoch(2461000.5).days(**kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        Epoch(2461000.5).days(_TwoPart(2461000.5, 0.0), jd=2461000.5)


def test_astropy_time_still_works_and_scale_mismatch_raises():
    """Fails if astropy Time is rejected or a scale mismatch passes."""
    time = pytest.importorskip("astropy.time")
    epoch = Epoch(2461000.5, 0.0, "tdb")
    stamp = time.Time(61001.0, format="mjd", scale="tdb")
    np.testing.assert_allclose(epoch.days(stamp), 1.0, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="scale"):
        epoch.days(time.Time(61001.0, format="mjd", scale="utc"))


_CLOCK_ORIGIN_D = 2461000.5
_CLOCK_OFFSETS = (-3, 0, 7)


def _clock_times():
    return np.array([_CLOCK_ORIGIN_D + k * 2**-20 for k in _CLOCK_OFFSETS])


def _clock_seconds(precision):
    return np.array([k * 2**-20 * 86400 for k in _CLOCK_OFFSETS], dtype=precision)


def test_elapsed_seconds_conditions_host_times(precision):
    """Fails if the host times are narrowed before subtraction or units change.

    Truth: the exactly representable binary day offsets times 86400. The
    float64 answer is exact and the float32 storage allows one ULP.
    """
    expected = _clock_seconds(precision)
    actual = elapsed_seconds(_clock_times(), origin_d=_CLOCK_ORIGIN_D)
    assert actual.dtype == precision
    assert not actual.weak_type
    if precision == np.float64:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.all(np.abs(actual - expected) <= ulps(1, expected))


def test_elapsed_seconds_zero_origin_precision(precision):
    """Fails if explicit width is lost or weak scalars ignore their owner.

    Truth: exact dtype equality to the input or owner, or the active lane.
    """
    explicit = jnp.asarray(1.0, dtype=np.float32)
    active = jnp.asarray(1.0, dtype=precision)
    assert elapsed_seconds(explicit, origin_d=0.0).dtype == np.float32
    assert elapsed_seconds(np.float32(1.0), origin_d=0.0).dtype == np.float32
    swapped = np.asarray(1.0, dtype=np.dtype(np.float32).newbyteorder("S"))
    assert elapsed_seconds(swapped, origin_d=0.0).dtype == np.float32
    for reference, expected in (
        ({"value": explicit}, np.float32),
        (active, precision),
        (None, precision),
    ):
        result = elapsed_seconds(1.0, origin_d=0.0, reference=reference)
        assert result.dtype == expected
        assert not result.weak_type
    np.testing.assert_array_equal(
        elapsed_seconds([1.0, 2.0], origin_d=0.0), [86400, 172800]
    )


@pytest.mark.parametrize(
    "transform",
    [None, jax.jit, jax.vmap, jax.grad],
    ids=["eager", "jit", "vmap", "grad"],
)
def test_elapsed_seconds_refuses_narrow_jax_time(transform):
    """Fails if float32 data can reach the absolute subtraction.

    Truth: the refusal contract. Under jit the exception is raised while
    lowering, before compilation or execution.
    """
    convert = partial(elapsed_seconds, origin_d=_CLOCK_ORIGIN_D)
    time = jnp.asarray([1.0] if transform is jax.vmap else 1.0, dtype=np.float32)
    with pytest.raises(PrecisionError, match="elapsed_seconds formed on the host"):
        if transform is jax.jit:
            jax.jit(convert).lower(time)
        elif transform is None:
            convert(time)
        else:
            transform(convert)(time)


def test_elapsed_seconds_refuses_narrow_and_complex_host_times():
    """Fails if an unsupported host dtype reaches absolute subtraction.

    Truth: narrower floating and complex inputs raise their documented types.
    """
    for narrow in (np.float32(1.0), np.float16(1.0)):
        with pytest.raises(PrecisionError, match="elapsed_seconds"):
            elapsed_seconds(narrow, origin_d=_CLOCK_ORIGIN_D)
    with pytest.raises(ValueError, match="time_d"):
        elapsed_seconds(np.complex64(1j), origin_d=_CLOCK_ORIGIN_D)


@pytest.mark.float64_only
def test_elapsed_seconds_float64_paths_agree(precision):
    """Fails if the device or traced clock disagrees with exact day offsets.

    Truth: exact algebra on binary offsets, with a one-ULP allowance.
    """
    times = _clock_times()
    expected = _clock_seconds(precision)
    convert = partial(elapsed_seconds, origin_d=_CLOCK_ORIGIN_D)
    for result in (
        convert(times),
        convert(jnp.asarray(times)),
        jax.jit(convert)(times),
    ):
        assert result.dtype == precision
        assert np.all(np.abs(result - expected) <= ulps(1, expected))


@pytest.mark.float32_only
def test_elapsed_seconds_float64_trace_requires_x64():
    """Fails if a float64 traced request runs with x64 unavailable.

    Truth: the refusal contract, even when concrete host subtraction is valid.
    """
    with jax.enable_x64():
        time = jnp.asarray(_clock_times())
    convert = partial(elapsed_seconds, origin_d=_CLOCK_ORIGIN_D)
    with pytest.raises(PrecisionError, match=r"pass elapsed_seconds.*Enable x64"):
        jax.jit(convert).lower(time)


def test_elapsed_seconds_requires_static_python_float_origin():
    """Fails if non-Python-float origins are accepted.

    Truth: the static-origin type contract, including under tracing.
    """
    for origin in (np.float64(0.0), 0, jnp.asarray(0.0)):
        with pytest.raises(TypeError, match="static host float"):
            elapsed_seconds(1.0, origin_d=origin)
    with pytest.raises(TypeError, match="static host float"):
        jax.jit(lambda origin: elapsed_seconds(1.0, origin_d=origin)).lower(0.0)


@pytest.mark.float64_only
def test_elapsed_seconds_traced_gradient(precision):
    """Fails if changing units or tracing changes the clock derivative.

    Truth: the derivative of the binary64 subtraction times 86400, to one ULP.
    """
    convert = partial(elapsed_seconds, origin_d=_CLOCK_ORIGIN_D)
    time = jnp.asarray(_CLOCK_ORIGIN_D, dtype=precision)
    expected = jnp.asarray(d2s, dtype=precision)
    actual = jax.grad(convert)(time)
    assert actual.dtype == precision
    assert np.all(np.abs(actual - expected) <= ulps(1, expected))
