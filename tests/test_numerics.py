"""Roundoff bounds and precision-independent layout decisions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import numerics


def test_roundoff_depth_and_spacing(precision):
    """Fails if epsilon replaces spacing or a pairwise sum uses a linear bound."""
    dtype = precision.type
    scale = dtype(8)
    unit = np.spacing(scale)
    assert numerics.eps(dtype) == np.finfo(dtype).eps
    assert numerics.eps(jnp.asarray(scale) * 1j) == np.finfo(dtype).eps
    tiny = dtype(8) * np.finfo(dtype).tiny
    np.testing.assert_array_equal(jax.jit(numerics.ulp)(tiny), np.spacing(tiny))
    assert numerics.ulp(-scale) == unit
    assert numerics.ulp(-scale, np.float32) == np.spacing(np.float32(8))
    assert numerics.roundoff(scale, n_ops=3) == 3 * unit
    assert numerics.sum_roundoff(scale, 5) == 3 * unit
    assert numerics.sum_roundoff(scale, 8) == 3 * unit
    assert numerics.sum_roundoff(scale, 1) == 0
    assert numerics.roundoff(scale, dtype=np.float32) == np.spacing(np.float32(8))


def test_comparison_slack():
    """Fails if inclusive endpoints, their roundoff slack, or a zero scale is lost."""
    ref = jnp.asarray(1.0)
    unit = np.finfo(ref.dtype).eps
    offsets = jnp.asarray([0, 4, 8]) * unit
    np.testing.assert_array_equal(numerics.isclose(1 + offsets, ref), [1, 1, 0])
    with pytest.raises(ValueError, match="scale"):
        numerics.isclose(ref, 0)
    near_zero = jax.jit(lambda x: numerics.isclose(x, 0, scale=ref))
    np.testing.assert_array_equal(near_zero(offsets), [1, 1, 0])
    bounded = jax.jit(numerics.within)
    for sign in (-1, 1):
        np.testing.assert_array_equal(
            bounded(sign * (1 + offsets), -ref, ref), [1, 1, 0]
        )
    wide = np.float64(1 + 3 * np.finfo(np.float32).eps)
    assert numerics.isclose(wide, np.float32(1))
    assert numerics.within(wide, np.float32(0), np.float64(1))


def test_host_structure():
    """Fails if host layout changes with the flag or resolves subthreshold steps."""
    assert numerics.COARSEST is np.float32
    wave = 600.0
    step = float(np.spacing(np.float32(wave)))
    assert numerics.same(wave, wave + 2 * step) is True
    assert numerics.same(wave, wave + 1e-3) is False
    assert not numerics.same(wave, wave + 4 * step)
    assert numerics.same(0.0, 0.5e-9, floor=1e-9)
    axis = np.linspace(600, 601, 21, dtype=np.float32)
    assert numerics.is_uniform(axis) is True
    axis[10] += 32 * step
    assert numerics.is_uniform(axis) is False


def test_traced_comparison_requires_scale_at_zero():
    """A zero in a runtime operand triggers the scale diagnostic."""
    compare = eqx.filter_jit(numerics.isclose)
    assert compare(jnp.float32(1), jnp.float32(1))
    with pytest.raises(eqx.EquinoxRuntimeError, match="Supply scale"):
        compare(jnp.float32(1), jnp.float32(0)).block_until_ready()


def test_within_unbounded_and_inverted_intervals():
    """An infinite endpoint leaves the opposite endpoint effective."""
    x = [-2.0, 0.0, 2.0]
    np.testing.assert_array_equal(numerics.within(x, -np.inf, 1), [True, True, False])
    np.testing.assert_array_equal(numerics.within(x, -1, np.inf), [False, True, True])
    assert not numerics.within(1.0, 1.0 + np.finfo(np.float32).eps, 1.0)


def test_same_broadcasts_at_the_larger_magnitude():
    """Each pair uses float32 spacing at its larger absolute operand."""
    unit = float(np.spacing(np.float32(1.0)))
    a = np.array([[1.0], [-1.0]])
    b = np.array([1 - 3 * unit, 1 + 4 * unit, -1 + 3 * unit])
    np.testing.assert_array_equal(
        numerics.same(a, b), [[True, False, False], [False, False, True]]
    )
    threshold = numerics.same_tolerance([-1.0, 0.0], ulps=3, floor=1e-9)
    assert threshold.dtype == np.float64
    np.testing.assert_array_equal(threshold, [3 * unit, 1e-9])


def test_uniform_axis_degenerate_and_invalid_inputs():
    """Short finite axes are uniform and non-finite coordinates are refused."""
    for axis in ([], [1], [3, 1]):
        assert numerics.is_uniform(axis)
    for value in (np.nan, np.inf, -np.inf):
        assert not numerics.is_uniform([value])
    with pytest.raises(ValueError, match="one-dimensional"):
        numerics.is_uniform([[1, 2]])
