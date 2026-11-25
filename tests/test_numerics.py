"""Roundoff bounds and precision-independent layout decisions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hwoutils


@pytest.mark.parametrize("x64", [False, True])
def test_roundoff_depth_and_spacing(x64):
    """Fails if epsilon replaces spacing or a pairwise sum uses a linear bound."""
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        scale = dtype(8)
        unit = np.spacing(scale)
        assert hwoutils.eps(dtype) == np.finfo(dtype).eps
        assert hwoutils.eps(jnp.asarray(scale) * 1j) == np.finfo(dtype).eps
        tiny = dtype(8) * np.finfo(dtype).tiny
        np.testing.assert_array_equal(jax.jit(hwoutils.ulp)(tiny), np.spacing(tiny))
        assert hwoutils.ulp(-scale) == unit
        assert hwoutils.ulp(-scale, np.float32) == np.spacing(np.float32(8))
        assert hwoutils.roundoff(scale, n_ops=3) == 3 * unit
        assert hwoutils.sum_roundoff(scale, 5) == 3 * unit
        assert hwoutils.sum_roundoff(scale, 8) == 3 * unit
        assert hwoutils.sum_roundoff(scale, 1) == 0
        assert hwoutils.roundoff(scale, dtype=np.float32) == np.spacing(np.float32(8))


@pytest.mark.parametrize("x64", [False, True])
def test_comparison_slack(x64):
    """Fails if inclusive endpoints, their roundoff slack, or a zero scale is lost."""
    with jax.enable_x64(x64):
        ref = jnp.asarray(1.0)
        unit = np.finfo(ref.dtype).eps
        offsets = jnp.asarray([0, 4, 8]) * unit
        np.testing.assert_array_equal(hwoutils.isclose(1 + offsets, ref), [1, 1, 0])
        with pytest.raises(ValueError, match="scale"):
            hwoutils.isclose(ref, 0)
        near_zero = jax.jit(lambda x: hwoutils.isclose(x, 0, scale=ref))
        np.testing.assert_array_equal(near_zero(offsets), [1, 1, 0])
        bounded = jax.jit(hwoutils.within)
        for sign in (-1, 1):
            np.testing.assert_array_equal(
                bounded(sign * (1 + offsets), -ref, ref), [1, 1, 0]
            )
        wide = np.float64(1 + 3 * np.finfo(np.float32).eps)
        assert hwoutils.isclose(wide, np.float32(1))
        assert hwoutils.within(wide, np.float32(0), np.float64(1))


@pytest.mark.parametrize("x64", [False, True])
def test_host_structure(x64):
    """Fails if host layout changes with the flag or resolves subthreshold steps."""
    with jax.enable_x64(x64):
        assert hwoutils.COARSEST is np.float32
        wave = 600.0
        step = float(np.spacing(np.float32(wave)))
        assert hwoutils.same(wave, wave + 2 * step) is True
        assert hwoutils.same(wave, wave + 1e-3) is False
        assert not hwoutils.same(wave, wave + 4 * step)
        assert hwoutils.same(0.0, 0.5e-9, floor=1e-9)
        axis = np.linspace(600, 601, 21, dtype=np.float32)
        assert hwoutils.is_uniform(axis) is True
        axis[10] += 32 * step
        assert hwoutils.is_uniform(axis) is False
