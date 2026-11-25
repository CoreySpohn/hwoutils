"""Precision follows the floating leaves of an owner."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hwoutils


@pytest.mark.parametrize("x64", [False, True])
def test_owner_precision(x64):
    """Fails if a complex partner, index, or metadata changes owner precision."""
    with jax.enable_x64(x64):
        real = np.float64 if x64 else np.float32
        partner = np.complex128 if x64 else np.complex64
        tree = {
            "field": jnp.ones(2, partner),
            "index": jnp.int32(1),
            "metadata": ("a", np.float32),
        }
        assert hwoutils.float_dtype() == real
        assert hwoutils.precision_of(tree) == real
        assert hwoutils.complex_of(real) == partner
        tree["other"] = np.ones(1, np.float32 if x64 else np.float64)
        with pytest.raises(hwoutils.PrecisionError, match="canonical"):
            hwoutils.precision_of(tree)


@pytest.mark.parametrize("x64", [False, True])
def test_like_follows_reference(x64):
    """Fails if host constants widen a narrow owner or lose the imaginary part."""
    with jax.enable_x64(x64):
        if not x64:
            with pytest.raises(hwoutils.PrecisionError, match="x64"):
                hwoutils.like(1, np.float64(1))
        for real in [np.float32, np.float64] if x64 else [np.float32]:
            ref = jnp.ones(2, real)
            c = jax.jit(hwoutils.like)(2 + 3j, ref)
            r = hwoutils.like([1, 2], c)
            assert c.dtype == (np.complex64 if real == np.float32 else np.complex128)
            assert r.dtype == real
            assert c == 2 + 3j
            np.testing.assert_array_equal(r, [1, 2])


@pytest.mark.parametrize("x64", [False, True])
def test_float64_refusal(x64):
    """Fails if an unavailable or narrow owner passes a float64-only entry."""
    with jax.enable_x64(x64):
        wide = np.ones(1, np.float64)
        if x64:
            hwoutils.require_float64("curvature", wide)
            hwoutils.require_float64("curvature")
            remedy = "canonical"
        else:
            remedy = "x64"
        with pytest.raises(hwoutils.PrecisionError, match=f"curvature.*{remedy}"):
            hwoutils.require_float64("curvature", wide, jnp.float32(2))


def test_configuration_compatibility():
    """Fails if either re-export loses the process setting or environment fallback."""
    script = """
import os
from hwoutils import precision, jax_config
assert precision.enable_x64 is jax_config.enable_x64
assert precision.float_dtype is jax_config.float_dtype
for flag in (True, False):
    jax_config.enable_x64(flag)
    assert precision.float_dtype().itemsize == (8 if flag else 4)
os.environ["JAX_ENABLE_X64"] = "1"
jax_config.enable_x64(False)
assert precision.float_dtype().itemsize == 8
"""
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
