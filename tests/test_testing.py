"""Check the numerical testing helpers on small traceable functions."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.numerics import roundoff
from hwoutils.testing import assert_precision_closed, precision_pair, ulps


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ulps_matches_roundoff(dtype):
    """Fails if ulps loses the input dtype or rounds at the wrong scale."""
    with jax.enable_x64():
        like = jnp.asarray([-2, 4], dtype=dtype)
        allowance = ulps(3, like)
        assert allowance.dtype == dtype
        np.testing.assert_array_equal(allowance, roundoff(jnp.abs(like), n_ops=3))


@pytest.mark.parametrize(
    ("narrow_dtype", "wide_dtype"),
    [(np.float32, np.float64), (np.complex64, np.complex128)],
)
def test_precision_closed_checks_nested_intermediates(narrow_dtype, wide_dtype):
    """Fails if the walk misses a wide intermediate or loses its source line."""

    @jax.jit
    def inner(x):
        wide = x.astype(wide_dtype)
        return wide.astype(x.dtype)

    with jax.enable_x64():
        x = jax.ShapeDtypeStruct((), narrow_dtype)
        assert_precision_closed(lambda v: v * 2, x)
        line = inspect.getsourcelines(inner)[1] + 2
        dtype = np.dtype(wide_dtype).name
        with pytest.raises(AssertionError, match=rf"test_testing.py:{line}\b.*{dtype}"):
            assert_precision_closed(lambda v: inner(v), x)


def test_precision_closed_checks_equation_free_output():
    """Fails if wide array outputs escape or static metadata is flagged."""
    x = jax.ShapeDtypeStruct((), np.float32)
    with jax.enable_x64():
        assert_precision_closed(lambda v: (v, {"scale": 1.0, "phase": 1j}), x)
        wide = jnp.asarray(1.0)
        with pytest.raises(AssertionError, match="float64"):
            assert_precision_closed(lambda v: wide, x)


@pytest.mark.parametrize("as_array", [True, False])
def test_precision_pair_enforces_relative_bound(as_array):
    """Fails if both evaluations use one precision or the bound is absolute."""
    with jax.enable_x64():
        original = 1 + 2**-25
        if as_array:
            original = jnp.asarray(original, dtype=np.float64)
        args = ({"value": original},)
        precision_pair(lambda tree: tree["value"] * 1024, args, bound=2**-24)
        with pytest.raises(AssertionError, match="relative"):
            precision_pair(lambda tree: tree["value"] * 1024, args, bound=2**-26)


def test_precision_pair_uses_a_scale_for_near_zeros():
    """Float32 sine at a null is compared at the scale of its output leaf."""
    bound = 2**-23
    with jax.enable_x64():
        null = jnp.asarray(jnp.pi)
        samples = jnp.stack((null, null / 2))
        precision_pair(jnp.sin, (samples,), bound=bound)
        precision_pair(jnp.sin, (null,), bound=bound, scale=1.0)
        with pytest.raises(AssertionError, match="relative"):
            precision_pair(jnp.sin, (null,), bound=bound)
        with pytest.raises(AssertionError, match=r"\['null'\].*relative"):
            precision_pair(
                lambda x: {"null": jnp.sin(x[0]), "peak": jnp.sin(x[1])},
                (samples,),
                bound=bound,
            )
