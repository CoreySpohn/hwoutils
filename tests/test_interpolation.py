"""Piecewise-linear table values, weights, and working precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.interpolation import interp_linear
from hwoutils.testing import assert_precision_closed, ulps


@pytest.fixture
def table(precision):
    # Slopes 4 and -2 on unequal intervals.
    return tuple(
        jnp.asarray(values, dtype=precision)
        for values in ([-1, 0, 0.5, 2, 3, 6, 7], [0, 2, 6], [2, 10, 2])
    )


@pytest.mark.parametrize(
    ("left", "right", "ends"),
    [(None, None, [2, 2]), (-5.0, 13.0, [-5, 13]), ("extrapolate", None, [-2, 2])],
)
def test_table_values(table, left, right, ends):
    x, xp, fp = table
    expected = [ends[0], 2, 4, 10, 8, 2, ends[1]]
    np.testing.assert_array_equal(interp_linear(x, xp, fp, left, right), expected)


@pytest.mark.parametrize(
    ("table_kind", "fills"),
    [("real", (None, None)), ("complex", (-2.5, 1.25)), ("weak", (None, None))],
)
def test_jax_compatibility(precision, table_kind, fills):
    xp = jnp.asarray([0.5, 3.25, 7.75], dtype=precision)
    x = jnp.asarray([-1, 0.5, 1.45, 3.25, 5.35, 7.75, 8], dtype=precision)
    fp = jnp.asarray([1.27, 3.78, 2.11], dtype=precision)
    if table_kind == "complex":
        fp = fp + 1j * fp[::-1]
    elif table_kind == "weak":
        xp = xp.astype(jnp.float32)
        x = x.astype(jnp.float32)
        fp = jnp.full(fp.shape, 3.78)
    actual = interp_linear(x, xp, fp, *fills)
    reference = jnp.interp(x, xp, fp, *fills)
    assert actual.dtype == reference.dtype
    if jax.__version_info__ < (0, 12, 1):
        assert np.asarray(actual).tobytes() == np.asarray(reference).tobytes()
    else:
        # Allow two rounding operations for differences in the compiled formula.
        assert jnp.all(jnp.abs(actual - reference) <= ulps(2, reference))


@pytest.mark.parametrize("fill", [None, 0.0])
def test_table_transpose(table, fill):
    x, xp, fp = table
    edge = 1 if fill is None else 0
    weights = [
        [edge, 0, 0],
        [1, 0, 0],
        [0.75, 0.25, 0],
        [0, 1, 0],
        [0, 0.75, 0.25],
        [0, 0, 1],
        [0, 0, edge],
    ]
    transpose = jax.linear_transpose(lambda f: interp_linear(x, xp, f, fill, fill), fp)
    (rows,) = jax.vmap(transpose)(jnp.eye(x.size, dtype=x.dtype))
    np.testing.assert_array_equal(rows, weights)


@pytest.mark.float64_only(reason="Detecting wide intermediates requires x64")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_precision_closure(dtype):
    x = jax.ShapeDtypeStruct((2,), jnp.float32)
    fp = jax.ShapeDtypeStruct((2,), dtype)
    assert_precision_closed(
        lambda q, p, f: interp_linear(q, p, f, np.nan, 0.0), x, x, fp
    )


@pytest.mark.parametrize("complex_table", [False, True])
def test_jit_and_vmap(table, complex_table):
    x, xp, fp = table
    if complex_table:
        fp = fp + 1j * fp[::-1] * 2
    eager = interp_linear(x, xp, fp)
    np.testing.assert_array_equal(jax.jit(interp_linear)(x, xp, fp), eager)
    np.testing.assert_array_equal(
        jax.vmap(interp_linear, (0, None, None))(x, xp, fp), eager
    )
    rows = jax.vmap(interp_linear, (None, None, 0))(x, xp, jnp.stack([fp, -fp]))
    np.testing.assert_array_equal(rows, jnp.stack([eager, -eager]))
    if complex_table:
        np.testing.assert_array_equal(eager.imag, [4, 4, 8, 20, 16, 4, 4])


def test_gradients(table):
    x, xp, fp = table
    dx, df = jax.grad(lambda q, f: interp_linear(q, xp, f).sum(), (0, 1))(x, fp)
    np.testing.assert_array_equal(dx, [0, 4, 4, -2, -2, -2, 0])
    np.testing.assert_array_equal(df, [2.75, 2, 2.25])


@pytest.mark.parametrize("singleton", [False, True])
def test_degenerate_interval(precision, singleton):
    width = np.spacing(np.finfo(precision).eps)
    xp = jnp.asarray([0] if singleton else [0, width], dtype=precision)
    fp = jnp.asarray([2] if singleton else [2, 6], dtype=jnp.int16)
    x = jnp.asarray([-1, 0, width / 2, width, 1], dtype=precision)
    result = interp_linear(x, xp, fp)
    assert result.dtype == precision
    np.testing.assert_array_equal(result, [2, 2, 2, 2, 2 if singleton else 6])
