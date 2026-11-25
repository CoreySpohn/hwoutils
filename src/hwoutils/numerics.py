"""Floating-point error bounds and host comparisons for structural decisions."""

from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hwoutils.leaves import is_traced
from hwoutils.precision import float_dtype, precision_of, require_float64

COARSEST = np.float32
"""Coarsest precision supported by structural comparisons."""


def _working_dtype(*values, dtype=None):
    """Choose the least precise floating operand, ignoring integer counts."""
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype == np.float64:
            require_float64("An explicit float64 tolerance")
        return dtype
    dtypes = (jnp.asarray(x).dtype for x in values)
    real = (jnp.finfo(d).dtype for d in dtypes if jnp.issubdtype(d, jnp.inexact))
    return min(real, key=lambda d: d.itemsize, default=float_dtype())


def eps(x_or_dtype) -> float:
    """Return the relative spacing at one for a value or dtype.

    Args:
        x_or_dtype: A floating or complex value or dtype.

    Returns:
        Machine epsilon for the corresponding real dtype.
    """
    if isinstance(x_or_dtype, str | type | np.dtype):
        dtype = np.dtype(x_or_dtype)
    else:
        dtype = precision_of(x_or_dtype)
    return float(jnp.finfo(dtype).eps)


def ulp(x, dtype=None) -> jax.Array:
    """Return the spacing at the magnitude of each input.

    Args:
        x: Input value or array.
        dtype: Real precision in which to measure spacing. Defaults to that of x.

    Returns:
        Distance to the next larger representable magnitude.

    Raises:
        PrecisionError: If the requested precision is unavailable.
    """
    dtype = _working_dtype(x, dtype=dtype)
    value = jnp.asarray(x)
    if jnp.iscomplexobj(value):
        value = jnp.abs(value)
    magnitude = jnp.abs(value.astype(dtype))
    # XLA can flush the subtraction inside spacing() when the gap is subnormal.
    # In that range the gap is 2**(biased_exponent - 1) subnormal units,
    # with a minimum of one unit for zero and subnormal inputs.
    mantissa_bits = jnp.finfo(dtype).nmant
    unsigned = np.dtype(f"u{dtype.itemsize}")
    exponent = magnitude.view(unsigned) >> mantissa_bits
    gap_bits = jnp.left_shift(jnp.ones_like(exponent), jnp.maximum(exponent, 1) - 1)
    return jnp.where(
        exponent <= mantissa_bits, gap_bits.view(dtype), jnp.spacing(magnitude)
    )


def roundoff(scale, *, n_ops=1, dtype=None):
    """Bound the absolute roundoff at a given scale.

    Args:
        scale: Nonnegative magnitude in the units of the result.
        n_ops: Number of rounding operations along a computation path.
        dtype: Precision for the allowance. Defaults to the coarser input precision.

    Returns:
        The scale times machine epsilon times the operation count.

    Raises:
        PrecisionError: If the requested precision is unavailable.
    """
    dtype = _working_dtype(scale, n_ops, dtype=dtype)
    return (
        jnp.asarray(n_ops, dtype=dtype) * eps(dtype) * jnp.asarray(scale, dtype=dtype)
    )


def sum_roundoff(scale, n_terms, *, dtype=None):
    """Bound roundoff along the longest path of a balanced pairwise sum.

    Args:
        scale: Nonnegative magnitude in the units of the sum.
        n_terms: Integer number of terms.
        dtype: Precision for the allowance. Defaults to the precision of the scale.

    Returns:
        Roundoff for ceil(log2(n_terms)) additions, zero for zero or one term.

    Raises:
        PrecisionError: If the requested precision is unavailable.
    """
    count = jnp.asarray(n_terms)
    # The bit length of n - 1 is exact even just above a large power of two.
    bits = jnp.iinfo(count.dtype).bits
    depth = bits - jax.lax.clz(jnp.maximum(count, 1) - 1)
    return roundoff(scale, n_ops=depth, dtype=dtype)


def isclose(a, b, *, ulps=4, scale=None):
    """Compare two values relative to their magnitudes and an optional scale.

    Args:
        a: Measured value.
        b: Expected value, broadcast against a.
        ulps: Number of machine epsilons allowed at the comparison scale.
        scale: Nonnegative reference magnitude. Supply it when b can be zero.

    Returns:
        Elementwise comparison at the coarser input precision.

    Raises:
        ValueError: If a concrete expected value is zero and scale is omitted.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    magnitude = jnp.maximum(jnp.abs(a), jnp.abs(b))
    if scale is None:
        message = "Supply scale when the expected value can be zero."
        if is_traced(b):
            b = eqx.error_if(b, jnp.any(b == 0), message)
        elif np.any(np.asarray(b) == 0):
            raise ValueError(message)
        dtype = _working_dtype(a, b)
    else:
        dtype = _working_dtype(a, b, scale)
        magnitude = jnp.maximum(magnitude, jnp.asarray(scale))
    slack = ulps * eps(dtype) * magnitude
    return (a == b) | (jnp.isfinite(magnitude) & (jnp.abs(a - b) <= slack))


def within(x, low, high, *, ulps=4):
    """Test inclusive bounds with roundoff slack at the interval's scale.

    Args:
        x: Value to test.
        low: Lower endpoint.
        high: Upper endpoint.
        ulps: Number of machine epsilons allowed at the comparison scale.

    Returns:
        Elementwise boolean JAX array, including both endpoints.
    """
    x, low, high = jnp.asarray(x), jnp.asarray(low), jnp.asarray(high)
    magnitude = jnp.maximum(
        jnp.abs(x),
        jnp.maximum(
            jnp.where(jnp.isfinite(low), jnp.abs(low), 0),
            jnp.where(jnp.isfinite(high), jnp.abs(high), 0),
        ),
    )
    slack = (
        ulps
        * eps(_working_dtype(x, low, high))
        * jnp.where(jnp.isfinite(x), magnitude, 0)
    )
    return (low <= high) & (low - slack <= x) & (x <= high + slack)


def same_tolerance(a, *, ulps=4, floor=0.0):
    """Return the host merge threshold at the magnitude of each value.

    Args:
        a: Real scalar or array.
        ulps: Number of spacings in the coarsest supported precision.
        floor: Minimum absolute threshold in the units of the input.

    Returns:
        Float64 tolerance with the shape of the input.
    """
    magnitude = np.abs(np.asarray(a, dtype=np.float64)).astype(COARSEST)
    spacing = np.spacing(magnitude).astype(np.float64)
    return np.maximum(ulps * spacing, floor)


@overload
def same(a: float, b: float, *, ulps=4, floor=0.0) -> bool: ...


@overload
def same(a: np.ndarray, b, *, ulps=4, floor=0.0) -> bool | np.ndarray: ...


@overload
def same(a, b: np.ndarray, *, ulps=4, floor=0.0) -> bool | np.ndarray: ...


def same(a, b, *, ulps=4, floor=0.0):
    """Compare host values at the resolution of the coarsest supported float.

    Args:
        a: First real value.
        b: Second real value.
        ulps: Spacings at the larger magnitude giving the strict merge threshold.
        floor: Minimum absolute merge threshold in the units of the operands.

    Returns:
        A bool for scalars, or a boolean array for broadcast operands.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    tolerance = same_tolerance(np.maximum(np.abs(a), np.abs(b)), ulps=ulps, floor=floor)
    result = (a == b) | (np.abs(a - b) < tolerance)
    return bool(result) if result.ndim == 0 else result


def is_uniform(axis, *, ulps=8) -> bool:
    """Test host axis increments at the roundoff scale of its coordinates.

    Args:
        axis: One-dimensional real coordinates.
        ulps: Machine epsilons of the coarsest precision at the largest coordinate.

    Returns:
        Whether the coordinates are finite and the steps agree within roundoff.

    Raises:
        ValueError: If the coordinates are not one-dimensional.
    """
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("axis must be one-dimensional")
    if not np.all(np.isfinite(values)):
        return False
    if values.size <= 2:
        return True
    steps = np.diff(values)
    tolerance = ulps * eps(COARSEST) * np.max(np.abs(values))
    return bool(np.all(np.abs(steps - steps[0]) <= tolerance))
