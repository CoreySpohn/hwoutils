"""Numerical assertions for tests of dtype-closed computations."""

from collections.abc import Callable, Iterator
from inspect import signature, unwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.extend.source_info_util import SourceInfo, summarize

from hwoutils.leaves import canonical
from hwoutils.numerics import roundoff
from hwoutils.precision import complex_of, require_float64

__all__ = ["assert_precision_closed", "precision_pair", "ulps"]

_WIDE_DTYPES = {np.dtype("float64"), np.dtype("complex128")}


def ulps(n: int, like: jax.typing.ArrayLike) -> jax.Array:
    """Allow n rounding operations at the magnitude and precision of a value.

    Args:
        n: Number of rounding operations to allow.
        like: Scalar or array setting the scale and dtype.

    Returns:
        Absolute roundoff bound from :func:`hwoutils.numerics.roundoff`.
    """
    return roundoff(jnp.abs(like), n_ops=n)


def _subjaxprs(value) -> Iterator[Jaxpr]:
    if isinstance(value, ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, Jaxpr):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _subjaxprs(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _subjaxprs(child)


def _wide_dtype(variables) -> np.dtype | None:
    for var in variables:
        dtype = getattr(var.aval, "dtype", None)
        if dtype in _WIDE_DTYPES:
            return dtype
    return None


def _first_wide(jaxpr: Jaxpr) -> tuple[SourceInfo | None, np.dtype] | None:
    for eqn in jaxpr.eqns:
        for child in _subjaxprs(eqn.params):
            if hit := _first_wide(child):
                return hit[0] or eqn.source_info, hit[1]
        if dtype := _wide_dtype((*eqn.invars, *eqn.outvars)):
            return eqn.source_info, dtype
    # A returned literal or captured constant can bypass all equations.
    if dtype := _wide_dtype(jaxpr.outvars):
        return None, dtype
    return None


def assert_precision_closed(fn: Callable, *args) -> None:
    """Trace a function on float32 inputs and reject float64 or complex128 values.

    Args:
        fn: JAX-traceable computation with scalar, array, or pytree outputs.
        *args: Example inputs with inexact array leaves in float32 or complex64.
            Abstract arrays such as `jax.ShapeDtypeStruct` work as well.

    Raises:
        PrecisionError: If the process does not have x64 enabled.
        AssertionError: If a traced value or returned output has wide precision.
    """
    require_float64("assert_precision_closed")
    trace, _, static = eqx.filter_make_jaxpr(fn)(*args)
    code = getattr(unwrap(fn), "__code__", None)
    source = (
        f"{code.co_filename}:{code.co_firstlineno}" if code else type(fn).__qualname__
    )
    hit = _first_wide(trace.jaxpr)
    if hit is not None:
        source_info, dtype = hit
        location = summarize(source_info) if source_info is not None else ""
        raise AssertionError(f"{location or source}: precision widened to {dtype}")
    for leaf in jax.tree.leaves(static):
        if isinstance(leaf, float | complex):
            raise AssertionError(
                f"{source}: precision widened to {np.asarray(leaf).dtype}"
            )


def _narrow_args(args):
    if "precision" in signature(canonical).parameters:
        return canonical(args, precision="float32")

    # TODO: Use canonical directly once its precision argument is available.
    float32 = np.dtype("float32")
    complex64 = complex_of(float32)

    def narrow(leaf):
        if isinstance(leaf, float | complex):
            return (
                complex64.type(leaf)
                if isinstance(leaf, complex)
                else float32.type(leaf)
            )
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            return leaf
        if dtype == np.dtype("float64"):
            return leaf.astype(float32)
        if dtype == np.dtype("complex128"):
            return leaf.astype(complex64)
        return leaf

    return jax.tree.map(narrow, args)


def precision_pair(fn: Callable, args64: tuple, *, bound: float) -> None:
    """Compare the float64 reference with a float32 run, element by element.

    Args:
        fn: Computation returning a pytree of numeric results.
        args64: Positional arguments with float64 or complex128 array leaves.
        bound: Maximum relative error against each float64 reference element.
            A zero reference requires exact agreement.

    Raises:
        PrecisionError: If the process does not have x64 enabled.
        ValueError: If the bound is negative or nonfinite.
        AssertionError: If the shape, structure, finiteness, or bound check fails.
    """
    require_float64("precision_pair")
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("The relative error bound must be finite and nonnegative.")
    reference, structure = jax.tree.flatten_with_path(fn(*args64))
    candidate, candidate_structure = jax.tree.flatten(fn(*_narrow_args(args64)))
    if structure != candidate_structure:
        raise AssertionError("The float32 and float64 outputs have different pytrees.")
    for (path, expected), actual in zip(reference, candidate, strict=True):
        label = jax.tree_util.keystr(path) or "<output>"
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape:
            raise AssertionError(f"{label}: float32 and float64 output shapes differ.")
        if not (np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))):
            raise AssertionError(f"{label}: relative error requires finite outputs.")
        dtype = np.result_type(expected, actual, np.float64)
        np.testing.assert_allclose(
            actual.astype(dtype),
            expected.astype(dtype),
            rtol=bound,
            atol=0,
            err_msg=f"{label}: float32 relative error against float64 exceeds {bound}",
        )
