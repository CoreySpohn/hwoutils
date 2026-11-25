"""Floating precision of arrays and model pytrees."""

import os

import jax
import jax.numpy as jnp
import numpy as np


class PrecisionError(RuntimeError):
    """An operation requires a different or a uniform floating precision."""


def enable_x64(use_x64: bool = True) -> None:
    """Enable 64-bit floating-point precision in JAX.

    By default JAX uses 32-bit precision. Call this before any array
    operations to switch to 64-bit (NumPy-compatible) precision.

    Args:
        use_x64: When True, JAX arrays use 64-bit precision.
            Falls back to the ``JAX_ENABLE_X64`` environment variable
            when False.
    """
    if not use_x64:
        use_x64 = bool(os.getenv("JAX_ENABLE_X64", 0))
    jax.config.update("jax_enable_x64", use_x64)


def float_dtype() -> np.dtype:
    """Return the active float dtype: float64 under x64, else float32.

    Read it when an object is built, not at import, so that a test which
    scopes ``jax.enable_x64()`` around a construction sees the precision in
    force there.

    Returns:
        The NumPy dtype that JAX uses for a default float array.
    """
    return np.dtype(jax.dtypes.canonicalize_dtype(np.float64))


def precision_of(tree) -> np.dtype:
    """Find the common real precision outside bulk fields and foreign Modules.

    Args:
        tree: An array or pytree. Python floats use the construction default.

    Returns:
        The real dtype, or the construction default for a tree without floats.

    Raises:
        PrecisionError: If floating leaves have different real precisions.
    """
    from hwoutils.leaves import _walk, is_foreign

    if is_foreign(tree):
        return float_dtype()
    leaves = _walk(tree, lambda module: not is_foreign(module), frozenset())
    return _precision_of_leaves(leaf for _, leaf, bulk in leaves if not bulk)


def _precision_of_leaves(leaves) -> np.dtype:
    precisions = set()
    for leaf in leaves:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            if isinstance(leaf, float | complex):
                precisions.add(float_dtype())
        elif isinstance(dtype, np.dtype) and jnp.issubdtype(dtype, jnp.inexact):
            precisions.add(jnp.finfo(dtype).dtype)
    if len(precisions) > 1:
        raise PrecisionError(
            "Mixed floating precisions. Use canonical(tree, precision=...) "
            "to choose one precision."
        )
    return precisions.pop() if precisions else float_dtype()


def complex_of(dtype) -> np.dtype:
    """Return the complex partner of a supported real dtype.

    Args:
        dtype: The real float32 or float64 dtype.

    Returns:
        The corresponding complex64 or complex128 dtype.

    Raises:
        PrecisionError: If the dtype has no supported complex partner.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return np.dtype(np.complex64)
    if dtype == np.float64:
        return np.dtype(np.complex128)
    raise PrecisionError(f"Expected float32 or float64, got {dtype}.")


@jax.custom_jvp
@jax.jit
def _convert_float_width(x):
    """Convert float32 to float64 or vice versa, retaining subnormals."""
    single = np.finfo(np.float32)
    step = float(single.smallest_subnormal)
    if x.dtype == np.float32:
        # Bits 0..22 count subnormal units. Arithmetic in float64 is exact.
        bits = jnp.abs(x).view(jnp.uint32)
        wide = x.astype(jnp.float64)
        subnormal = jnp.copysign(bits.astype(jnp.float64) * step, wide)
        return jnp.where(bits < (1 << single.nmant), subnormal, wide)
    # Round the magnitude to the nearest subnormal significand, ties to even.
    magnitude = jnp.minimum(jnp.abs(x), float(single.tiny))
    units = jnp.rint(magnitude / step).astype(jnp.uint32)
    subnormal = jnp.copysign(units.view(jnp.float32), x.astype(jnp.float32))
    return jnp.where(jnp.abs(x) < float(single.tiny), subnormal, x.astype(jnp.float32))


@_convert_float_width.defjvp
def _convert_float_width_jvp(primals, tangents):
    result = _convert_float_width(*primals)
    return result, tangents[0].astype(result.dtype)


def _as_inexact_array(x, dtype):
    """Cast arrays between working precisions without flushing subnormals."""
    if not isinstance(x, jax.Array) or not jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.asarray(x, dtype=dtype)
    source = jnp.finfo(x.dtype).dtype
    target = jnp.finfo(dtype).dtype
    if source == target or {source.name, target.name} != {"float32", "float64"}:
        return jnp.asarray(x, dtype=dtype)
    if float_dtype() != np.float64:
        # A concrete float64 array can outlive the scope that constructed it.
        return jnp.asarray(np.asarray(x), dtype=dtype)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        return jax.lax.complex(
            _convert_float_width(x.real), _convert_float_width(x.imag)
        )
    return jnp.asarray(_convert_float_width(x), dtype=dtype)


def like(x, ref) -> jax.Array:
    """Cast a value to the precision of its owner, preserving complex values.

    Args:
        x: A real or complex value to cast.
        ref: An array or pytree supplying the working precision.

    Returns:
        A real or complex JAX array at the reference precision.

    Raises:
        PrecisionError: If the reference has mixed or unavailable precision.
    """
    dtype = precision_of(ref)
    if dtype == np.float64:
        require_float64("like")
    return _as_inexact_array(x, complex_of(dtype) if jnp.iscomplexobj(x) else dtype)


def require_float64(what: str, *trees) -> None:
    """Require float64 availability and float64 floating leaves on every owner.

    Args:
        what: Name of the operation requiring float64.
        *trees: Optional owner arrays or model pytrees to check.

    Raises:
        PrecisionError: If float64 is unavailable or an owner is not float64.
    """
    if float_dtype() != np.float64:
        raise PrecisionError(f"{what} requires float64. Enable x64 with enable_x64().")
    if not trees:
        return
    remedy = f"{what} requires float64. Use canonical(tree, precision='float64')."
    try:
        dtype = precision_of(trees)
    except PrecisionError as exc:
        raise PrecisionError(remedy) from exc
    if dtype != np.float64:
        raise PrecisionError(remedy)
