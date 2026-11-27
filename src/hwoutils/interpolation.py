"""Piecewise-linear interpolation on a one-dimensional grid."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _as_inexact(value, *peers):
    """Cast to the common inexact dtype, retaining weak scalar promotion."""
    dtype, weak = jax.dtypes.result_type(value, *peers, return_weak_type_flag=True)
    if weak:
        # An explicit dtype would turn a weak table into a strong operand.
        value = jnp.asarray(value)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value * 1.0
        if jnp.issubdtype(dtype, jnp.complexfloating) and not jnp.iscomplexobj(value):
            value = jax.lax.complex(value, jnp.zeros_like(value))
        return value
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = np.float64 if dtype.itemsize == 8 else np.float32
    return jnp.asarray(value, dtype=dtype)


@jax.jit
def _interpolate(x, xp, fp):
    """Weight adjacent samples using only the interval coordinates."""
    if xp.size == 1:
        return jnp.broadcast_to(_as_inexact(fp[0], x), x.shape)

    upper = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, xp.size - 1)
    x_lo, f_lo = xp[upper - 1], fp[upper - 1]
    width = xp[upper] - x_lo
    # Guard both the quotient and its gradient on nearly coincident knots.
    tiny = jnp.abs(width) <= np.spacing(jnp.finfo(xp.dtype).eps)
    fraction = (x - x_lo) / jnp.where(tiny, jnp.ones_like(width), width)
    return jnp.where(tiny, f_lo, f_lo + fraction * (fp[upper] - f_lo))


def interp_linear(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | str | None = None,
    right: ArrayLike | str | None = None,
) -> jax.Array:
    """Interpolate real or complex table values on a finite, increasing grid.

    The interval and its weights depend only on the coordinates, so the
    endpoint-clamped interpolation can be transposed with respect to ``fp``.

    Args:
        x: Real query coordinates of any shape.
        xp: Nonempty one-dimensional array of finite, strictly increasing knots.
        fp: Sample values at the knots, with the same shape as ``xp``.
        left: Fill below the first knot. Defaults to the first sample. Use
            ``"extrapolate"`` to extend the first segment.
        right: Fill above the last knot. Defaults to the last sample. Use
            ``"extrapolate"`` to extend the last segment.

    Returns:
        Interpolated values with the shape of ``x``. A singleton table is
        constant, subject to any supplied boundary fills.

    Raises:
        ValueError: If the table shapes are invalid, the coordinates are
            complex, or an extrapolation string is unrecognized.
    """
    x, xp = _as_inexact(x, xp), _as_inexact(xp, x)
    if jnp.iscomplexobj(x):
        raise ValueError("Interpolation coordinates must be real.")
    fp = _as_inexact(fp)
    if xp.ndim != 1 or xp.size == 0 or fp.shape != xp.shape:
        raise ValueError("xp and fp must be nonempty 1-D arrays of equal shape.")

    value = _interpolate(x, xp, fp)
    for fill, mask, endpoint in (
        (left, x < xp[0], fp[0]),
        (right, x > xp[-1], fp[-1]),
    ):
        if isinstance(fill, str):
            if fill != "extrapolate":
                raise ValueError(f"Unknown extrapolation mode {fill!r}.")
            continue
        fill = endpoint if fill is None else fill
        value = jnp.where(mask, _as_inexact(fill, value), value)
    return value
