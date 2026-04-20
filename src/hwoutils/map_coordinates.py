"""Image coordinate mapping with sub-pixel interpolation for JAX.

Adapted from the JAX project (PR #14218 by Louis Desdoigts); Apache 2.0.

At ``order=3`` the kernel is the Keys cubic convolution (``a = -0.5``,
Catmull-Rom), a true 4-tap interpolant. See ``hwoutils/docs/interpolation.md``
for details.

Original JAX source:
    https://github.com/google/jax/blob/main/jax/_src/scipy/ndimage.py
"""

import functools
import itertools
import operator
from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jax import lax
from jax._src import api
from jax._src.typing import Array, ArrayLike


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1
    return jnp.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


_INDEX_FIXERS: dict[str, Callable[[Array, int], Array]] = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: jnp.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
    index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = coordinate.dtype.type(1) - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _keys_basis(t: Array) -> Array:
    """Keys cubic convolution kernel with a = -0.5 (Catmull-Rom).

    Piecewise cubic with compact support [-2, 2]:

        inner (abs(t) <= 1):       1.5 abs(t)^3 - 2.5 abs(t)^2 + 1
        outer (1 < abs(t) <= 2):  -0.5 abs(t)^3 + 2.5 abs(t)^2 - 4 abs(t) + 2
        else:                     0

    Properties:
      - True interpolant: K(0) = 1, K(k) = 0 for non-zero integer k,
        so evaluating at integer grid points returns the sample exactly.
      - Partition of unity at integer grid spacing: sum over integer
        shifts of K equals 1 for any offset. This makes it
        flux-preserving on integer downsampling of band-limited inputs.
      - Reproduces constants and linear functions exactly under
        translation; reproduces quadratics approximately (O(h^3)).
      - Has small negative lobes (min value ~ -0.0625), so outputs can
        be slightly negative even for non-negative inputs.
    """
    abs_t = jnp.abs(t)
    inner = 1.5 * abs_t**3 - 2.5 * abs_t**2 + 1.0
    outer = -0.5 * abs_t**3 + 2.5 * abs_t**2 - 4.0 * abs_t + 2.0
    return jnp.where(
        abs_t <= 1.0,
        inner,
        jnp.where(abs_t <= 2.0, outer, 0.0),
    )


def _cubic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
    """4-tap Keys stencil: samples at floor(coord) + {-1, 0, 1, 2}."""
    idx = jnp.floor(coordinate).astype(jnp.int32)
    indexes = jnp.array([idx - 1, idx, idx + 1, idx + 2])
    t = coordinate - indexes
    weights = _keys_basis(t)
    return [(i, w) for i, w in zip(indexes, weights, strict=True)]


def _map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str,
    cval: ArrayLike,
) -> Array:
    input_arr = jnp.asarray(input)
    coordinates_arr = [jnp.asarray(c, dtype=input_arr.dtype) for c in coordinates]
    cval = jnp.asarray(cval, input_arr.dtype)

    if len(coordinates_arr) != input_arr.ndim:
        raise ValueError(
            f"coordinates must be a sequence of length input.ndim = {input_arr.ndim}, "
            f"got {len(coordinates_arr)}"
        )

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        raise NotImplementedError(
            f"jax.scipy.ndimage.map_coordinates does not support mode {mode!r}"
        )

    if mode == "constant":

        def is_valid(index, size):
            return (index >= 0) & (index < size)
    else:

        def is_valid(index, size):
            return True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    elif order == 3:
        interp_fun = _cubic_indices_and_weights
    else:
        raise NotImplementedError(
            f"map_coordinates only supports order 0, 1, or 3, got {order}"
        )

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinates_arr, input_arr.shape, strict=True):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, weight, valid))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices = []
        validities = []
        weights = []
        for index, weight, valid in items:
            indices.append(index)
            weights.append(weight)
            validities.append(valid)

        if all(v is True for v in validities):
            contribution = input_arr[tuple(indices)]
        else:
            all_valid = _nonempty_prod(validities) if validities else True
            contribution = jnp.where(all_valid, input_arr[tuple(indices)], cval)

        outputs.append(_nonempty_prod(weights) * contribution)

    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result


def map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str = "constant",
    cval: ArrayLike = 0.0,
) -> Array:
    """Map an input array onto new coordinates via sub-pixel interpolation.

    Args:
        input: The input array.
        coordinates: Sequence of coordinate arrays for each dimension.
        order: Interpolation order:

            - 0: nearest neighbor.
            - 1: linear.
            - 3: Keys cubic convolution (``a = -0.5``, Catmull-Rom). A
              true 4-tap interpolant with partition of unity at integer
              grid spacing, so resampling preserves sample values at
              integer offsets and conserves flux on integer downsampling
              of band-limited inputs. See ``docs/interpolation.md``.

        mode: Boundary handling ('constant', 'nearest', 'wrap', 'mirror',
            'reflect').
        cval: Value for 'constant' mode outside boundaries.

    Returns:
        Interpolated values at the given coordinates.
    """
    return api.jit(_map_coordinates, static_argnums=(2, 3))(
        input, coordinates, order, mode, cval
    )
