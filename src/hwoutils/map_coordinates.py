"""Cubic spline interpolation for JAX.

This module contains an implementation of ``map_coordinates`` with cubic spline
interpolation. It is adapted from the JAX project (PR #14218 by Louis Desdoigts)
and is licensed under the Apache 2.0 license.

Original JAX source:
    https://github.com/google/jax/blob/main/jax/_src/scipy/ndimage.py
"""

import functools
import itertools
import operator
from collections.abc import Callable, Sequence
from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax._src import api
from jax._src.numpy.linalg import inv
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


_INDEX_FIXERS: Dict[str, Callable[[Array, int], Array]] = {
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


def _cubic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
    return _spline_point(None, coordinate)


def _build_matrix(n: int, diag: float = 4) -> Array:
    M = jnp.zeros((n, n))
    idx = jnp.arange(n)
    M = M.at[idx, idx].set(diag)
    M = M.at[idx[:-1], idx[:-1] + 1].set(1)
    M = M.at[idx[:-1] + 1, idx[:-1]].set(1)
    return M


def _construct_vector(data: Array, c2: Array, cnp2: Array) -> Array:
    n = data.shape[0]
    d = jnp.zeros(n)
    d = d.at[0].set(6 * data[0] - c2)
    d = d.at[-1].set(6 * data[-1] - cnp2)
    d = d.at[1:-1].set(6 * data[1:-1])
    return d


def _solve_coefficients(data: Array, A_inv: Array, h=1) -> Array:
    n = data.shape[0]
    finite_diff = jnp.diff(data, axis=0) / h
    c2 = 0.0
    cnp2 = 0.0
    d = vmap(_construct_vector, in_axes=(1, None, None))(finite_diff, c2, cnp2)
    c = vmap(jnp.dot, in_axes=(None, 0))(A_inv, d).T
    c = jnp.concatenate(
        [jnp.zeros((1, c.shape[1])), c, jnp.zeros((1, c.shape[1]))], axis=0
    )
    return c


def _spline_coefficients(data: Array) -> Array:
    n = data.shape[0]
    A = _build_matrix(n - 2)
    A_inv = inv(A)
    coefficients = _solve_coefficients(data, A_inv)
    return coefficients


def _spline_basis(t: Array) -> Array:
    abs_t = jnp.abs(t)
    return jnp.where(
        abs_t <= 1,
        2 / 3 - abs_t**2 + abs_t**3 / 2,
        jnp.where(abs_t <= 2, (2 - abs_t) ** 3 / 6, 0.0),
    )


def _spline_value(coefficients: Array, coordinate: Array, indexes: Array) -> Array:
    t = coordinate - indexes
    weights = _spline_basis(t)
    return jnp.sum(coefficients * weights, axis=0)


def _spline_point(coefficients: Array, coordinate: Array) -> Array:
    idx = jnp.floor(coordinate).astype(jnp.int32)
    indexes = jnp.array([idx - 1, idx, idx + 1, idx + 2])
    t = coordinate - indexes
    weights = _spline_basis(t)
    return [(i, w) for i, w in zip(indexes, weights)]


def _cubic_spline(input: Array, coordinates: Array) -> Array:
    coefficients = _spline_coefficients(input)
    indexes = jnp.arange(input.shape[0])
    return vmap(_spline_value, in_axes=(None, 0, None))(
        coefficients, coordinates, indexes
    )


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
        is_valid = lambda index, size: (index >= 0) & (index < size)
    else:
        is_valid = lambda index, size: True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    elif order == 3:
        interp_fun = _cubic_indices_and_weights
    else:
        raise NotImplementedError(
            f"jax.scipy.ndimage.map_coordinates only supports order 0, 1, or 3, got {order}"
        )

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinates_arr, input_arr.shape):
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

        if all(googles is True for googles in validities):
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
    """Map coordinates using cubic spline interpolation.

    Args:
        input: The input array.
        coordinates: Sequence of coordinate arrays for each dimension.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic spline).
        mode: Boundary handling ('constant', 'nearest', 'wrap', 'mirror',
            'reflect').
        cval: Value for 'constant' mode outside boundaries.

    Returns:
        Interpolated values at the given coordinates.
    """
    return api.jit(_map_coordinates, static_argnums=(2, 3))(
        input, coordinates, order, mode, cval
    )
