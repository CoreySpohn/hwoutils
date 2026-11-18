"""Radial statistics for images with explicit pixel geometry."""

from operator import index
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from hwoutils.grids import Image


def get_center(shape: tuple[int, int]) -> tuple[float, float]:
    """Return the zero-indexed geometric center (cy, cx) of shape (ny, nx)."""
    return ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)


@jax.jit(static_argnames=("stat", "nbins"))
def radial_profile(
    image: "Image",
    *,
    stat: str = "mean",
    edges: jax.typing.ArrayLike | None = None,
    nbins: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return radial bin centers, statistics, and pixel counts for a 2D image.

    Distances and bin centers are in the image's grid units, measured from
    its optical axis. Supply strictly increasing edges in those units, or
    use equally spaced edges from zero to the farthest sample center.
    Explicit edges take precedence over nbins. By default there are
    max(1, max(grid.shape) // 2) bins.

    Bins include their left edge, and the last bin also includes its right
    edge. Pixels outside the edge range are excluded. Empty bins return NaN.
    The statistic (mean, median, or max) propagates NaNs within a bin.
    Under JIT, stat and nbins must be static.
    """
    if stat not in ("mean", "median", "max"):
        raise ValueError(f"unknown stat: {stat!r}; use 'mean', 'median', or 'max'")
    if image.data.ndim != 2 or jnp.iscomplexobj(image.data):
        raise ValueError("radial_profile requires a real 2D Image")
    radius = image.grid.radius().ravel()
    if edges is None:
        nbins = max(1, max(image.grid.shape) // 2) if nbins is None else index(nbins)
        if nbins < 1:
            raise ValueError("nbins must be positive")
        edges = jnp.linspace(0, radius.max(), nbins + 1)
    else:
        edges = jnp.asarray(edges)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be one-dimensional with at least two entries")
        nbins = edges.size - 1
        edges = edges.astype(jnp.result_type(edges, radius))

    centers = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    bins = jnp.searchsorted(edges[1:-1], radius, side="right")
    inside = (radius >= edges[0]) & (radius <= edges[-1])
    bins = jnp.where(inside, bins, nbins)
    counts = jnp.zeros(nbins, dtype=bins.dtype).at[bins].add(1, mode="drop")
    data = jnp.asarray(
        image.data.ravel(), dtype=jnp.result_type(image.data, jnp.float32)
    )
    if stat == "mean":
        totals = jnp.zeros(nbins, dtype=data.dtype).at[bins].add(data, mode="drop")
        values = totals / jnp.maximum(counts, 1)
    elif stat == "max":
        values = (
            jnp.full(nbins, -jnp.inf, dtype=data.dtype).at[bins].max(data, mode="drop")
        )
    else:
        # Sort by bin, then by value. The two middle positions also give the
        # odd-count median, where both positions select the same sample.
        sorted_data = data[jnp.lexsort((data, bins))]
        start = jnp.where(counts > 0, jnp.cumsum(counts) - counts, 0)
        lo = sorted_data[start + (jnp.maximum(counts, 1) - 1) // 2]
        hi = sorted_data[start + counts // 2]
        values = jnp.where(counts % 2 == 1, lo, lo / 2 + hi / 2)
        has_nan = (
            jnp.zeros(nbins, dtype=bool).at[bins].max(jnp.isnan(data), mode="drop")
        )
        values = jnp.where(has_nan, jnp.nan, values)
    return centers, jnp.where(counts > 0, values, jnp.nan), counts
