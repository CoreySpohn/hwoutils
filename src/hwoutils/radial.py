"""Radial utilities for image analysis.

Functions for computing radial distance grids and extracting radial profiles
from 2D images. All functions are JIT-compilable and differentiable.
"""

import jax
import jax.numpy as jnp


def radial_distance(
    shape: tuple[int, int],
    center: tuple[float, float] | None = None,
) -> jax.Array:
    """Calculate radial distance from center for each pixel.

    Args:
        shape: Image shape (ny, nx).
        center: Center coordinates (cy, cx). If None, uses geometric center
            ``((ny-1)/2, (nx-1)/2)``.

    Returns:
        2D array of radial distances in pixels.
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    cy, cx = center

    y, x = jnp.ogrid[:ny, :nx]
    return jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)


@jax.jit(static_argnames=["nbins"])
def radial_profile(
    image: jax.Array,
    pixel_scale: float = 1.0,
    center: tuple[float, float] | None = None,
    nbins: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute the radial profile of a 2D image.

    Bins pixels by their distance from center and computes the mean value
    in each radial bin.

    Args:
        image: 2D input image.
        pixel_scale: Conversion factor from pixels to physical units
            (e.g. λ/D per pixel). Default 1.0 gives bins in pixels.
        center: Center coordinates (cy, cx). If None, uses geometric center.
        nbins: Number of radial bins. If None, uses ``floor(max_dim / 2)``.

    Returns:
        ``(separations, profile)`` where separations are bin centers in
        physical units and profile is the mean value in each bin.
    """
    ny, nx = image.shape
    if nbins is None:
        nbins = int(max(ny, nx) // 2)

    r = radial_distance((ny, nx), center)
    max_radius = jnp.max(r)

    bin_edges = jnp.linspace(0, max_radius, nbins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    r_flat = r.ravel()
    image_flat = image.ravel()

    # Assign each pixel to a bin (1-indexed; 0 = below first edge)
    inds = jnp.digitize(r_flat, bin_edges)
    # Clamp overflow into last bin
    inds = jnp.clip(inds, 1, nbins)

    # Compute mean per bin using scatter
    bin_sums = jnp.zeros(nbins).at[inds - 1].add(image_flat)
    bin_counts = jnp.zeros(nbins).at[inds - 1].add(1.0)

    profile = jnp.where(bin_counts > 0, bin_sums / bin_counts, 0.0)
    separations = bin_centers * pixel_scale

    return separations, profile
