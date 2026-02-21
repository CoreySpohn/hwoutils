"""Image transformation utilities.

Flux-conserving resampling and sub-pixel image operations. All functions
are JIT-compilable and differentiable.
"""

import functools

import jax
import jax.numpy as jnp

from hwoutils.map_coordinates import map_coordinates


def ccw_rotation_matrix(rotation_deg: float) -> jax.Array:
    """Return the counter-clockwise rotation matrix for a given angle.

    Args:
        rotation_deg: Rotation angle in degrees. Positive = counter-clockwise.

    Returns:
        2x2 rotation matrix as a JAX array.
    """
    theta = jnp.deg2rad(rotation_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ]
    )


@functools.partial(jax.jit, static_argnames=["order", "mode"])
def shift_image(
    image: jax.Array,
    shift_y: float,
    shift_x: float,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
) -> jax.Array:
    """Shift an image with sub-pixel precision using cubic splines.

    Uses inverse mapping: to shift content by (+dy, +dx), sample from
    (y-dy, x-dx).

    Args:
        image: 2D input image.
        shift_y: Shift in Y direction (pixels). Positive = Down.
        shift_x: Shift in X direction (pixels). Positive = Right.
        order: Interpolation order (3 = cubic splines).
        mode: Boundary handling mode.
        cval: Value for 'constant' mode outside boundaries.

    Returns:
        Shifted image with same shape as input.
    """
    ny, nx = image.shape
    y_grid, x_grid = jnp.mgrid[:ny, :nx]
    coords = [y_grid - shift_y, x_grid - shift_x]
    return map_coordinates(image, coords, order=order, mode=mode, cval=cval)


@functools.partial(jax.jit, static_argnames=["shape_tgt"])
def resample_flux(
    f_src: jax.Array,
    pixscale_src: float,
    pixscale_tgt: float,
    shape_tgt: tuple[int, int],
    rotation_deg: float = 0.0,
) -> jax.Array:
    """Resample an image onto a new grid while conserving total flux.

    Performs an affine transformation (rotation and scaling) to map
    the source image onto a target grid. Converts to surface brightness,
    interpolates, then converts back to integrated flux per pixel.

    Args:
        f_src: Source image (2D) with integrated flux per pixel.
        pixscale_src: Pixel scale of source image.
        pixscale_tgt: Pixel scale of target image (same units as src).
        shape_tgt: Target shape (ny_tgt, nx_tgt).
        rotation_deg: CCW rotation angle in degrees.

    Returns:
        Resampled image with total flux conserved. Shape: (ny_tgt, nx_tgt).
    """
    ny_src, nx_src = f_src.shape
    ny_tgt, nx_tgt = shape_tgt

    # Surface brightness (flux per unit area)
    s_src = f_src / (pixscale_src**2)

    # Affine matrix (TARGET pixel centres -> SOURCE coordinates)
    scale = pixscale_tgt / pixscale_src
    a_mat = ccw_rotation_matrix(rotation_deg) * scale

    c_src = jnp.array([(ny_src - 1) / 2.0, (nx_src - 1) / 2.0])
    c_tgt = jnp.array([(ny_tgt - 1) / 2.0, (nx_tgt - 1) / 2.0])
    offset = c_src - a_mat @ c_tgt

    # Grid of TARGET pixel centres
    y_coords = jnp.arange(ny_tgt)
    x_coords = jnp.arange(nx_tgt)
    y_tgt, x_tgt = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    # (2, ny_tgt, nx_tgt)
    coords = jnp.stack([y_tgt, x_tgt], axis=0)
    coords_src = (a_mat @ coords.reshape(2, -1) + offset[:, None]).reshape(coords.shape)

    # Interpolate surface brightness
    s_tgt = map_coordinates(
        s_src, [coords_src[0], coords_src[1]], order=3, mode="constant", cval=0.0
    )

    # Back to integrated flux per target pixel
    return s_tgt * (pixscale_tgt**2)
