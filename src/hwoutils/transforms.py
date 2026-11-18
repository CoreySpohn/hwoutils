"""Image transformation utilities.

Area-scaled interpolation and Fourier rotation.
JAX derivatives describe the chosen reconstruction and its piecewise boundaries.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

from hwoutils.fft import fft_shear_setup, fft_shear_x, fft_shear_y
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
    """Shift an image with sub-pixel precision.

    Uses inverse mapping: to shift content by (+dy, +dx), sample from
    (y-dy, x-dx).

    Args:
        image: 2D input image.
        shift_y: Shift in Y direction (pixels). Positive = Down.
        shift_x: Shift in X direction (pixels). Positive = Right.
        order: Interpolation order passed to ``map_coordinates``. Default
            is 3, which uses the Keys cubic convolution kernel (see
            ``docs/interpolation.md``).
        mode: Boundary handling mode.
        cval: Value for 'constant' mode outside boundaries.

    Returns:
        Shifted image with same shape as input.
    """
    ny, nx = image.shape
    y_grid, x_grid = jnp.mgrid[:ny, :nx]
    coords = [y_grid - shift_y, x_grid - shift_x]
    return map_coordinates(image, coords, order=order, mode=mode, cval=cval)


@functools.partial(jax.jit, static_argnames=["shape_tgt", "order"])
def resample_flux(
    f_src: jax.Array,
    pixscale_src: float,
    pixscale_tgt: float,
    shape_tgt: tuple[int, int],
    rotation_deg: float = 0.0,
    order: int = 3,
) -> jax.Array:
    """Interpolate at target pixel centers and apply the pixel-area ratio.

    This is a center-sampling approximation to integrated target-pixel flux,
    not a conservative pixel integrator or an antialiasing filter. Use
    ``hwoutils.grids.integrate_footprints`` for conservative pixel integration.

    Approximate photometry requires a source smooth on both grids and
    negligible at the zero-padded source boundaries. A Gaussian truncated
    while still bright at those boundaries has a sharp edge that can alias:
    a sigma=1.5 source-pixel Gaussian on a 5x5 cutout can exceed 5% flux
    error even with a target/source scale ratio below two and a target
    containing the entire reconstruction. For this Gaussian, conservation
    checks use source samples extending at least four sigma from the peak
    on every side and scale ratios below two (target sigma > 0.75 pixels).
    These conditions describe a tested smooth-source regime, not an error
    bound for arbitrary images. Check convergence for the chosen sampling.

    Args:
        f_src: Source image (2D), interpreted as flux per source pixel.
        pixscale_src: Source pixel scale.
        pixscale_tgt: Target pixel scale, in the same units.
        shape_tgt: Target shape (ny, nx).
        rotation_deg: CCW rotation in degrees with the origin at lower left.
        order: Interpolation order: 0, 1, or 3 (Keys cubic convolution).

    Returns:
        Interpolated image multiplied by the target/source pixel-area ratio.
        The grids share their geometric centers. Outside samples are zero;
        cropping, undersampling, and reconstruction can change total flux.
    """
    ny_src, nx_src = f_src.shape
    ny_tgt, nx_tgt = shape_tgt
    scale = pixscale_tgt / pixscale_src
    theta = jnp.deg2rad(rotation_deg)
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    y = (jnp.arange(ny_tgt) - (ny_tgt - 1) / 2)[:, None] * scale
    x = (jnp.arange(nx_tgt) - (nx_tgt - 1) / 2)[None, :] * scale
    coords = [
        cosine * y - sine * x + (ny_src - 1) / 2,
        sine * y + cosine * x + (nx_src - 1) / 2,
    ]
    # Promote integer pixel fluxes before interpolation, avoiding output rounding.
    source = jnp.asarray(f_src, dtype=jnp.result_type(f_src, 1.0))
    return map_coordinates(source, coords, order=order) * scale**2


def _decompose_angle(angle: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split an angle into a (-45, 45] remainder plus a count of 90 deg turns.

    The three-shear Fourier rotation is only well behaved for ``|angle| <= 45``
    deg, so larger rotations are handled by lossless 90 deg array rotations
    plus a small residual shear rotation.
    """
    angle = angle % 360
    n_rot = (angle // 90).astype(int)
    adjusted_angle = angle % 90
    adjusted_angle, n_rot = lax.cond(
        adjusted_angle > 45,
        lambda x: (x - 90, n_rot + 1),
        lambda x: (x, n_rot),
        adjusted_angle,
    )
    # (315, 360) lands on n_rot == 4; fold it back to 0.
    n_rot = lax.cond(n_rot == 4, lambda x: 0, lambda x: x, n_rot)
    return adjusted_angle, n_rot


def _rot90_traceable(m: jax.Array, k: jax.Array, axes=(0, 1)) -> jax.Array:
    """Traceable ``jnp.rot90`` (``k`` may be a tracer)."""
    k = k % 4
    branches = [functools.partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)]
    return lax.switch(k, branches)


def _rotate_with_shear(image, rot_deg, pad_factor=0.5, translation=None):
    """Rotate a residual angle on one padded grid, then crop only once."""
    if pad_factor < 0:
        raise ValueError("pad_factor must be nonnegative")
    n = image.shape[0]
    pad = int(pad_factor * n)
    padded = jnp.pad(image, pad)
    theta = jnp.deg2rad(rot_deg)
    a, b = jnp.tan(theta / 2), -jnp.sin(theta)
    fx, dy, fy, dx = fft_shear_setup(padded, pad_factor=0.0)
    padded = fft_shear_x(padded, a, fx, dy, pad_factor=0.0)
    padded = fft_shear_y(padded, b, fy, dx, pad_factor=0.0)
    padded = fft_shear_x(padded, a, fx, dy, pad_factor=0.0)
    if translation is not None:
        # Move the rotation origin without interpolating/cropping the source first.
        for axis, frequency, shift in (
            (1, fx, translation[1]),
            (0, fy, translation[0]),
        ):
            spectrum = jnp.fft.fft(padded, axis=axis)
            padded = jnp.fft.ifft(
                spectrum * jnp.exp(-2j * jnp.pi * frequency * shift), axis=axis
            )
            if not jnp.iscomplexobj(image):
                padded = padded.real
    return padded[pad : pad + n, pad : pad + n]


@functools.partial(jax.jit, static_argnames=["pad_factor"])
def rotate_image(
    image: jax.Array,
    rotation_deg: float,
    *,
    pad_factor: float = 0.5,
    center: tuple[float, float] | None = None,
) -> jax.Array:
    """Rotate a square image using three Fourier shears on a padded grid.

    Positive angles are CCW when displayed with ``origin="lower"``. Right-angle
    array turns reduce the shear angle to (-45, 45]. The residual path is also
    evaluated at zero to retain the angle derivative; identity and right-angle
    outputs therefore agree to FFT roundoff rather than necessarily bitwise.

    Real images remain real after each shear. On even FFT grids this chooses
    the real cosine interpolant for the Nyquist mode, whose amplitude can
    decrease under fractional shifts. Rotation is not exactly unitary for
    arbitrary sampled images, and cropping can lose flux. No positivity clamp
    or flux renormalization is applied.

    Args:
        image: Nonempty square 2D image, real or complex.
        rotation_deg: Counter-clockwise rotation angle in degrees.
        pad_factor: Static, nonnegative padding per side as a fraction of N.
            Default 0.5 gives width 2N (2N-1 for odd N). Set 1.5 for approximately
            4N. Tight crops, edge content, and precision PSF wings need a padding
            convergence check; padding does not remove sampling aliasing.
        center: Rotation origin (row, column) in input pixel coordinates.
            Default is the geometric center ((N-1)/2, (N-1)/2). Specify the FITS
            optical center explicitly when it differs. Nondefault centers may
            require more padding to contain the shifted intermediate image.

    Returns:
        Rotated image with the input shape. Angle derivatives are local to the
        selected shear decomposition; sampled images can differ at its seams.
    """
    image = jnp.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not image.shape[0]:
        raise ValueError("rotate_image requires a nonempty square image")
    translation = None
    if center is not None:
        center_array = jnp.asarray(center)
        if center_array.shape != (2,):
            raise ValueError("center must contain (row, column)")
        cy, cx = center_array - (image.shape[0] - 1) / 2
        theta = jnp.deg2rad(rotation_deg)
        cosine, sine = jnp.cos(theta), jnp.sin(theta)
        translation = (
            cy - (sine * cx + cosine * cy),
            cx - (cosine * cx - sine * cy),
        )
    rot_deg, n_rot = _decompose_angle(-jnp.asarray(rotation_deg))
    image = _rot90_traceable(image, n_rot)
    return _rotate_with_shear(image, rot_deg, pad_factor, translation)
