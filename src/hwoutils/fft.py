"""FFT-based sub-pixel image shifting.

Provides Fourier shift primitives for sub-pixel image translation. The JAX
versions (fft_shift_x, fft_shift_y) accept precomputed phasors for efficient
repeated shifts. The NumPy versions (fft_shift, fft_shift_1d) are standalone.

All functions operate on 2D images via separable 1D FFTs along each axis,
which is O(2N * N log N) vs O(N^2 log N^2) for a full 2D FFT.
"""

import jax.numpy as jnp
import numpy as np
from jax import lax

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_pad_info(image, pad_factor):
    """Compute padding sizes for FFT shift operations.

    Args:
        image: 2D input image (JAX or NumPy array).
        pad_factor: Factor by which to pad (e.g. 1.5 gives 50% on each side).

    Returns:
        Tuple of (n_pixels_orig, n_pad, img_edge, n_pixels_final).
    """
    n_pixels_orig = image.shape[0]
    n_pad = int(pad_factor * n_pixels_orig)
    img_edge = n_pad + n_pixels_orig
    n_pixels_final = int(2 * n_pixels_orig * pad_factor + n_pixels_orig)
    return n_pixels_orig, n_pad, img_edge, n_pixels_final


# ---------------------------------------------------------------------------
# JAX versions (JIT-compatible, require precomputed phasors)
# ---------------------------------------------------------------------------


def fft_shift_x(image, shift_pixels, phasor, clamp=True):
    """Apply a Fourier shift along the x-axis (JAX, JIT-compatible).

    Uses a precomputed phasor for efficient repeated shifts of images
    with the same shape.

    Args:
        image: 2D input image (JAX array).
        shift_pixels: Sub-pixel shift amount along x.
        phasor: Precomputed exp(-2j * pi * fft_freqs) for the padded size.
        clamp: If True, clamp negative values to zero after shift.

    Returns:
        Shifted image with same shape as input.
    """
    _n_pixels_orig, n_pad, img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    pad_val = jnp.zeros((), dtype=image.dtype)
    padded = lax.pad(image, pad_val, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])
    padded = jnp.fft.fft(padded, axis=1)

    phasor = jnp.tile(phasor**shift_pixels, (padded.shape[0], 1))
    padded = padded * phasor

    padded = jnp.real(jnp.fft.ifft(padded, axis=1))
    image = padded[n_pad:img_edge, n_pad:img_edge]

    if clamp:
        return jnp.maximum(image, 0.0)
    return image


def fft_shift_y(image, shift_pixels, phasor, clamp=True):
    """Apply a Fourier shift along the y-axis (JAX, JIT-compatible).

    Uses a precomputed phasor for efficient repeated shifts of images
    with the same shape.

    Args:
        image: 2D input image (JAX array).
        shift_pixels: Sub-pixel shift amount along y.
        phasor: Precomputed exp(-2j * pi * fft_freqs) for the padded size.
        clamp: If True, clamp negative values to zero after shift.

    Returns:
        Shifted image with same shape as input.
    """
    _n_pixels_orig, n_pad, img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    pad_val = jnp.zeros((), dtype=image.dtype)
    padded = lax.pad(image, pad_val, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])
    padded = jnp.fft.fft(padded, axis=0)

    phasor = jnp.tile(phasor**shift_pixels, (padded.shape[1], 1)).T
    padded = padded * phasor

    padded = jnp.real(jnp.fft.ifft(padded, axis=0))
    image = padded[n_pad:img_edge, n_pad:img_edge]

    if clamp:
        return jnp.maximum(image, 0.0)
    return image


# ---------------------------------------------------------------------------
# NumPy versions (standalone, no precomputed phasors needed)
# ---------------------------------------------------------------------------


def fft_shift_1d(image, shift_pixels, axis):
    """Apply a Fourier shift along a specified axis (NumPy).

    Pads, applies a 1D FFT phasor shift, and unpads. Standalone version
    that computes its own phasor internally.

    Args:
        image: 2D input image (NumPy array).
        shift_pixels: Sub-pixel shift amount.
        axis: Axis to shift (0 for vertical/y, 1 for horizontal/x).

    Returns:
        Shifted image with same shape as input.
    """
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    padded = np.pad(image, n_pad, mode="constant")
    padded = np.fft.fft(padded, axis=axis)

    freqs = np.fft.fftfreq(4 * n_pixels)
    phasor = np.exp(-2j * np.pi * freqs * shift_pixels)

    if axis == 1:
        phasor = np.tile(phasor, (padded.shape[0], 1))
    else:
        phasor = np.tile(phasor, (padded.shape[1], 1)).T

    padded = padded * phasor
    padded = np.real(np.fft.ifft(padded, axis=axis))
    return padded[n_pad:img_edge, n_pad:img_edge]


def fft_shift(image, x=0, y=0):
    """Apply Fourier shifts along x and/or y axes (NumPy).

    Convenience wrapper that calls fft_shift_1d for each non-zero axis.

    Args:
        image: 2D input image (NumPy array).
        x: Sub-pixel shift along x-axis.
        y: Sub-pixel shift along y-axis.

    Returns:
        Shifted image with same shape as input.

    Raises:
        AssertionError: If both x and y are zero.
    """
    assert x != 0 or y != 0, "One of x or y must be non-zero."

    if x != 0:
        image = fft_shift_1d(image, x, axis=1)
    if y != 0:
        image = fft_shift_1d(image, y, axis=0)

    return image


# ---------------------------------------------------------------------------
# FFT shears (Fourier-domain building blocks for image rotation)
# ---------------------------------------------------------------------------


def fft_shear_setup(image):
    """Precompute per-axis Fourier frequencies and center distances for shears.

    The padded grid matches ``fft_shear_x`` / ``fft_shear_y`` (50% zero pad on
    each side). Compute once and reuse across the three shears of a rotation.

    Args:
        image: 2D square input image.

    Returns:
        Tuple ``(x_freqs, x_dists, y_freqs, y_dists)`` for the shear phase ramps.
    """
    _, n_pad, _, _ = get_pad_info(image, 1.5)
    padded = jnp.pad(image, n_pad, mode="constant")

    padded_height, padded_width = padded.shape
    center_y, center_x = (jnp.array(padded.shape) - 1) / 2
    grid_y, grid_x = jnp.mgrid[0:padded_height, 0:padded_width]

    x_dists = grid_x - center_x
    x_freqs = jnp.fft.fftshift(jnp.fft.fftfreq(x_dists.shape[1]))
    x_freqs = jnp.tile(x_freqs, (x_dists.shape[1], 1)).T

    y_dists = grid_y - center_y
    y_freqs = jnp.fft.fftshift(jnp.fft.fftfreq(y_dists.shape[0]))
    y_freqs = jnp.tile(y_freqs, (y_dists.shape[0], 1))

    return x_freqs, x_dists, y_freqs, y_dists


def fft_shear_x(image, shear_factor, x_freqs, x_dists):
    """Shear an image along the x-axis via a Fourier-domain phase ramp.

    Args:
        image: 2D square input image.
        shear_factor: Shear coefficient (e.g. ``tan(theta/2)`` for rotation).
        x_freqs: x frequencies from ``fft_shear_setup``.
        x_dists: x distances from center from ``fft_shear_setup``.

    Returns:
        Sheared image, same shape as the input (zero padding removed).
    """
    _, n_pad, img_edge, _ = get_pad_info(image, 1.5)
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.fftshift(jnp.fft.fft(padded, axis=1))
    padded = jnp.exp(-2j * jnp.pi * shear_factor * x_freqs * x_dists) * padded
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.ifft(padded, axis=1)
    padded = jnp.fft.fftshift(padded)
    return jnp.real(padded[n_pad:img_edge, n_pad:img_edge])


def fft_shear_y(image, shear_factor, y_freqs, y_dists):
    """Shear an image along the y-axis via a Fourier-domain phase ramp.

    Args:
        image: 2D square input image.
        shear_factor: Shear coefficient (e.g. ``-sin(theta)`` for rotation).
        y_freqs: y frequencies from ``fft_shear_setup``.
        y_dists: y distances from center from ``fft_shear_setup``.

    Returns:
        Sheared image, same shape as the input (zero padding removed).
    """
    _, n_pad, img_edge, _ = get_pad_info(image, 1.5)
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.fftshift(jnp.fft.fft(padded, axis=0))
    padded = jnp.exp(-2j * jnp.pi * shear_factor * y_freqs * y_dists) * padded
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.ifft(padded, axis=0)
    padded = jnp.fft.fftshift(padded)
    return jnp.real(padded[n_pad:img_edge, n_pad:img_edge])
