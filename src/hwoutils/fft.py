"""FFT-based sub-pixel image shifting.

Provides Fourier shift primitives for sub-pixel image translation with JAX
(fft_shift_x, fft_shift_y) and NumPy (fft_shift, fft_shift_1d).

All functions operate on 2D images via separable 1D FFTs along each axis,
which is O(2N * N log N) vs O(N^2 log N^2) for a full 2D FFT.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_pad_info(image, pad_factor):
    """Compute padding sizes for FFT shift operations.

    Args:
        image: 2D input image (JAX or NumPy array).
        pad_factor: Padding on each side as a fraction of the input height.
            For example, 1.5 adds 150% per side, giving width 4N for even N.

    Returns:
        Tuple of (n_pixels_orig, n_pad, img_edge, n_pixels_final).
    """
    n_pixels_orig = image.shape[0]
    n_pad = int(pad_factor * n_pixels_orig)
    img_edge = n_pad + n_pixels_orig
    n_pixels_final = n_pixels_orig + 2 * n_pad
    return n_pixels_orig, n_pad, img_edge, n_pixels_final


def _reduce_float_shift(shift_pixels, n, xp):
    if shift_pixels.dtype.itemsize < 4:
        shift_pixels = shift_pixels.astype(xp.float32)
    turns = xp.round(shift_pixels / n)
    # Keep small shifts bit for bit, including signed zero.
    return xp.where(turns == 0, shift_pixels, shift_pixels - n * turns)


def _reduce_integer_shift(shift_pixels, n, dtype, xp):
    # Reduce full-width unsigned shifts before casting to a signed dtype.
    if shift_pixels.dtype.kind != "u" or shift_pixels.dtype.itemsize < dtype.itemsize:
        shift_pixels = shift_pixels.astype(dtype)
    remainder = xp.fmod(shift_pixels, n).astype(dtype)
    # The signed remainder preserves both half-period ties and avoids overflow.
    return xp.where(
        xp.abs(remainder) <= n // 2, remainder, remainder - xp.sign(remainder) * n
    )


# ---------------------------------------------------------------------------
# JAX versions (JIT-compatible)
# ---------------------------------------------------------------------------


def _jax_phase_ramp(n, shift_pixels, real_dtype):
    shift_pixels = jnp.asarray(shift_pixels)
    if jnp.issubdtype(shift_pixels.dtype, jnp.floating):
        shift_pixels = _reduce_float_shift(shift_pixels, n, jnp)
    elif shift_pixels.dtype.kind in "biu":
        shift_pixels = _reduce_integer_shift(shift_pixels, n, jnp.result_type(int), jnp)
    freqs = jnp.fft.fftfreq(n, dtype=real_dtype)
    return jnp.exp(-2j * jnp.pi * freqs * shift_pixels)


def fft_shift_x(image, shift_pixels, clamp=True):
    """Apply a Fourier shift along the x-axis (JAX, JIT-compatible).

    Rectangular images keep real/complex kind; complex input requires clamp=False.
    Keep float32 shift magnitudes well below 2**24 times the desired accuracy
    in pixels (about 1e5 pixels for 0.01-pixel accuracy). Reduce larger shifts
    modulo the padded period in float64 before calling.

    Args:
        image: 2D input image (JAX array).
        shift_pixels: Sub-pixel shift along x, reduced modulo the padded period if real.
        clamp: If True, clamp negative values to zero after shift.

    Returns:
        Shifted image with same shape as input.

    Raises:
        TypeError: If clamp is True and image is complex.
    """
    if clamp and jnp.iscomplexobj(image):
        raise TypeError("Cannot clamp a complex image; use clamp=False.")
    _n_pixels_orig, n_pad, _img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    pad_val = jnp.zeros((), dtype=image.dtype)
    padded = lax.pad(image, pad_val, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])
    padded = jnp.fft.fft(padded, axis=1)

    phase_ramp = _jax_phase_ramp(padded.shape[1], shift_pixels, padded.real.dtype)
    padded = padded * phase_ramp

    padded = jnp.fft.ifft(padded, axis=1)
    if not jnp.iscomplexobj(image):
        padded = jnp.real(padded)
    image = padded[n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]]

    if clamp:
        return jnp.maximum(image, 0.0)
    return image


def fft_shift_y(image, shift_pixels, clamp=True):
    """Apply a Fourier shift along the y-axis (JAX, JIT-compatible).

    Rectangular images keep real/complex kind; complex input requires clamp=False.
    Keep float32 shift magnitudes well below 2**24 times the desired accuracy
    in pixels (about 1e5 pixels for 0.01-pixel accuracy). Reduce larger shifts
    modulo the padded period in float64 before calling.

    Args:
        image: 2D input image (JAX array).
        shift_pixels: Sub-pixel shift along y, reduced modulo the padded period if real.
        clamp: If True, clamp negative values to zero after shift.

    Returns:
        Shifted image with same shape as input.

    Raises:
        TypeError: If clamp is True and image is complex.
    """
    if clamp and jnp.iscomplexobj(image):
        raise TypeError("Cannot clamp a complex image; use clamp=False.")
    _n_pixels_orig, n_pad, _img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    pad_val = jnp.zeros((), dtype=image.dtype)
    padded = lax.pad(image, pad_val, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])
    padded = jnp.fft.fft(padded, axis=0)

    phase_ramp = _jax_phase_ramp(padded.shape[0], shift_pixels, padded.real.dtype)
    padded = padded * phase_ramp[:, None]

    padded = jnp.fft.ifft(padded, axis=0)
    if not jnp.iscomplexobj(image):
        padded = jnp.real(padded)
    image = padded[n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]]

    if clamp:
        return jnp.maximum(image, 0.0)
    return image


# ---------------------------------------------------------------------------
# NumPy versions
# ---------------------------------------------------------------------------


def fft_shift_1d(image, shift_pixels, axis):
    """Apply a Fourier shift along a specified axis (NumPy).

    Pads the image, applies a phase ramp in the Fourier domain, and unpads.
    Rectangular images keep the input's real or complex kind.
    Keep float32 shift magnitudes well below 2**24 times the desired accuracy
    in pixels (about 1e5 pixels for 0.01-pixel accuracy). Reduce larger shifts
    modulo the padded period in float64 before calling.

    Args:
        image: 2D input image (NumPy array).
        shift_pixels: Sub-pixel shift, reduced modulo the padded axis period if real.
        axis: Axis to shift (0 for vertical/y, 1 for horizontal/x).

    Returns:
        Shifted image with same shape as input.
    """
    _, n_pad, _, _ = get_pad_info(image, 1.5)
    padded = np.pad(image, n_pad, mode="constant")
    padded = np.fft.fft(padded, axis=axis)

    n = padded.shape[axis]
    # A JAX shift also controls promotion in the mixed NumPy/JAX ramp.
    xp = jnp if isinstance(shift_pixels, Array) else np
    shift = xp.asarray(shift_pixels)
    if jnp.issubdtype(shift.dtype, jnp.floating):
        shift = _reduce_float_shift(shift, n, xp)
        shift_pixels = shift.item() if type(shift_pixels) is float else shift
    elif shift.dtype.kind in "biu":
        shift_pixels = xp.asarray(
            _reduce_integer_shift(np.asarray(shift), n, np.dtype(np.int64), np)
        )
    freqs = np.fft.fftfreq(n)
    phase_ramp = np.exp(-2j * np.pi * freqs * shift_pixels)

    if axis == 1:
        phase_ramp = np.tile(phase_ramp, (padded.shape[0], 1))
    else:
        phase_ramp = np.tile(phase_ramp, (padded.shape[1], 1)).T

    padded = padded * phase_ramp
    padded = np.fft.ifft(padded, axis=axis)
    if not np.iscomplexobj(image):
        padded = np.real(padded)
    return padded[n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]]


def fft_shift(image, x=0, y=0):
    """Apply Fourier shifts along x and/or y axes (NumPy).

    Convenience wrapper that calls fft_shift_1d for each non-zero axis.
    Rectangular images keep the input's real or complex kind.

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


def fft_shear_setup(image, pad_factor=1.5):
    """Precompute per-axis Fourier frequencies and center distances for shears.

    Frequencies use native FFT ordering. A horizontal shear pairs column
    frequencies with row distances; a vertical shear uses the converse.

    Args:
        image: 2D input image.
        pad_factor: Padding per side as a fraction of input height. Use the
            same value in the shear calls; zero operates on an already padded
            array without cropping intermediate results.

    Returns:
        Tuple ``(x_freqs, x_dists, y_freqs, y_dists)`` for the shear phase ramps.
    """
    if pad_factor < 0:
        raise ValueError("pad_factor must be nonnegative")
    _, n_pad, _, _ = get_pad_info(image, pad_factor)
    ny, nx = (size + 2 * n_pad for size in image.shape)
    dtype = jnp.result_type(image.real.dtype, jnp.float32)
    x_freqs = jnp.fft.fftfreq(nx, dtype=dtype)[None, :]
    x_dists = (jnp.arange(ny, dtype=dtype) - (ny - 1) / 2)[:, None]
    y_freqs = jnp.fft.fftfreq(ny, dtype=dtype)[:, None]
    y_dists = (jnp.arange(nx, dtype=dtype) - (nx - 1) / 2)[None, :]
    return x_freqs, x_dists, y_freqs, y_dists


def fft_shear_x(image, shear_factor, x_freqs, x_dists, pad_factor=1.5):
    """Shear an image along the x-axis via a Fourier-domain phase ramp.

    Args:
        image: 2D input image.
        shear_factor: Shear coefficient (e.g. ``tan(theta/2)`` for rotation).
        x_freqs: x frequencies from ``fft_shear_setup``.
        x_dists: Row distances from center from ``fft_shear_setup``.
        pad_factor: Padding fraction per side, matching ``fft_shear_setup``.

    Returns:
        Sheared image, same shape as input. For real input the real part is
        taken after inversion, choosing the cosine interpolant at the even
        grid's Nyquist frequency. Complex input retains its imaginary part.
    """
    _, n_pad, _, _ = get_pad_info(image, pad_factor)
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fft(padded, axis=1)
    padded = jnp.exp(-2j * jnp.pi * shear_factor * x_freqs * x_dists) * padded
    padded = jnp.fft.ifft(padded, axis=1)
    if not jnp.iscomplexobj(image):
        padded = padded.real
    return padded[n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]]


def fft_shear_y(image, shear_factor, y_freqs, y_dists, pad_factor=1.5):
    """Shear an image along the y-axis via a Fourier-domain phase ramp.

    Args:
        image: 2D input image.
        shear_factor: Shear coefficient (e.g. ``-sin(theta)`` for rotation).
        y_freqs: y frequencies from ``fft_shear_setup``.
        y_dists: Column distances from center from ``fft_shear_setup``.
        pad_factor: Padding fraction per side, matching ``fft_shear_setup``.

    Returns:
        Sheared image, same shape as input. Real input uses the same Nyquist
        cosine convention as ``fft_shear_x``.
    """
    _, n_pad, _, _ = get_pad_info(image, pad_factor)
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fft(padded, axis=0)
    padded = jnp.exp(-2j * jnp.pi * shear_factor * y_freqs * y_dists) * padded
    padded = jnp.fft.ifft(padded, axis=0)
    if not jnp.iscomplexobj(image):
        padded = padded.real
    return padded[n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]]
