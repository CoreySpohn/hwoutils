"""Unit tests for hwoutils.fft sub-pixel shifting functions.

Tests the core mathematical properties of the FFT phasor shift:
shift fidelity, flux conservation, Parseval's theorem, positivity
clamp behavior, and padding arithmetic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import fft
from hwoutils.fft import fft_shift, fft_shift_1d, get_pad_info


@pytest.fixture()
def gaussian_psf():
    """A synthetic Gaussian PSF for shift tests (no coronagraph needed)."""
    n = 64
    y, x = np.mgrid[:n, :n]
    cx, cy = n / 2, n / 2
    sigma = 5.0
    psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


@pytest.fixture()
def periodic_padding(monkeypatch):
    # Exact wave shifts and round trips require periodic boundaries. Production
    # zero padding and cropping are covered separately against a direct DFT.
    numpy_pad = np.pad

    def pad_numpy(image, pad_width, mode):
        return numpy_pad(image, pad_width, mode="wrap")

    def pad_jax(image, value, config):
        return jnp.pad(image, [(lo, hi) for lo, hi, _ in config], mode="wrap")

    monkeypatch.setattr(fft.np, "pad", pad_numpy)
    monkeypatch.setattr(fft.lax, "pad", pad_jax)


def _shift(image, shift_pixels, axis, backend):
    if backend == "numpy":
        return fft_shift_1d(image, shift_pixels, axis)
    _, n_pad, _, _ = get_pad_info(image, 1.5)
    phasor = jnp.exp(-2j * jnp.pi * jnp.fft.fftfreq(image.shape[axis] + 2 * n_pad))
    func = fft.fft_shift_x if axis == 1 else fft.fft_shift_y
    return func(jnp.asarray(image), shift_pixels, phasor, clamp=False)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("axis", [0, 1])
def test_complex_plane_wave_periodic(periodic_padding, backend, axis):
    shape = (32, 48)
    coordinate = np.indices(shape)[axis]
    wave = np.exp(2j * np.pi * 3 * coordinate / shape[axis])
    expected = np.exp(2j * np.pi * 3 * (coordinate - 2.5) / shape[axis])
    with jax.enable_x64():
        shifted = _shift(wave, 2.5, axis, backend)
    assert shifted.shape == shape
    assert shifted.dtype == np.complex128
    np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("shape", [(32, 48), (48, 32)])
def test_real_rectangular_roundtrip_periodic(periodic_padding, backend, shape):
    y, x = np.indices(shape)
    image = 1 + np.cos(2 * np.pi * 3 * y / shape[0]) * np.sin(
        2 * np.pi * 2 * x / shape[1]
    )
    with jax.enable_x64():
        shifted = _shift(image, 1.25, 0, backend)
        roundtrip = _shift(shifted, -1.25, 0, backend)
    assert roundtrip.shape == shape
    assert roundtrip.dtype == np.float64
    np.testing.assert_allclose(roundtrip, image, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("complex_input", [False, True])
def test_rectangular_zero_padding_direct_dft(backend, axis, complex_input):
    y, x = np.indices((8, 12))
    image = np.cos(x + y / 3)
    if complex_input:
        image = image + 1j * np.sin(x / 2 - y)
    n_pad = 12
    padded = np.pad(image, n_pad)
    n = padded.shape[axis]
    samples = np.arange(n)
    freqs = np.fft.fftfreq(n)
    # Evaluate the zero-extended trigonometric interpolant at shifted samples
    # using explicit Fourier sums, independently of the FFT implementation.
    forward = np.exp(-2j * np.pi * freqs[:, None] * samples)
    inverse = np.exp(2j * np.pi * (samples[:, None] - 1.25) * freqs) / n
    expected = np.moveaxis(inverse @ (forward @ np.moveaxis(padded, axis, 0)), 0, axis)[
        n_pad : n_pad + image.shape[0], n_pad : n_pad + image.shape[1]
    ]
    if not complex_input:
        expected = expected.real
    with jax.enable_x64():
        shifted = _shift(image, 1.25, axis, backend)
    assert shifted.shape == image.shape
    assert np.iscomplexobj(shifted) == complex_input
    np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-12)


def test_two_axis_complex_wave_periodic(periodic_padding):
    y, x = np.indices((32, 48))
    wave = np.exp(2j * np.pi * (3 * x / 48 + 2 * y / 32))
    expected = np.exp(2j * np.pi * (3 * (x - 2.5) / 48 + 2 * (y + 1.25) / 32))
    shifted = fft_shift(wave, x=2.5, y=-1.25)
    assert shifted.shape == wave.shape
    assert shifted.dtype == np.complex128
    np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("func", [fft.fft_shift_x, fft.fft_shift_y])
def test_complex_clamp_rejected_on_host(func):
    # No phasor is needed: reject the dtype before any FFT is dispatched.
    with pytest.raises(TypeError, match="Cannot clamp a complex image"):
        func(jnp.ones((32, 48), dtype=jnp.complex64), 2.5, None)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("axis", [0, 1])
def test_real_square_bitwise_base(gaussian_psf, backend, axis):
    # Frozen square-real implementation from 121d2c1. Keep its operation order
    # to detect any rounding change, including the JAX clamp.
    xp = np if backend == "numpy" else jnp
    with jax.enable_x64():
        image = xp.asarray(gaussian_psf)
        _, n_pad, edge, n_final = get_pad_info(image, 1.5)
        padded = xp.pad(image, n_pad, mode="constant")
        spectrum = xp.fft.fft(padded, axis=axis)
        freqs = xp.fft.fftfreq(n_final)
        if backend == "numpy":
            phasor = xp.exp(-2j * xp.pi * freqs * 1.25)
        else:
            phasor = xp.exp(-2j * xp.pi * freqs) ** 1.25
        phasor = xp.tile(phasor, (n_final, 1))
        if axis == 0:
            phasor = phasor.T
        expected = xp.real(xp.fft.ifft(spectrum * phasor, axis=axis))
        expected = expected[n_pad:edge, n_pad:edge]
        actual = _shift(image, 1.25, axis, backend)
        np.testing.assert_array_equal(actual, expected)
        if backend == "jax":
            func = fft.fft_shift_x if axis == 1 else fft.fft_shift_y
            phasor = xp.exp(-2j * xp.pi * freqs)
            clamped = func(image, 1.25, phasor)
            np.testing.assert_array_equal(clamped, xp.maximum(expected, 0.0))


class TestShiftFidelity:
    """Verify the phasor-multiply shift preserves signal integrity."""

    def test_integer_shift_matches_roll(self, gaussian_psf):
        """Integer pixel shift via FFT must match np.roll exactly."""
        for shift in [1, -1, 3, -5]:
            shifted = fft_shift(gaussian_psf, x=shift, y=0)
            rolled = np.roll(gaussian_psf, shift, axis=1)
            margin = abs(shift) + 2
            np.testing.assert_allclose(
                shifted[margin:-margin, margin:-margin],
                rolled[margin:-margin, margin:-margin],
                atol=1e-10,
                err_msg=f"Integer shift by {shift} failed",
            )

    def test_roundtrip_subpixel(self, gaussian_psf):
        """Shift by +delta then -delta should recover the original."""
        for delta in [0.3, 0.5, 0.7, 1.4]:
            shifted_fwd = fft_shift(gaussian_psf, x=delta, y=0)
            roundtrip = fft_shift(shifted_fwd, x=-delta, y=0)
            mask = gaussian_psf > 0.05 * np.max(gaussian_psf)
            np.testing.assert_allclose(
                roundtrip[mask],
                gaussian_psf[mask],
                rtol=0.05,
                err_msg=f"Roundtrip failed for delta={delta}",
            )

    def test_shift_accumulation(self, gaussian_psf):
        """Shift(+0.3) then Shift(+0.7) must equal Shift(+1.0)."""
        two_step = fft_shift(fft_shift(gaussian_psf, x=0.3, y=0), x=0.7, y=0)
        one_step = fft_shift(gaussian_psf, x=1.0, y=0)
        mask = one_step > 0.05 * np.max(one_step)
        np.testing.assert_allclose(
            two_step[mask],
            one_step[mask],
            rtol=0.05,
            err_msg="Shift accumulation (0.3+0.7 vs 1.0) failed",
        )

    def test_shift_preserves_total_flux(self, gaussian_psf):
        """FFT shift must not change total flux by more than 5%."""
        original_flux = np.sum(gaussian_psf)
        for delta in [0.1, 0.5, 1.3, 3.7]:
            shifted = fft_shift(gaussian_psf, x=delta, y=0)
            shifted_flux = np.sum(shifted)
            np.testing.assert_allclose(
                shifted_flux,
                original_flux,
                rtol=0.05,
                err_msg=f"Flux changed after shift by {delta}",
            )

    def test_y_shift_works(self, gaussian_psf):
        """Shifting along y-axis should be equivalent to transposed x-shift."""
        shifted_y = fft_shift(gaussian_psf, x=0, y=1.3)
        shifted_x = fft_shift(gaussian_psf.T, x=1.3, y=0).T
        mask = gaussian_psf > 0.05 * np.max(gaussian_psf)
        np.testing.assert_allclose(
            shifted_y[mask],
            shifted_x[mask],
            rtol=0.05,
            err_msg="Y-shift does not match transposed X-shift",
        )


class TestFftShift1d:
    """Verify the 1D NumPy FFT shift primitive."""

    def test_axis_parameter(self, gaussian_psf):
        """fft_shift_1d along axis=1 should match fft_shift(x=...)."""
        shifted_via_1d = fft_shift_1d(gaussian_psf, 1.3, axis=1)
        shifted_via_wrapper = fft_shift(gaussian_psf, x=1.3, y=0)
        np.testing.assert_allclose(
            shifted_via_1d,
            shifted_via_wrapper,
            atol=1e-12,
            err_msg="fft_shift_1d(axis=1) differs from fft_shift(x=...)",
        )


class TestPositivityClamp:
    """Verify positivity clamp behavior on JAX fft_shift_x."""

    def test_clamp_flux_loss_is_negligible(self, gaussian_psf):
        """Energy eliminated by clamping must be < 1e-4 of total energy."""
        import jax
        import jax.numpy as jnp

        from hwoutils.fft import fft_shift_x

        with jax.enable_x64():
            psf = jnp.array(gaussian_psf, dtype=jnp.float64)
            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)

            unclamped = np.array(fft_shift_x(psf, 0.37, phasor, clamp=False))
            clamped = np.array(fft_shift_x(psf, 0.37, phasor, clamp=True))

        total_energy = np.sum(unclamped**2)
        neg_mask = unclamped < 0
        neg_energy = np.sum(unclamped[neg_mask] ** 2)
        neg_fraction = neg_energy / total_energy

        assert neg_fraction < 1e-4, (
            f"Negative pixel energy is {neg_fraction:.2e} of total (limit: 1e-4)"
        )
        # Clamped output should match unclamped with negatives zeroed,
        # allowing for machine epsilon differences
        np.testing.assert_allclose(clamped, np.maximum(unclamped, 0), atol=1e-15)


class TestParsevalTheorem:
    """Verify FFT shift preserves energy (Parseval's theorem)."""

    def test_phasor_shift_preserves_energy_exactly(self, gaussian_psf):
        """fft_shift_x(clamp=False) must preserve energy exactly."""
        import jax
        import jax.numpy as jnp

        from hwoutils.fft import fft_shift_x

        with jax.enable_x64():
            psf = jnp.array(gaussian_psf, dtype=jnp.float64)
            energy_before = float(jnp.sum(psf**2))

            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)
            shifted = fft_shift_x(psf, 0.37, phasor, clamp=False)
            energy_after = float(jnp.sum(shifted**2))

        np.testing.assert_allclose(
            energy_after,
            energy_before,
            rtol=1e-5,
            err_msg="Phasor shift violated energy conservation",
        )

    def test_clamp_energy_loss_is_small(self, gaussian_psf):
        """fft_shift_x(clamp=True) should lose less than 1% energy."""
        import jax
        import jax.numpy as jnp

        from hwoutils.fft import fft_shift_x

        with jax.enable_x64():
            psf = jnp.array(gaussian_psf, dtype=jnp.float64)
            energy_before = float(jnp.sum(psf**2))

            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)
            shifted = fft_shift_x(psf, 0.37, phasor, clamp=True)
            energy_after = float(jnp.sum(shifted**2))

        loss = 1.0 - energy_after / energy_before
        assert loss < 0.01, f"Clamp lost {loss:.4%} of energy (limit: 1%)"
        assert loss >= -1e-14, f"Energy increased by {-loss:.2e} -- beyond rounding"


class TestFloat32VsFloat64:
    """Quantify precision noise floor from float32 vs float64."""

    def test_shift_precision_with_x64(self, gaussian_psf):
        """JAX fft_shift in float32 vs float64 should agree."""
        import jax
        import jax.numpy as jnp

        from hwoutils.fft import fft_shift_x

        psf_f32 = jnp.array(gaussian_psf, dtype=jnp.float32)
        _, _, _, n_final = get_pad_info(psf_f32, 1.5)
        kx_f32 = jnp.fft.fftfreq(n_final)
        phasor_f32 = jnp.exp(-2j * jnp.pi * kx_f32)
        result_f32 = np.array(fft_shift_x(psf_f32, 0.37, phasor_f32))

        with jax.enable_x64():
            psf_f64 = jnp.array(gaussian_psf, dtype=jnp.float64)
            _, _, _, n_final_64 = get_pad_info(psf_f64, 1.5)
            kx_f64 = jnp.fft.fftfreq(n_final_64)
            phasor_f64 = jnp.exp(-2j * jnp.pi * kx_f64)
            result_f64 = np.array(fft_shift_x(psf_f64, 0.37, phasor_f64))

        peak = np.max(np.abs(result_f64))
        max_diff = np.max(np.abs(result_f64 - result_f32.astype(np.float64)))
        relative_diff = max_diff / peak

        assert relative_diff < 1e-4, (
            f"float32 vs float64 relative diff {relative_diff:.2e} exceeds 1e-4"
        )


class TestGetPadInfo:
    """Verify padding dimensions from get_pad_info."""

    def test_standard_padding(self):
        """Default 1.5x padding on a 10x10 image."""
        import jax.numpy as jnp

        image = jnp.zeros((10, 10))
        n_orig, n_pad, img_edge, n_final = get_pad_info(image, 1.5)

        assert n_orig == 10
        assert n_pad == 15
        assert img_edge == 25
        assert n_final == 40

    def test_real_psf_size(self, gaussian_psf):
        """Padding on a real PSF shape."""
        n_orig, n_pad, img_edge, n_final = get_pad_info(gaussian_psf, 1.5)

        assert n_orig == gaussian_psf.shape[0]
        assert n_pad == int(1.5 * n_orig)
        assert img_edge == n_pad + n_orig
        assert n_final == 2 * n_pad + n_orig

    def test_different_pad_factors(self):
        """Various pad factors should scale correctly."""
        import jax.numpy as jnp

        image = jnp.zeros((20, 20))

        for factor in [1.0, 1.5, 2.0, 3.0]:
            n_orig, n_pad, img_edge, _n_final = get_pad_info(image, factor)
            assert n_orig == 20
            assert n_pad == int(factor * 20)
            assert img_edge == n_pad + n_orig
