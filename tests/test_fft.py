"""Unit tests for hwoutils.fft sub-pixel shifting functions.

Tests the core mathematical properties of the FFT phasor shift:
shift fidelity, flux conservation, Parseval's theorem, positivity
clamp behavior, and padding arithmetic.
"""

import numpy as np
import pytest

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
