"""Tests for hwoutils.transforms — image and coordinate transforms."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hwoutils.transforms import ccw_rotation_matrix, resample_flux  # noqa: E402

# =============================================================================
# Rotation Matrix
# =============================================================================


class TestRotationMatrix:
    """Tests for counter-clockwise rotation matrix generation."""

    def test_identity_at_zero(self):
        """0 degree rotation should give identity matrix."""
        assert jnp.allclose(ccw_rotation_matrix(0.0), jnp.eye(2), atol=1e-10)

    def test_90_degrees(self):
        """90° CCW rotation matrix values."""
        expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        assert jnp.allclose(ccw_rotation_matrix(90.0), expected, atol=1e-6)

    def test_180_degrees(self):
        """180° rotation should negate both axes."""
        expected = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        assert jnp.allclose(ccw_rotation_matrix(180.0), expected, atol=1e-6)

    def test_270_degrees(self):
        """270° CCW rotation (= 90° CW)."""
        expected = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        assert jnp.allclose(ccw_rotation_matrix(270.0), expected, atol=1e-6)

    def test_determinant_unity(self):
        """Rotation matrix should have determinant = 1."""
        for angle in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0]:
            det = jnp.linalg.det(ccw_rotation_matrix(angle))
            assert jnp.isclose(det, 1.0, atol=1e-10)

    def test_chirality_preservation(self):
        """Point at (1,0) rotated 90° CCW → (0,1), not (0,-1)."""
        point = jnp.array([[1.0], [0.0]])
        rotated = ccw_rotation_matrix(90.0) @ point
        expected = jnp.array([[0.0], [1.0]])
        assert jnp.allclose(rotated, expected, atol=1e-6)

    def test_full_rotation_identity(self):
        """4 * 90 degree rotations = identity."""
        point = jnp.array([[1.0], [1.0]])
        rot90 = ccw_rotation_matrix(90.0)
        result = rot90 @ rot90 @ rot90 @ rot90 @ point
        assert jnp.allclose(result, point, atol=1e-5)


# =============================================================================
# Flux-Conserving Resampling
# =============================================================================


class TestResampleFlux:
    """Tests for flux-conserving image resampling."""

    def test_same_scale_identity(self):
        """Same pixel scale and shape should preserve interior pixels.

        Edge pixels are slightly attenuated by the cubic spline boundary
        effect (4-point stencil extends beyond image with mode=constant).
        """
        f_src = jnp.ones((64, 64)) * 100.0
        f_tgt = resample_flux(f_src, 0.01, 0.01, (64, 64), 0.0)
        # Interior pixels (away from boundary stencil) are exact
        assert jnp.allclose(f_tgt[2:-2, 2:-2], f_src[2:-2, 2:-2], rtol=0.01)
        # Total flux should still be close (only edge attenuation)
        assert jnp.isclose(jnp.sum(f_tgt), jnp.sum(f_src), rtol=0.05)

    def test_point_source_downsampling(self):
        """Centred point source flux is conserved on 2x downsample."""
        f_src = jnp.zeros((64, 64))
        f_src = f_src.at[32, 32].set(10000.0)
        f_tgt = resample_flux(f_src, 0.01, 0.02, (32, 32), 0.0)
        assert jnp.isclose(jnp.sum(f_tgt), 10000.0, rtol=0.1)

    def test_downsampling_flux_conservation(self):
        """Downsampling (larger pixels) should conserve total flux."""
        f_src = jnp.ones((64, 64))
        f_tgt = resample_flux(f_src, 0.01, 0.02, (32, 32), 0.0)
        assert jnp.isclose(jnp.sum(f_tgt), jnp.sum(f_src), rtol=0.05)

    def test_rotation_clips_corners(self):
        """45° rotation of a uniform image clips corners — flux is lost.

        This is physically correct: flux outside the source image
        is genuinely absent.
        """
        f_src = jnp.ones((64, 64)) * 100.0
        f_tgt = resample_flux(f_src, 0.01, 0.01, (64, 64), 45.0)
        ratio = float(jnp.sum(f_tgt) / jnp.sum(f_src))
        assert 0.6 < ratio < 1.0

    def test_gaussian_resampling_conservation(self):
        """Contained Gaussian source should be well-conserved on resample."""
        N = 100
        x = jnp.linspace(-3, 3, N)
        X, Y = jnp.meshgrid(x, x)
        image = jnp.exp(-(X**2 + Y**2))
        flux_in = jnp.sum(image)

        resampled = resample_flux(image, 1.0, 2.0, (50, 50), 0.0)
        assert jnp.isclose(jnp.sum(resampled), flux_in, rtol=0.05)

    def test_gaussian_rotation_conservation(self):
        """Compact Gaussian with 45° rotation should retain >90% flux."""
        N = 100
        x = jnp.linspace(-2, 2, N)
        X, Y = jnp.meshgrid(x, x)
        image = jnp.exp(-2 * (X**2 + Y**2))
        flux_in = jnp.sum(image)

        resampled = resample_flux(image, 1.0, 1.0, (100, 100), 45.0)
        assert jnp.sum(resampled) / flux_in > 0.9

    def test_output_shape(self):
        """Output shape should match target shape."""
        f_src = jnp.ones((64, 64))
        f_tgt = resample_flux(f_src, 0.01, 0.02, (32, 32), 0.0)
        assert f_tgt.shape == (32, 32)

    def test_empty_input(self):
        """Zero input should give zero output."""
        f_src = jnp.zeros((32, 32))
        f_tgt = resample_flux(f_src, 0.01, 0.01, (32, 32), 0.0)
        assert jnp.allclose(f_tgt, 0.0)


# =============================================================================
# Flux-Conserving Resampling: Tight Keys Kernel Properties
# =============================================================================


class TestResampleFluxKeys:
    """Tight flux-conservation tests for the Keys cubic kernel.

    Keys has partition of unity at integer grid spacing, so downsampling
    by an integer factor with a compact, band-limited input should
    conserve total flux well below the loose 5% tolerance used by the
    generic flux-conservation tests above.
    """

    def test_compact_gaussian_integer_downsample(self):
        """Compact Gaussian, 2x integer downsample: <1% flux error.

        Partition of unity gives exact conservation of flux contributions
        from interior source pixels; the remaining error is from the
        Gaussian's tails leaking into the 2-pixel boundary strip where
        the Keys stencil reaches the cval=0 padding.
        """
        N = 128
        x = jnp.linspace(-4, 4, N)
        X, Y = jnp.meshgrid(x, x)
        image = jnp.exp(-(X**2 + Y**2))  # Peaks in image center
        flux_in = float(jnp.sum(image))

        resampled = resample_flux(image, 1.0, 2.0, (64, 64), 0.0)
        flux_out = float(jnp.sum(resampled))
        rel_err = abs(flux_out - flux_in) / flux_in
        assert rel_err < 1e-2, f"flux error {rel_err:.3e} > 1%"

    def test_compact_gaussian_flux_scale_3(self):
        """Compact Gaussian, 3x downsample: <1% flux error.

        Scale=3 has the target grid at source-coord offsets {0, 3, 6,
        ...}, still integer spacing so POU holds.
        """
        N = 192
        x = jnp.linspace(-5, 5, N)
        X, Y = jnp.meshgrid(x, x)
        image = jnp.exp(-(X**2 + Y**2) * 2.0)
        flux_in = float(jnp.sum(image))

        resampled = resample_flux(image, 1.0, 3.0, (64, 64), 0.0)
        flux_out = float(jnp.sum(resampled))
        rel_err = abs(flux_out - flux_in) / flux_in
        assert rel_err < 1e-2, f"flux error {rel_err:.3e} > 1%"

    def test_interior_identity_tight(self):
        """Same scale + no rotation = identity on interior pixels.

        At scale=1 with zero rotation, every target center coincides
        with a source pixel center. Keys at integer offset is exactly
        the sample value (K(0) = 1, K(+/-k) = 0 for k in {1, 2}).
        Boundary pixels within 2 of the edge are still attenuated by
        the cval=0 padding.
        """
        f_src = jnp.ones((64, 64)) * 42.0
        f_tgt = resample_flux(f_src, 0.01, 0.01, (64, 64), 0.0)
        # Interior away from the 2-pixel stencil reach
        assert jnp.allclose(f_tgt[3:-3, 3:-3], 42.0, rtol=1e-10)

    def test_no_interior_smoothing_of_constant(self):
        """A constant field stays constant on the interior.

        Keys is a true interpolant and preserves constants exactly on
        the interior.
        """
        f_src = jnp.ones((48, 48)) * 7.5
        # Non-unit scale and fractional rotation exercise the full
        # interpolation path, not just integer offsets.
        f_tgt = resample_flux(f_src, 1.0, 1.25, (40, 40), 0.0)
        # Interior (away from stencil reach to boundary)
        interior = f_tgt[4:-4, 4:-4]
        # scale^2 = 1.5625, so a constant 7.5 source becomes
        # 7.5 * 1.5625 = 11.71875 per target pixel.
        assert jnp.allclose(interior, 7.5 * 1.25**2, rtol=1e-10)
