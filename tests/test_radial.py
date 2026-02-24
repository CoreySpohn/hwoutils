"""Tests for hwoutils.radial — radial distance and profile utilities."""

import jax.numpy as jnp

from hwoutils.radial import radial_distance, radial_profile

# =============================================================================
# Radial Distance
# =============================================================================


class TestRadialDistance:
    """Tests for radial distance grid generation."""

    def test_center_is_zero(self):
        """Distance at the geometric center should be zero."""
        r = radial_distance((11, 11))
        assert jnp.isclose(r[5, 5], 0.0, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match input shape."""
        assert radial_distance((32, 64)).shape == (32, 64)

    def test_symmetry(self):
        """Distance grid should be symmetric about center."""
        r = radial_distance((21, 21))
        assert jnp.allclose(r, jnp.flip(r, axis=0), atol=1e-10)
        assert jnp.allclose(r, jnp.flip(r, axis=1), atol=1e-10)

    def test_custom_center(self):
        """Custom center should shift the zero point."""
        r = radial_distance((11, 11), center=(2.0, 3.0))
        assert jnp.isclose(r[2, 3], 0.0, atol=1e-10)
        # Corner should be far from custom center
        assert r[0, 0] > 0

    def test_corner_distance(self):
        """Corner distance should equal sqrt(5^2 + 5^2) for 11x11 grid."""
        r = radial_distance((11, 11))
        expected = jnp.sqrt(5.0**2 + 5.0**2)
        assert jnp.isclose(r[0, 0], expected, atol=1e-6)

    def test_even_dimensions(self):
        """Even-sized image should put center at (N-1)/2."""
        r = radial_distance((10, 10))
        # Center at (4.5, 4.5), equidistant from [4,4] and [5,5]
        assert jnp.isclose(r[4, 4], r[5, 5], atol=1e-10)


# =============================================================================
# Radial Profile
# =============================================================================


class TestRadialProfile:
    """Tests for radial profile extraction."""

    def test_uniform_image_flat_profile(self):
        """Uniform image should have a flat radial profile."""
        image = jnp.ones((64, 64)) * 42.0
        _, profile = radial_profile(image)
        assert jnp.allclose(profile, 42.0, atol=0.5)

    def test_output_shapes_match(self):
        """Separations and profile should have the same length."""
        image = jnp.ones((32, 32))
        seps, prof = radial_profile(image, nbins=10)
        assert seps.shape == (10,)
        assert prof.shape == (10,)

    def test_pixel_scale_scaling(self):
        """Pixel scale should scale the separation axis."""
        image = jnp.ones((32, 32))
        seps1, _ = radial_profile(image, pixel_scale=1.0, nbins=10)
        seps2, _ = radial_profile(image, pixel_scale=0.25, nbins=10)
        assert jnp.allclose(seps2, seps1 * 0.25, atol=1e-6)

    def test_gaussian_decreasing(self):
        """Gaussian image should have a monotonically decreasing profile."""
        N = 64
        c = (N - 1) / 2.0
        y, x = jnp.ogrid[:N, :N]
        image = jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * 5.0**2))
        _seps, prof = radial_profile(image, nbins=20)
        # Profile should generally decrease (allow small noise in last bins)
        for i in range(1, 15):
            assert prof[i] <= prof[i - 1] + 1e-6

    def test_zero_image_zero_profile(self):
        """Zero image should produce zero profile."""
        image = jnp.zeros((32, 32))
        _, profile = radial_profile(image)
        assert jnp.allclose(profile, 0.0)
