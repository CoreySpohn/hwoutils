"""Tests for hwoutils.map_coordinates — cubic spline interpolation."""

import jax
import jax.numpy as jnp
import pytest
from scipy import ndimage

jax.config.update("jax_enable_x64", True)

from hwoutils.map_coordinates import map_coordinates

# =============================================================================
# Interpolation Orders
# =============================================================================


class TestInterpolationOrders:
    """Test standard interpolation orders compared to SciPy reference."""

    # We use float64 for testing comparison against scipy
    f_src = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
    )

    y_coords = jnp.array([0.5, 1.5], dtype=jnp.float64)
    x_coords = jnp.array([0.5, 1.5], dtype=jnp.float64)
    coords = [y_coords, x_coords]

    def test_nearest_neighbor(self):
        """Order 0 (nearest neighbor) should match SciPy."""
        res_jax = map_coordinates(self.f_src, self.coords, order=0)
        res_sp = ndimage.map_coordinates(self.f_src, self.coords, order=0)
        assert jnp.allclose(res_jax, res_sp)

    def test_linear(self):
        """Order 1 (linear/bilinear) should match SciPy."""
        res_jax = map_coordinates(self.f_src, self.coords, order=1)
        res_sp = ndimage.map_coordinates(self.f_src, self.coords, order=1)
        assert jnp.allclose(res_jax, res_sp)

    def test_cubic_spline(self):
        """Order 3 (cubic spline) uses custom coefficients diverging slightly from SciPy."""
        res_jax = map_coordinates(self.f_src, self.coords, order=3)
        expected = jnp.array([2.99869792, 6.58897569])
        assert jnp.allclose(res_jax, expected, rtol=1e-5)


# =============================================================================
# Dimensionality
# =============================================================================


class TestDimensionality:
    """Test interpolation across different array dimensions."""

    def test_1d_array(self):
        """1D array interpolation."""
        f_src = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])
        coords = [jnp.array([0.5, 1.5, 2.5, 3.5])]

        # Linear and nearest should match SciPy
        for order in [0, 1]:
            res_jax = map_coordinates(f_src, coords, order=order)
            res_sp = ndimage.map_coordinates(f_src, coords, order=order)
            assert jnp.allclose(res_jax, res_sp, rtol=1e-5)

        # Spline is custom evaluated
        res_cubic = map_coordinates(f_src, coords, order=3)
        expected = jnp.array([2.58333333, 6.58333333, 12.58333333, 19.83333333])
        assert jnp.allclose(res_cubic, expected, rtol=1e-5)

    def test_3d_array(self):
        """3D array interpolation."""
        f_src = jnp.arange(27).reshape((3, 3, 3)).astype(jnp.float64)
        z = jnp.array([0.5, 1.5])
        y = jnp.array([0.5, 1.5])
        x = jnp.array([0.5, 1.5])
        coords = [z, y, x]

        # Linear and nearest should match SciPy
        for order in [0, 1]:
            res_jax = map_coordinates(f_src, coords, order=order)
            res_sp = ndimage.map_coordinates(f_src, coords, order=order)
            assert jnp.allclose(res_jax, res_sp, rtol=1e-5)

        # Spline is custom evaluated
        res_cubic = map_coordinates(f_src, coords, order=3)
        expected = jnp.array([6.49165401, 17.91696506])
        assert jnp.allclose(res_cubic, expected, rtol=1e-5)


# =============================================================================
# Boundary Modes
# =============================================================================


class TestBoundaryModes:
    """Test boundary handling modes compared to SciPy."""

    f_src = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
    )
    y_coords = jnp.array([-0.5, 2.5])
    x_coords = jnp.array([-0.5, 2.5])
    coords = [y_coords, x_coords]

    def test_constant_mode(self):
        """'constant' mode with cval."""
        res_jax = map_coordinates(
            self.f_src, self.coords, order=1, mode="constant", cval=-99.0
        )
        expected = jnp.array([-74.0, -72.0])
        assert jnp.allclose(res_jax, expected)

    def test_nearest_mode(self):
        """'nearest' mode extends edge pixels."""
        res_jax = map_coordinates(self.f_src, self.coords, order=1, mode="nearest")
        res_sp = ndimage.map_coordinates(
            self.f_src, self.coords, order=1, mode="nearest"
        )
        assert jnp.allclose(res_jax, res_sp)

    def test_wrap_mode(self):
        """'wrap' mode wraps around the domain."""
        res_jax = map_coordinates(self.f_src, self.coords, order=1, mode="wrap")
        expected = jnp.array([5.0, 5.0])
        assert jnp.allclose(res_jax, expected)

    def test_mirror_mode(self):
        """'mirror' mode reflects around the centers of boundary pixels."""
        res_jax = map_coordinates(self.f_src, self.coords, order=1, mode="mirror")
        res_sp = ndimage.map_coordinates(
            self.f_src, self.coords, order=1, mode="mirror"
        )
        assert jnp.allclose(res_jax, res_sp)

    def test_reflect_mode(self):
        """'reflect' mode reflects across the edge of the domain."""
        res_jax = map_coordinates(self.f_src, self.coords, order=1, mode="reflect")
        res_sp = ndimage.map_coordinates(
            self.f_src, self.coords, order=1, mode="reflect"
        )
        assert jnp.allclose(res_jax, res_sp)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test complex edge cases that triggered previous bugs."""

    def test_cubic_spline_off_axis_points(self):
        """Test the specific 2D cubic spline bug regarding safe_zip vs itertools.product.

        The bug occurred when dimensions had multiple interpolation validities
        that needed to be mixed, typically requiring the Cartesian product
        of valid/invalid interpolation stencils near boundaries.
        """
        f_src = jnp.ones((4, 4), dtype=jnp.float64)

        # We target coordinates very far outside the domain where some
        # axes are valid and others are invalid in the stencil, forcing the
        # Cartesian product contribution logic to fire.
        y_coords = jnp.array([4.5, -1.0, 1.5, 10.0])
        x_coords = jnp.array([1.5, 10.0, 4.5, -1.0])
        coords = [y_coords, x_coords]

        res_jax = map_coordinates(f_src, coords, order=3, mode="constant", cval=0.0)

        expected = jnp.array([0.02083333, 0.0, 0.02083333, 0.0])
        assert jnp.allclose(res_jax, expected, atol=1e-5)
