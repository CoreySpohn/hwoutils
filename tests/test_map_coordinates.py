"""Tests for hwoutils.map_coordinates -- cubic spline interpolation."""

import jax.numpy as jnp
import pytest
from scipy import ndimage

from hwoutils.map_coordinates import map_coordinates

# =============================================================================
# Interpolation Orders
# =============================================================================


class TestInterpolationOrders:
    """Test standard interpolation orders compared to SciPy reference."""

    f_src = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    y_coords = jnp.array([0.5, 1.5])
    x_coords = jnp.array([0.5, 1.5])
    coords = (y_coords, x_coords)

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

    def test_cubic(self):
        """Order 3 uses the Keys cubic convolution kernel (a = -0.5).

        Expected values computed by hand from the Keys kernel on the 3x3
        matrix ``[[1,2,3],[4,5,6],[7,8,9]]`` at (0.5, 0.5) and (1.5, 1.5)
        with mode='constant', cval=0.0. At each coord the 4-tap stencil
        samples at indices {-1, 0, 1, 2}; out-of-bounds entries are 0.

        Keys weights at offset 0.5:  K(1.5), K(0.5), K(-0.5), K(-1.5)
                                  =  -0.0625,  0.5625,  0.5625, -0.0625
        """
        res_jax = map_coordinates(self.f_src, self.coords, order=3)
        expected = jnp.array([2.98828125, 8.30078125])
        assert jnp.allclose(res_jax, expected, rtol=1e-10)

    def test_cubic_identity_at_integer_coords(self):
        """Keys is a true interpolant: integer coords return sample values.

        K(0) = 1 and K(k) = 0 for k in {-2, -1, 1, 2}, so at an integer
        grid point the kernel selects a single sample exactly.
        """
        f = jnp.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 80.0],
                [11.0, 22.0, 33.0, 44.0],
                [55.0, 66.0, 77.0, 88.0],
            ]
        )
        # Interior integer coords - no boundary effects with 4-tap stencil
        coords = (jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
        res = map_coordinates(f, coords, order=3)
        assert jnp.allclose(res, jnp.array([60.0, 33.0]), rtol=1e-12)

    @pytest.mark.float64_only(reason="Cubic partition-of-unity reference at rtol=1e-12")
    def test_cubic_partition_of_unity(self):
        """Sum of Keys weights over integer shifts equals 1 at any offset.

        For any real offset t in [0, 1), the 4 Keys weights
        {K(t+1), K(t), K(t-1), K(t-2)} must sum to 1. This is the
        partition-of-unity property that reproduces constant fields; it
        does not establish conservative downsampling.
        """
        f = jnp.ones((8, 8))
        # Interior coords (stencil fully inside the array)
        coords = (
            jnp.array([2.0, 2.3, 2.5, 2.7, 3.9]),
            jnp.array([4.0, 4.1, 4.5, 4.8, 5.0]),
        )
        res = map_coordinates(f, coords, order=3)
        assert jnp.allclose(res, 1.0, rtol=1e-12)


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

        # Keys cubic convolution
        res_cubic = map_coordinates(f_src, coords, order=3)
        expected = jnp.array([2.25, 6.25, 12.25, 22.5])
        assert jnp.allclose(res_cubic, expected, rtol=1e-10)

    def test_3d_array(self, precision):
        """3D array interpolation."""
        f_src = jnp.arange(27).reshape((3, 3, 3)).astype(precision)
        z = jnp.array([0.5, 1.5])
        y = jnp.array([0.5, 1.5])
        x = jnp.array([0.5, 1.5])
        coords = [z, y, x]

        # Linear and nearest should match SciPy
        for order in [0, 1]:
            res_jax = map_coordinates(f_src, coords, order=order)
            res_sp = ndimage.map_coordinates(f_src, coords, order=order)
            assert jnp.allclose(res_jax, res_sp, rtol=1e-5)

        # Keys cubic convolution
        res_cubic = map_coordinates(f_src, coords, order=3)
        expected = jnp.array([6.4206543, 24.76538086])
        assert jnp.allclose(res_cubic, expected, rtol=1e-7)


# =============================================================================
# Boundary Modes
# =============================================================================


class TestBoundaryModes:
    """Test boundary handling modes compared to SciPy."""

    f_src = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y_coords = jnp.array([-0.5, 2.5])
    x_coords = jnp.array([-0.5, 2.5])
    coords = (y_coords, x_coords)

    def test_constant_mode(self):
        """'constant' mode with cval."""
        res_jax = map_coordinates(
            self.f_src, self.coords, order=1, mode="constant", cval=-99.0
        )
        expected = jnp.array([-74.0, -72.0])
        assert jnp.allclose(res_jax, expected)
        # A NaN fill marks the exterior only: edge pixel centres keep their
        # stored values, as in scipy, while points outside stay NaN.
        n_y, n_x = self.f_src.shape
        edges = map_coordinates(
            self.f_src,
            (jnp.array([0.0, n_y - 1.0, n_y - 0.5]), jnp.array([0.0, n_x - 1.0, 0.0])),
            order=1,
            mode="constant",
            cval=jnp.nan,
        )
        assert edges[0] == self.f_src[0, 0]
        assert edges[1] == self.f_src[-1, -1]
        assert jnp.isnan(edges[2])

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

    def test_cubic_off_axis_points(self):
        """Cartesian product of valid/invalid stencils near boundaries.

        Regression test: the original bug mixed per-axis validity with
        ``safe_zip`` instead of ``itertools.product``, producing wrong
        values whenever stencils straddled a boundary along different
        axes. This test targets coordinates where some axes are fully
        in-bounds and others fully out-of-bounds, forcing the Cartesian
        product contribution logic to fire. The expected -0.0625 values
        come from the Keys outer-lobe weight K(1.5) = -0.0625 applied to
        a single in-bounds unit-valued stencil tap on one axis while the
        other axis contributes partition-of-unity weight 1.
        """
        f_src = jnp.ones((4, 4))

        y_coords = jnp.array([4.5, -1.0, 1.5, 10.0])
        x_coords = jnp.array([1.5, 10.0, 4.5, -1.0])
        coords = [y_coords, x_coords]

        res_jax = map_coordinates(f_src, coords, order=3, mode="constant", cval=0.0)

        expected = jnp.array([-0.0625, 0.0, -0.0625, 0.0])
        assert jnp.allclose(res_jax, expected, rtol=1e-10)
