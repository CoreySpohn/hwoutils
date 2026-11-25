"""Regression tests for transform geometry and integrated-pixel semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.fft import fft_shear_setup, fft_shear_x, fft_shift_1d, get_pad_info
from hwoutils.map_coordinates import map_coordinates
from hwoutils.transforms import rotate_image


@pytest.mark.float64_only(reason="Analytic Gaussian reference at rtol=1e-9")
@pytest.mark.parametrize("n", [63, 64])
@pytest.mark.parametrize("angle", [0.0, 30.0, 44.999, 45.001, 90.0, -30.0])
def test_rotation_matches_analytic_gaussian(n, angle):
    """A smooth displaced source rotates about (N-1)/2 without a pixel offset."""
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    source = np.exp(-((y + 4) ** 2 + (x - 6) ** 2) / 18)
    theta = np.deg2rad(angle)
    cx = 6 * np.cos(theta) + 4 * np.sin(theta)
    cy = 6 * np.sin(theta) - 4 * np.cos(theta)
    expected = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 18)
    actual = rotate_image(jnp.asarray(source), angle)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.float64_only(reason="Analytic angle-derivative reference at atol=1e-9")
@pytest.mark.parametrize("angle", [0.0, 90.0, -90.0, 180.0])
def test_rotation_angle_derivative_at_right_angles(angle):
    """An exact array turn must not erase the local angle derivative."""
    y, x = jnp.mgrid[:64, :64] - 31.5
    source = jnp.exp(-((y + 4) ** 2 + (x - 6) ** 2) / 18)

    def moment(degrees):
        rotated = rotate_image(source, degrees)
        return (rotated * y).sum() / rotated.sum()

    expected = (
        np.pi / 180 * (6 * np.cos(np.deg2rad(angle)) + 4 * np.sin(np.deg2rad(angle)))
    )
    np.testing.assert_allclose(jax.grad(moment)(angle), expected, atol=1e-9)


@pytest.mark.parametrize("n", [31, 32])
def test_horizontal_shear_uses_row_distance(n):
    """The horizontal displacement is a*y, not a function of column index."""
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    source = jnp.asarray(np.exp(-((y - 2) ** 2 + (x + 3) ** 2) / 8))
    fx, dy, _, _ = fft_shear_setup(source)
    actual = fft_shear_x(source, 0.25, fx, dy)
    expected = np.exp(-((y - 2) ** 2 + (x - 0.25 * y + 3) ** 2) / 8)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_padding_reports_actual_odd_size():
    """The frequency grid length must equal the actual padded array length."""
    _, pad, _, size = get_pad_info(jnp.zeros((31, 31)), 0.5)
    assert size == 31 + 2 * pad


@pytest.mark.parametrize("shape", [(31, 31), (31, 32)])
@pytest.mark.parametrize("axis", [0, 1])
def test_numpy_shift_uses_actual_fft_length(shape, axis):
    """Odd and rectangular inputs must not assume a 4N frequency vector."""
    source = np.zeros(shape)
    source[12, 13] = 1.0
    expected = np.zeros(shape)
    expected[12 + (axis == 0), 13 + (axis == 1)] = 1.0
    actual = fft_shift_1d(source, 1.0, axis)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("order,expected", [(0, 10), (1, 6), (3, 6)])
def test_integer_samples_keep_fractional_coordinates(order, expected):
    """Sample dtype must not truncate coordinates before interpolation."""
    source = jnp.array([-10, 0, 10, 20, 30])
    result = map_coordinates(source, [jnp.array([1.6])], order=order)
    np.testing.assert_array_equal(result, [expected])


@pytest.mark.parametrize("angle", [30.0, 90.0, 180.0])
def test_rotation_about_optical_center(angle, precision):
    """FITS optical centers need not equal the geometric array center."""
    y, x = np.mgrid[:64, :64]
    source = np.exp(-((y - 28) ** 2 + (x - 38) ** 2) / 18)
    theta = np.deg2rad(angle)
    cx = 32 + 6 * np.cos(theta) + 4 * np.sin(theta)
    cy = 32 + 6 * np.sin(theta) - 4 * np.cos(theta)
    expected = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 18)
    actual = rotate_image(
        jnp.asarray(source), angle, center=(32.0, 32.0), pad_factor=1.5
    )
    np.testing.assert_allclose(
        actual, expected, atol=1e-10 if precision.itemsize == 8 else 1e-6
    )


@pytest.mark.float64_only(reason="Analytic Nyquist shear reference at atol=1e-14")
def test_nyquist_shear_uses_real_cosine_interpolant():
    """Even-grid Nyquist content stays real with the expected cosine amplitude."""
    source = jnp.tile((-1.0) ** jnp.arange(16), (16, 1))
    fx, dy, _, _ = fft_shear_setup(source, pad_factor=0.0)
    actual = fft_shear_x(source, 0.25, fx, dy, pad_factor=0.0)
    expected = (
        np.asarray(source) * np.cos(np.pi * 0.25 * (np.arange(16) - 7.5))[:, None]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
