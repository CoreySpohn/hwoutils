"""Regression tests for transform geometry and integrated-pixel semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.fft import fft_shear_setup, fft_shear_x, fft_shift_1d, get_pad_info
from hwoutils.grids import (
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)
from hwoutils.map_coordinates import map_coordinates
from hwoutils.transforms import rotate_image


@pytest.fixture(autouse=True)
def enable_x64():
    """Resolve numerical geometry errors independently of float32 roundoff."""
    with jax.enable_x64():
        yield


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


def _integral(data, step=1.0):
    return Image(
        data,
        PixelGrid.regular(data.shape[-2:], step, "pixel"),
        (),
        Measure.integral(),
        "ph",
    )


@pytest.mark.parametrize("row,col", [(0, 0), (3, 4), (4, 4), (5, 5), (11, 11)])
def test_downsampling_integrates_every_source_pixel(row, col):
    """Fails if a 3x reduction discards an impulse or multiplies its flux by 9."""
    source = _integral(jnp.zeros((12, 12)).at[row, col].set(7.0), 0.25)
    target = PixelGrid.regular((4, 4), 0.75, "pixel")
    result = integrate_footprints(source, footprint_overlap(source.grid, target))
    expected = np.zeros((4, 4))
    expected[row // 3, col // 3] = 7
    np.testing.assert_array_equal(result.data, expected)


def test_clipped_rotated_cell_and_disjoint_grid_sequence():
    """Fails if a rotated square is collected by its enclosing rectangle.

    The two unit squares differ by 45 degrees. Four isosceles triangles
    of leg 1 - 1/sqrt(2) are outside their intersection.
    """
    source = _integral(jnp.array([[1.0]]))
    c = 2**-0.5
    diamond = PixelGrid((1, 1), 0, 0, ((c, -c), (c, c)), "pixel")
    area = 1 - 2 * (1 - c) ** 2
    one = footprint_overlap(source.grid, diamond)
    np.testing.assert_allclose(
        integrate_footprints(source, one).data, [[area]], atol=2e-15, rtol=0
    )
    distant = PixelGrid.regular((1, 2), 1, "pixel", center=(10, 10))
    overlap = footprint_overlap(source.grid, (source.grid, diamond, distant))
    got = jax.jit(integrate_footprints)(source, overlap)
    np.testing.assert_allclose(got, [1, area, 0, 0], atol=2e-15, rtol=0)


def test_rebin_partial_coverage():
    """Fails if the integrator renormalizes cropped source pixels."""
    source = _integral(jnp.array([[1.0, 2.0, 3.0, 4.0]]))
    target = PixelGrid.regular((1, 2), 1.5, "pixel")
    result = integrate_footprints(source, footprint_overlap(source.grid, target))
    np.testing.assert_array_equal(result.data, [[2.5, 5.0]])


def test_rebin_noninteger_ratio_and_gradient():
    """Fails if non-integer tiling loses flux or detaches it from gradients."""
    source = _integral(jnp.arange(225.0).reshape(15, 15))
    target = PixelGrid.regular((6, 6), 2.5, "pixel")
    overlap = footprint_overlap(source.grid, target)
    result = integrate_footprints(source, overlap)
    assert result.measure == Measure.integral()
    np.testing.assert_array_equal(result.data.sum(), source.data.sum())
    grad = jax.grad(lambda im: integrate_footprints(im, overlap).data.sum())(source)
    np.testing.assert_array_equal(grad.data, 1.0)
    fine = PixelGrid.regular((30, 30), 0.5, "pixel")
    refined = integrate_footprints(source, footprint_overlap(source.grid, fine))
    expected = np.repeat(np.repeat(source.data, 2, 0), 2, 1) / 4
    np.testing.assert_array_equal(refined.data, expected)


def test_footprints_preserve_storage_precision():
    """Fails if storage dtype rounds pixel edges or high precision weights leak."""
    for dtype, width in ((jnp.float16, 4096), (jnp.bfloat16, 512), (jnp.float32, 15)):
        source = _integral(jnp.ones((1, width), dtype=dtype))
        target = PixelGrid.regular((1, 1), 1.0, "pixel")
        result = integrate_footprints(source, footprint_overlap(source.grid, target))
        np.testing.assert_array_equal(result.data, [[1.0]])
        assert result.data.dtype == dtype


@pytest.mark.parametrize("angle", [30.0, 90.0, 180.0])
@pytest.mark.parametrize("x64", [False, True])
def test_rotation_about_optical_center(angle, x64):
    """FITS optical centers need not equal the geometric array center."""
    y, x = np.mgrid[:64, :64]
    source = np.exp(-((y - 28) ** 2 + (x - 38) ** 2) / 18)
    theta = np.deg2rad(angle)
    cx = 32 + 6 * np.cos(theta) + 4 * np.sin(theta)
    cy = 32 + 6 * np.sin(theta) - 4 * np.cos(theta)
    expected = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 18)
    with jax.enable_x64(x64):
        actual = rotate_image(
            jnp.asarray(source), angle, center=(32.0, 32.0), pad_factor=1.5
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10 if x64 else 1e-6)


def test_nyquist_shear_uses_real_cosine_interpolant():
    """Even-grid Nyquist content stays real with the expected cosine amplitude."""
    source = jnp.tile((-1.0) ** jnp.arange(16), (16, 1))
    fx, dy, _, _ = fft_shear_setup(source, pad_factor=0.0)
    actual = fft_shear_x(source, 0.25, fx, dy, pad_factor=0.0)
    expected = (
        np.asarray(source) * np.cos(np.pi * 0.25 * (np.arange(16) - 7.5))[:, None]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
