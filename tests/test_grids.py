"""Coordinate, handedness, PyTree, and image resampling contracts."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.grids import Frame, GridArray, Image, PixelGrid, SkyOrientation, resample


@pytest.fixture(autouse=True, scope="module")
def precision():
    with jax.enable_x64():
        yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def frame():
    return Frame(550, 7.2, "circumscribed")


def test_worked_pixel_and_corner_count(frame):
    """Fails on a row/column swap or a half-pixel center convention error.

    Binary-exact quarter-pixel spacing needs exact equality, no tolerance.
    """
    grid = PixelGrid.focal(256, 0.25, frame)
    coords = grid.coordinates()
    assert coords.x.shape == coords.y.shape == (256, 256)
    assert (coords.x[127, 128], coords.y[127, 128]) == (0.125, -0.125)
    assert (grid.center_x, grid.center_y) == (127.5, 127.5)
    header = PixelGrid.from_corner_count((256, 256), 128, 128, grid.basis, "lod", frame)
    assert header == grid
    point = grid.unit_to_pixel(0.125, -0.125)
    assert (point.x, point.y) == (128, 127)
    pupil = PixelGrid.pupil(256)
    assert pupil.step == 1 / 256
    assert pupil.unit == "pupil_diameter"


def test_frame_conversion_and_native_geometry(frame):
    """Fails if lambda/D uses wrong wavelength units or a duplicated constant.

    Primitives: 1 nm=1e-9 m, pi rad=180 degrees, 3600 arcsec/degree.
    rtol=3e-15, atol=0 allows float64 arithmetic and rounded conversion constants.
    """
    expected = 550e-9 / 7.2 * (180 / np.pi) * 3600
    actual = jax.jit(frame.lod_to_arcsec)(jnp.array(550.0))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    grid = PixelGrid.focal(8, 0.25, frame)
    angular = grid.to_arcsec(550)
    assert angular.unit == "arcsec" and angular.frame == frame
    np.testing.assert_allclose(angular.step, expected * 0.25, rtol=3e-15, atol=0)
    assert grid.is_native(550) and not grid.is_native(600)
    assert not angular.is_native(550)
    with pytest.raises(ValueError, match="convention"):
        Frame(550, 7.2, "unknown")
    with pytest.raises(ValueError, match="positive"):
        Frame(550, 0, "inscribed")


def test_sky_orientation_parity_anchor_and_roundtrip():
    """Fails on an east sign error, reversed PA, or use of inverse as forward.

    North-up east-left anchors use exact zeros/ones. Determinants and round
    trips allow atol=2e-15 for a few float64 trigonometric operations.
    """
    orientation = SkyOrientation(0, -1)
    assert orientation.sky_to_detector(1, 0) == (-1, 0)
    assert orientation.sky_to_detector(0, 1) == (0, 1)
    for parity in (-1, 1):
        orientation = SkyOrientation(31, parity)
        east = orientation.sky_to_detector(1, 0)
        north = orientation.sky_to_detector(0, 1)
        determinant = east.x * north.y - north.x * east.y
        np.testing.assert_allclose(determinant, parity, rtol=0, atol=2e-15)
        detector = orientation.sky_to_detector(2.3, -0.7)
        sky = orientation.detector_to_sky(*detector)
        np.testing.assert_allclose(
            [sky.east, sky.north], [2.3, -0.7], rtol=0, atol=2e-15
        )
        aligned = orientation.sky_to_detector(
            np.sin(np.deg2rad(31)), np.cos(np.deg2rad(31))
        )
        np.testing.assert_allclose(aligned, [0, 1], rtol=0, atol=2e-15)


def test_equal_grids_share_trace_and_arrays_transform(frame):
    """Fails if grid identity enters the trace key or data becomes static.

    Two independently allocated equal grids must cause exactly one trace.
    All data arithmetic here is binary exact; no numeric tolerance is used.
    This test intentionally builds its own jit and vmap programs to count traces.
    """
    grid = PixelGrid.regular((4, 5), 0.25, "lod", frame=frame)
    equal = PixelGrid([4, 5], np.float64(2), 1.5, np.array(grid.basis), "lod", frame)
    assert grid == equal and hash(grid) == hash(equal)
    assert grid != replace(grid, center_x=2.5)
    with pytest.raises(FrozenInstanceError):
        grid.center_x = 0
    traces = []

    @jax.jit
    def twice(image):
        traces.append(None)
        return jax.tree.map(lambda a: 2 * a, image)

    data = jnp.arange(20.0).reshape(4, 5)
    for geometry in (grid, equal):
        image = Image(data, geometry, (), "per_pixel", "phot/s")
        result = twice(image)
        np.testing.assert_array_equal(result.data, 2 * data)
        assert result.grid == geometry
    assert len(traces) == 1
    stacked = Image(
        jnp.stack((data, 2 * data)), grid, ("wavelength",), "per_pixel", "phot/s"
    )
    mapped = jax.vmap(lambda image: jax.tree.map(lambda a: a + 1, image))(stacked)
    np.testing.assert_array_equal(mapped.data, stacked.data + 1)
    assert mapped.grid == grid and mapped.leading_axes == ("wavelength",)
    sampled = jax.vmap(lambda image: resample(image, grid))(stacked)
    np.testing.assert_array_equal(sampled.data, stacked.data)
    assert sampled.leading_axes == stacked.leading_axes
    raw = GridArray(data, grid, ())
    np.testing.assert_array_equal(jax.jit(lambda x: x)(raw).data, data)


def test_validation_rejects_inconsistent_shapes_and_geometry(frame):
    """Fails if bad geometry or metadata reaches traced kernels; no tolerance."""
    grid = PixelGrid.focal(4, 0.25, frame)
    with pytest.raises(ValueError, match="spatial shape"):
        GridArray(jnp.zeros((4, 3)), grid, ())
    with pytest.raises(ValueError, match="leading_axes count"):
        Image(jnp.zeros((2, 4, 4)), grid, (), "per_pixel", "phot/s")
    with pytest.raises(ValueError, match="measure"):
        Image(jnp.zeros((4, 4)), grid, (), "invalid", "phot/s")
    with pytest.raises(ValueError, match="requires a frame"):
        PixelGrid.regular((4, 4), 1, "lod")
    with pytest.raises(ValueError, match="nonsingular"):
        replace(grid, basis=((1, 2), (2, 4)))
    with pytest.raises(ValueError, match="step"):
        _ = replace(grid, basis=((1, 0.1), (0, 1))).step
    with pytest.raises(ValueError, match="step"):
        _ = replace(grid, basis=((1, 0), (0, 2))).step


def test_resample_identity_with_leading_axes():
    """Fails if identity shifts by half a pixel or loses image metadata.

    rtol=0, atol=8e-15 allows the float64 floor at values below 20.
    Orders 0,1,3 need distinct programs because each uses a different stencil.
    """
    grid = PixelGrid.regular((4, 5), 0.25, "arcsec")
    data = jnp.arange(40.0).reshape(2, 4, 5) / 2
    image = Image(data, grid, ("time",), "per_pixel", "phot/s")
    for order in (0, 1, 3):
        actual = jax.jit(lambda im, order=order: resample(im, grid, order=order))(image)
        np.testing.assert_allclose(actual.data, data, rtol=0, atol=8e-15)
        assert actual.grid == grid
        assert actual.measure == image.measure and actual.quantity == image.quantity
        assert actual.leading_axes == image.leading_axes


def test_resample_interior_flux_on_integer_regrid():
    """Fails if regridding drops or duplicates integrated source flux.

    A compact source must retain its sum on integer reductions/refinements.
    rtol=3e-15, atol=0 is rounding only, independent of source shape or scale ratio.
    Per-unit-area constants retain amplitude to the same float64 floor.
    """
    source = PixelGrid.regular((12, 12), 1, "arcsec")
    coarse = PixelGrid.regular((6, 6), 2, "arcsec")
    coarser = PixelGrid.regular((4, 4), 3, "arcsec")
    fine = PixelGrid.regular((24, 24), 0.5, "arcsec")
    data = jnp.zeros(source.shape).at[4:8, 4:8].set(jnp.arange(16.0).reshape(4, 4))
    image = Image(data, source, (), "per_pixel", "phot/s")
    for target in (coarse, coarser, fine):
        actual = resample(image, target)
        np.testing.assert_allclose(actual.data.sum(), data.sum(), rtol=3e-15, atol=0)
    # Refinement samples a density, so a flat patch stays flat with no gaps.
    flat = Image(jnp.ones(source.shape) * 4, source, (), "per_pixel", "phot/s")
    np.testing.assert_allclose(
        resample(flat, fine).data[4:20, 4:20], 1, rtol=3e-15, atol=0
    )
    uniform = Image(
        jnp.ones(source.shape) * 7, source, (), "per_unit_area", "Jy/arcsec2"
    )
    np.testing.assert_allclose(resample(uniform, coarse).data, 7, rtol=3e-15, atol=0)


def test_resample_half_pixel_center_shift_matches_linear_ramp():
    """Fails if center displacement has the wrong sign or uses corner counts.

    Linear interpolation reproduces an affine ramp exactly away from zero
    padding; atol=2e-14 covers float64 multiply/add in values up to 40.
    """
    grid = PixelGrid.regular((8, 9), 1, "arcsec")
    x, y = grid.coordinates()
    image = Image(10 + 2 * x - 3 * y, grid, (), "per_unit_area", "Jy/arcsec2")
    shifted = replace(grid, center_x=grid.center_x + 0.5, center_y=grid.center_y - 0.5)
    tx, ty = shifted.coordinates()
    for measure in ("per_pixel", "per_unit_area"):
        actual = resample(replace(image, measure=measure), shifted)
        np.testing.assert_allclose(
            actual.data[1:-1, 1:-1],
            (10 + 2 * tx - 3 * ty)[1:-1, 1:-1],
            rtol=0,
            atol=2e-14,
        )


def test_resample_composes_anisotropic_rotation_and_orientation():
    """Fails if basis columns, inverse mapping, or orientation order are swapped.

    An affine sky ramp is exact under linear interpolation. atol=3e-14
    permits float64 matrix inversion/trig rounding at these coordinate scales.
    The area tolerance 1e-16 permits two ULP at determinant 0.34.
    """
    source = PixelGrid.regular((12, 14), 1, "arcsec", orientation=SkyOrientation(0))
    target = PixelGrid(
        (4, 5),
        1.7,
        1.4,
        ((0.4, -0.3), (0.2, 0.7)),
        "arcsec",
        orientation=SkyOrientation(27),
    )
    coords = source.coordinates()
    sky = source.orientation.detector_to_sky(*coords)
    image = Image(
        10 + 2 * sky.east - 3 * sky.north, source, (), "per_unit_area", "Jy/arcsec2"
    )
    target_sky = target.orientation.detector_to_sky(*target.coordinates())
    expected = 10 + 2 * target_sky.east - 3 * target_sky.north
    actual = jax.jit(resample)(image, target)
    np.testing.assert_allclose(actual.data, expected, rtol=0, atol=3e-14)
    np.testing.assert_allclose(target.pixel_area, 0.34, rtol=0, atol=1e-16)
    # Refinement (area ratio 0.34): per-pixel flux is the sampled density
    # times the target pixel area, exact for the affine ramp.
    refined = resample(replace(image, measure="per_pixel"), target)
    np.testing.assert_allclose(refined.data, 0.34 * expected, rtol=0, atol=3e-14)
    # Coarsening (area ratio 3.06): deposition conserves a compact source.
    coarse = replace(
        target,
        shape=(9, 9),
        center_x=4.0,
        center_y=4.0,
        basis=((1.2, -0.9), (0.6, 2.1)),
    )
    compact = jnp.zeros(source.shape).at[5:7, 6:8].set(1.0)
    flux = Image(compact, source, (), "per_pixel", "phot/s")
    np.testing.assert_allclose(resample(flux, coarse).data.sum(), 4, rtol=0, atol=3e-14)
