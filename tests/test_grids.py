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
    assert frame.coordinate_model == "paraxial_entrance_tilt"
    assert frame == Frame(550, 7.2, "circumscribed", "paraxial_entrance_tilt")
    with pytest.raises(ValueError, match="coordinate_model"):
        replace(frame, coordinate_model="tangent")


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
    The three local reconstructions each need a distinct stencil program.
    """
    grid = PixelGrid.regular((4, 5), 0.25, "arcsec")
    data = jnp.arange(40.0).reshape(2, 4, 5) / 2
    image = Image(data, grid, ("time",), "per_pixel", "phot/s")
    for reconstruction in ("nearest", "linear", "cubic"):
        actual = jax.jit(
            lambda im, mode=reconstruction: resample(im, grid, reconstruction=mode)
        )(image)
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


def test_band_limited_periodic_exactness():
    """Fails on phase, axis, pixel-area, or sinc normalization errors.

    Basis: exact sinusoidal point values and analytic rectangular integrals;
    200 epsilon allows DFT phase/reduction rounding in float32 and float64.
    Rectangular odd/even axes, anisotropic spacing, and parity reversal cover
    distinct geometry failures using the same small transform structure.
    """
    source = PixelGrid(
        (7, 8),
        3.2,
        2.6,
        ((0.4, 0), (0, 0.6)),
        "arcsec",
        orientation=SkyOrientation(13, -1),
    )
    target = PixelGrid(
        (11, 12),
        5.7,
        4.5,
        ((0.23, 0), (0, 0.37)),
        "arcsec",
        orientation=SkyOrientation(13, 1),
    )

    def field(grid, width_x=0, width_y=0):
        # Coordinates in the SOURCE detector frame: target parity flips x.
        x, y = map(np.asarray, grid.coordinates())
        if grid.orientation.parity == 1:
            x = -x
        fx, fy = 2 / (8 * 0.4), 1 / (7 * 0.6)
        return (
            2
            + 0.3
            * np.sin(2 * np.pi * (fx * x + fy * y) + 0.2)
            * np.sinc(fx * width_x)
            * np.sinc(fy * width_y)
            + 0.4 * np.cos(2 * np.pi * fy * y - 0.7) * np.sinc(fy * width_y)
        )

    for dtype in (jnp.float32, jnp.float64):
        for measure in ("per_unit_area", "per_pixel"):
            source_area = 0.4 * 0.6 if measure == "per_pixel" else 1
            image = Image(
                jnp.asarray(field(source) * source_area, dtype=dtype),
                source,
                (),
                measure,
                "flux",
                "band_limited",
            )
            # Keep complete boxes inside the footprint for this periodic oracle.
            shifted = replace(source, shape=(5, 6), center_x=2.57, center_y=1.31)
            for grid in (target, shifted):
                expected = field(grid)
                if measure == "per_pixel":
                    dx, dy = grid.basis[0][0], grid.basis[1][1]
                    expected = field(grid, dx, dy) * dx * dy
                actual = resample(image, grid, pad_factor=1)
                assert actual.data.dtype == dtype
                np.testing.assert_allclose(
                    actual.data, expected, rtol=200 * np.finfo(dtype).eps, atol=0
                )


def test_band_limited_metadata_and_overrides():
    """Fails if reconstruction metadata disappears or overrides are ignored.

    Basis: exact metadata and local-kernel parity; float64 DFT identity has
    atol=3e-14. Rotation/shear must fail only for the Fourier path, including
    rotation introduced by orientation with diagonal grid bases.
    """
    source = PixelGrid.regular((4, 5), 1, "arcsec")
    image = Image(
        jnp.arange(20.0).reshape(4, 5),
        source,
        (),
        "per_unit_area",
        "flux",
        "band_limited",
    )
    for transformed in (
        jax.jit(lambda im: im)(image),
        jax.tree.map(lambda d: d * 2, image),
        replace(image, quantity="density"),
    ):
        assert transformed.reconstruction == "band_limited"
    with pytest.raises(ValueError, match="reconstruction"):
        replace(image, reconstruction="unknown")
    with pytest.raises(ValueError, match="reconstruction"):
        resample(image, source, reconstruction="unknown")
    for factor in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="pad_factor"):
            resample(image, source, pad_factor=factor)
    target = replace(source, center_x=2.3)
    for mode in ("nearest", "linear", "cubic"):
        actual = resample(image, target, reconstruction=mode)
        expected = resample(replace(image, reconstruction=mode), target)
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.reconstruction == "band_limited"
    overridden = resample(
        replace(image, reconstruction="linear"), target, reconstruction="band_limited"
    )
    np.testing.assert_array_equal(overridden.data, resample(image, target).data)
    assert overridden.reconstruction == "linear"
    np.testing.assert_allclose(
        resample(
            replace(image, reconstruction="linear"),
            source,
            reconstruction="band_limited",
        ).data,
        image.data,
        rtol=0,
        atol=3e-14,
    )
    assert resample(image, source).reconstruction == "band_limited"
    # Area decreases, but x coarsens: an area-only rule would be wrong.
    coarse = replace(source, basis=((1.2, 0), (0, 0.5)))
    assert resample(image, coarse).reconstruction == "linear"
    for basis in (((0.8, -0.6), (0.6, 0.8)), ((1, 0.2), (0, 1))):
        rotated = replace(target, basis=basis)
        with pytest.raises(NotImplementedError, match="rotated or sheared"):
            resample(image, rotated)
        for mode in ("nearest", "linear", "cubic"):
            resample(image, rotated, reconstruction=mode)
    oriented = replace(image, grid=replace(source, orientation=SkyOrientation(0)))
    with pytest.raises(NotImplementedError, match="composed"):
        resample(oriented, replace(source, orientation=SkyOrientation(27)))


def test_band_limited_period_flux_and_nyquist():
    """Fails if box integration loses DC flux or mixes real Nyquist axes.

    Basis: with pad_factor=1 the footprint is exactly one Fourier period, whose
    integral is the sample sum to 100 eps. The target contains that footprint.
    Even coarsening conserves this one integral, although
    modes above the new Nyquist alias and must lose the band_limited label.
    Nyquist cosines on an even grid must remain separable, not cos(x+y).
    """
    source = PixelGrid.regular((6, 8), 0.5, "arcsec")
    rng = np.random.default_rng(0)
    for dtype in (jnp.float32, jnp.float64):
        image = Image(
            jnp.asarray(rng.uniform(size=(2, 6, 8)), dtype=dtype),
            source,
            ("time",),
            "per_pixel",
            "flux",
            "band_limited",
        )
        for shape, step in (((18, 24), 1 / 3), ((6, 8), 1)):
            target = PixelGrid.regular(shape, step, "arcsec")
            actual = resample(image, target, pad_factor=1)
            np.testing.assert_allclose(
                actual.data.sum(axis=(-2, -1)),
                image.data.sum(axis=(-2, -1)),
                rtol=100 * np.finfo(dtype).eps,
                atol=0,
            )
            assert actual.reconstruction == ("band_limited" if step < 0.5 else "linear")
    grid = PixelGrid.regular((4, 4), 1, "arcsec", center=(0, 0))
    rows, cols = np.indices((4, 4))
    checker = Image(
        (-1.0) ** (rows + cols), grid, (), "per_unit_area", "flux", "band_limited"
    )
    shifted = replace(grid, center_x=-0.2, center_y=-0.3)
    expected = np.cos(np.pi * (cols + 0.2)) * np.cos(np.pi * (rows + 0.3))
    np.testing.assert_allclose(
        resample(checker, shifted, pad_factor=1).data, expected, rtol=0, atol=4e-15
    )


def test_band_limited_jit_and_transpose_gradient():
    """Fails if tracing changes values or the data derivative is not linear.

    Basis: transpose of independently evaluated impulse responses, 200 eps;
    finite differences additionally check both AD modes at 1e-8 in float64.
    Batch axes, real/complex data, both extensions, and both measures share one
    small geometry. Reflection adds real and complex float64 transform programs.
    """
    from jax.test_util import check_grads

    source = PixelGrid.regular((3, 4), 0.7, "arcsec")
    target = PixelGrid.regular((5, 6), 0.39, "arcsec", center=(2.2, 1.8))
    rng = np.random.default_rng(0)
    for dtype, outside in (
        (jnp.float32, "zero"),
        (jnp.float64, "zero"),
        (jnp.complex128, "zero"),
        (jnp.float64, "reflect"),
        (jnp.complex128, "reflect"),
    ):
        for measure in ("per_unit_area", "per_pixel"):
            data = jnp.asarray(rng.normal(size=(2, 3, 4)), dtype=dtype)
            if jnp.iscomplexobj(data):
                data = data + 0.2j * data[::-1]
            image = Image(
                data, source, ("time",), measure, "flux", "band_limited", outside
            )

            def apply(values, image=image):
                return resample(replace(image, data=values), target).data

            eager = apply(data)
            tolerance = 200 * np.finfo(data.real.dtype).eps
            np.testing.assert_allclose(
                jax.jit(apply)(data), eager, rtol=tolerance, atol=tolerance
            )
            weights = jnp.asarray(rng.normal(size=eager.shape), dtype=dtype)
            gradient = jax.grad(
                lambda d, weights=weights: jnp.real(jnp.sum(apply(d) * weights))
            )(data)
            impulses = jnp.eye(data.size, dtype=dtype).reshape((-1, *data.shape))
            responses = jax.vmap(apply)(impulses)
            expected = jnp.einsum("ibyx,byx->i", responses, weights).reshape(data.shape)
            np.testing.assert_allclose(
                gradient, expected, rtol=tolerance, atol=tolerance
            )
            if dtype == jnp.float64 and measure == "per_pixel":
                check_grads(
                    apply,
                    (data,),
                    order=1,
                    modes=("fwd", "rev"),
                    eps=1e-4,
                    atol=1e-8,
                    rtol=1e-8,
                )


def _circle_square_areas(grid, radius):
    """Exact circle-square intersections from a signed quadrant primitive."""
    x, y = map(np.asarray, grid.coordinates())
    half = grid.step / 2

    def primitive(x, y):
        sign = np.sign(x) * np.sign(y)
        x, y = np.minimum(abs(x), radius), np.minimum(abs(y), radius)
        split = np.minimum(x, np.sqrt(np.maximum(radius**2 - y**2, 0)))

        def segment(t):
            return (
                t * np.sqrt(np.maximum(radius**2 - t**2, 0))
                + radius**2 * np.arcsin(t / radius)
            ) / 2

        return sign * (y * split + segment(x) - segment(split))

    return (
        primitive(x + half, y + half)
        - primitive(x - half, y + half)
        - primitive(x + half, y - half)
        + primitive(x - half, y - half)
    )


def test_band_limited_airy_aperture(frame):
    """Fails if Nyquist PSF photometry uses a local interpolation kernel.

    Basis: analytic Airy encircled energy (code verification), plus a bound
    derived from discarded wings and periodic replicas, not observed errors.
    Set u=sqrt(x)*J1(x), q=1-3/(4*x*x). The Bessel ODE gives
    (u'^2+q*u^2)'=q'*u^2>=0 and limiting energy 2/pi, hence
    I(r)<=C/r^3 for r>=1, C=4/pi^3 (a conservative envelope).
    On a padded square of side L, replica sup norm <=88*C/L^3: shell m
    has 8m copies at distance >=L*(m-1/2), and sum m/(m-1/2)^3<=11.
    Parseval and Cauchy-Schwarz bound the aperture error by
    sqrt(A)*h*||discarded samples||_2 + (sqrt(A)*L+A)*replica_bound.

    Each detector covers one full padded period. Its declared reconstruction
    is evaluated on a fine 0.005 lambda/D aperture grid (no further padding),
    with exact circle-square areas. Thus coarse aperture-edge quadrature is
    excluded. The extra midpoint bound is O(h_fine^2): interior Hessian and
    boundary-strip gradient terms, using the reconstructed Fourier coefficient
    absolute sum to bound derivatives (each axis has bandwidth <=1).
    This tests density photometry; exact per-pixel boxes are tested separately.
    """
    from scipy.special import j0, j1

    def airy(x, y):
        z = np.pi * np.hypot(x, y)
        amplitude = np.where(z == 0, 1.0, 2 * j1(z) / np.where(z == 0, 1, z))
        return np.pi / 4 * amplitude**2

    radius = 0.7
    source = PixelGrid.focal(32, 0.5, frame)
    image = Image(
        airy(*map(np.asarray, source.coordinates())),
        source,
        (),
        "per_unit_area",
        "fraction/lod2",
        "band_limited",
    )
    expected = 1 - j0(np.pi * radius) ** 2 - j1(np.pi * radius) ** 2
    aperture = PixelGrid.focal(282, 0.005, frame)
    weights = _circle_square_areas(aperture, radius)
    area = np.pi * radius**2
    np.testing.assert_allclose(weights.sum(), area, rtol=0, atol=3e-14)
    padded = PixelGrid.focal(64, 0.5, frame)
    x, y = map(np.asarray, padded.coordinates())
    wings = np.where((abs(x) >= 8) | (abs(y) >= 8), airy(x, y), 0)
    period = 32
    replicas = 88 * (4 / np.pi**3) / period**3
    truncation_bound = (
        np.sqrt(area) * 0.5 * np.linalg.norm(wings)
        + (np.sqrt(area) * period + area) * replicas
    )
    coefficient_sum = np.abs(np.fft.fft2(np.pad(np.asarray(image.data), 16))).sum()
    coefficient_sum /= 64**2
    quadrature_bound = (
        aperture.step**2
        * coefficient_sum
        * (area * (2 * np.pi) ** 2 / 6 + 8 * np.pi * radius * (2 * np.pi))
    )
    fluxes = []
    for pitch in (0.2, 0.1, 0.05):
        detector = PixelGrid.focal(round(period / pitch), pitch, frame)
        row = []
        for mode in ("band_limited", "linear"):
            sampled = resample(replace(image, reconstruction=mode), detector)
            fine = resample(sampled, aperture, pad_factor=1)
            row.append(float(np.sum(np.asarray(fine.data) * weights)))
        band_error, linear_error = (abs(value - expected) for value in row)
        assert band_error <= truncation_bound + quadrature_bound
        assert band_error * 20 < linear_error
        fluxes.append(row)
    assert np.ptp(np.asarray(fluxes)[:, 0]) <= truncation_bound
    # Report the measured comparison without freezing it as a golden value.
    print(
        f"Airy analytic={expected:.12f}; bound={truncation_bound:.6g}"
        f"+{quadrature_bound:.6g}; pitches/fluxes={fluxes}"
    )


def test_reflect_band_limited_constant_and_cosine():
    """Fails if reflection pads with zeros or uses whole-sample symmetry.

    Basis: constant density and half-sample cosine modes are exact solutions,
    including analytic box integrals. 200 eps covers phase/reduction rounding
    in both precisions. The zero-extension negative control must ring >1e-5
    within three source pixels of an edge, inside the source footprint.
    Odd/even rectangular axes and non-unit areas expose axis and scale errors.
    """
    source = PixelGrid.regular((7, 8), 0.5, "arcsec", center=(0, 0))
    target = PixelGrid.regular((13, 15), 0.23, "arcsec", center=(2.3, -2.2))
    sy, sx = np.indices(source.shape)
    tx, ty = map(np.asarray, target.coordinates())
    tx, ty = tx / 0.5, ty / 0.5
    # Pixel [6, 3] is 0.822 source pixels from the left half-sample edge.
    assert 0 < tx[6, 3] + 0.5 < 3
    assert -0.5 < ty[6, 3] < 6.5
    for dtype in (jnp.float32, jnp.float64):
        for measure in ("per_unit_area", "per_pixel"):
            source_area = 0.5**2 if measure == "per_pixel" else 1
            target_area = 0.23**2 if measure == "per_pixel" else 1
            image = Image(
                jnp.full(source.shape, 3 * source_area, dtype=dtype),
                source,
                (),
                measure,
                "flux",
                "band_limited",
                "reflect",
            )
            actual = resample(image, target)
            assert actual.data.dtype == dtype
            tolerance = 200 * np.finfo(dtype).eps
            np.testing.assert_allclose(
                actual.data, 3 * target_area, rtol=tolerance, atol=0
            )
            ringing = resample(replace(image, outside="zero"), target).data[6, 3]
            assert abs(float(ringing) / (3 * target_area) - 1) > 1e-5
            # Exact nonconstant even interpolant; kx=2, ky=1, below Nyquist.
            values = 3 + np.cos(np.pi * 2 * (sx + 0.5) / 8) * np.cos(
                np.pi * (sy + 0.5) / 7
            )
            box = (
                np.sinc(2 * (0.23 / 0.5) / 16) * np.sinc((0.23 / 0.5) / 14)
                if measure == "per_pixel"
                else 1
            )
            expected = target_area * (
                3
                + box
                * np.cos(np.pi * 2 * (tx + 0.5) / 8)
                * np.cos(np.pi * (ty + 0.5) / 7)
            )
            actual = resample(
                replace(image, data=jnp.asarray(values * source_area, dtype=dtype)),
                target,
            )
            np.testing.assert_allclose(actual.data, expected, rtol=tolerance, atol=0)
            distant = replace(
                target,
                center_x=target.center_x + 16 * 0.5 / 0.23,
                center_y=target.center_y - 14 * 0.5 / 0.23,
            )
            repeated = resample(
                replace(image, data=jnp.asarray(values * source_area, dtype=dtype)),
                distant,
            )
            np.testing.assert_allclose(repeated.data, expected, rtol=tolerance, atol=0)
            wide = PixelGrid.regular((21, 24), 0.5, "arcsec")
            # Three times the source width in both axes, including both edges.
            np.testing.assert_allclose(
                resample(image, wide).data, 3 * source_area, rtol=tolerance, atol=0
            )


def test_reflect_smooth_interior_error():
    """Fails if a nonzero edge still imposes the zero-padding discontinuity.

    Basis: analytic polynomial times Gaussian; all targets are >4 source
    pixels inside the edges. The requested 10x RMS improvement is a negative
    control threshold (spike: ~370x), not a universal convergence-rate claim.
    No periodicity or vanishing edge is imposed on the analytic field.
    """
    source = PixelGrid.regular((32, 30), 0.125, "arcsec")
    target = PixelGrid.regular((37, 35), 0.071, "arcsec", center=(17.13, 18.27))

    def field(grid):
        x, y = map(np.asarray, grid.coordinates())
        return (1 + 0.1 * x + 0.03 * y + 0.02 * x * y) * np.exp(-0.08 * (x * x + y * y))

    image = Image(field(source), source, (), "per_unit_area", "flux", "band_limited")
    expected = field(target)
    errors = [
        np.sqrt(
            np.mean(
                (
                    np.asarray(resample(replace(image, outside=mode), target).data)
                    - expected
                )
                ** 2
            )
        )
        for mode in ("zero", "reflect")
    ]
    assert errors[1] * 10 < errors[0]
    print(f"Smooth interior RMS zero={errors[0]:.9g}, reflect={errors[1]:.9g}")


def test_outside_metadata_and_padding():
    """Fails if static outside is lost in transforms or changes the wrong path.

    Basis: exact metadata and unchanged arrays when an ignored pad_factor
    changes. A jitted resample includes both PyTree input and output rebuilds;
    coarsening must keep outside even when reconstruction becomes linear.
    """
    grid = PixelGrid.regular((4, 5), 0.5, "arcsec")
    image = Image(jnp.arange(20.0).reshape(4, 5), grid, (), "per_unit_area", "flux")
    assert image.outside == "zero"
    with pytest.raises(ValueError, match="outside"):
        replace(image, outside="unknown")
    image = replace(image, outside="reflect", reconstruction="band_limited")
    coarse = replace(grid, basis=((1, 0), (0, 1)))
    eager = resample(image, coarse)
    compiled = jax.jit(resample)(image, coarse)
    np.testing.assert_allclose(compiled.data, eager.data, rtol=0, atol=1e-14)
    for result in (
        jax.jit(lambda im: im)(image),
        jax.tree.map(lambda d: d * 2, image),
        replace(image, quantity="density"),
        eager,
        compiled,
        resample(image, grid, reconstruction="linear"),
    ):
        assert result.outside == "reflect"
    assert eager.reconstruction == "linear"
    for factor in (1, 3, float("nan")):
        np.testing.assert_array_equal(
            resample(image, coarse, pad_factor=factor).data, eager.data
        )


def test_reflect_local_kernels_outside_footprint():
    """Fails if local kernels use zero/mirror boundaries or deposit finite flux.

    Basis: hand-evaluated nearest, linear and Keys stencils at quarter offsets
    on both sides of [1,3,7,13], half-sample reflection; binary-exact arithmetic.
    Equal and coarser pixels must sample the exterior for both measures.
    Refinement of a constant additionally checks area scaling at the edges.
    Zero-extension stencils have finite support and cannot produce distant
    copies. Linear/cubic can spill just past a half-sample edge; at x=-0.75
    a unit constant has values 1/4 and 13/64, respectively, not zero.
    """
    source = PixelGrid.regular((4, 4), 0.5, "arcsec", center=(0, 0))
    values = np.array([1, 3, 7, 13])
    samples = values[None, :] + 2 * values[:, None]
    expected_rows = {
        "nearest": np.array([1, 1, 3, 7, 13, 13]),
        "linear": np.array([1.5, 1, 2.5, 6, 11.5, 13]),
        "cubic": np.array([1.3125, 0.8125, 2.3125, 5.8125, 11.875, 13.5625]),
    }
    for mode, row in expected_rows.items():
        for measure in ("per_unit_area", "per_pixel"):
            image = Image(samples, source, (), measure, "flux", mode, "reflect")
            for ratio in (1, 2):
                target = PixelGrid.regular(
                    (6 // ratio, 6 // ratio),
                    0.5 * ratio,
                    "arcsec",
                    center=(1.25 / ratio, 1.25 / ratio),
                )
                expected = row[None, ::ratio] + 2 * row[::ratio, None]
                if measure == "per_pixel":
                    expected *= ratio**2
                np.testing.assert_array_equal(resample(image, target).data, expected)
                distant = replace(
                    target,
                    center_x=target.center_x + 8 / ratio,
                    center_y=target.center_y - 16 / ratio,
                )
                np.testing.assert_array_equal(resample(image, distant).data, expected)
            fine = PixelGrid.regular((10, 10), 0.25, "arcsec")
            flat = replace(image, data=jnp.ones(source.shape))
            np.testing.assert_array_equal(
                resample(flat, fine).data, 0.25 if measure == "per_pixel" else 1
            )
        zero = Image(jnp.ones(source.shape), source, (), "per_unit_area", "flux", mode)
        exterior = PixelGrid.regular((12, 12), 0.5, "arcsec", center=(4, 4))
        actual = np.asarray(resample(zero, exterior).data)
        expected = np.zeros((12, 12))
        expected[4:8, 4:8] = 1
        np.testing.assert_array_equal(actual, expected)
        edge = PixelGrid.regular((1, 1), 0.5, "arcsec", center=(0.75, -1))
        np.testing.assert_array_equal(
            resample(zero, edge).data,
            {"nearest": 0, "linear": 1 / 4, "cubic": 13 / 64}[mode],
        )


def test_band_limited_zero_wide_gaussian(frame):
    """Fails if exterior target positions see periodic copies of a PSF.

    Basis: sigma=0.8 lambda/D at (1,0), sampled at 0.5 on 24x24, has its
    nearest edge 6.25 sigma away. The omitted analytic mass is <2.1e-10;
    Fourier truncation adds <1e-7 relative (float64 spike). The requested
    16 float32 eps allowance covers reconstruction and rounding. Both target
    pitches partition the entire source footprint, so their exact box sums
    must agree. Point samples outside and boxes wholly outside are exactly
    zero; outside-centered straddlers must still integrate their overlap.
    Two target shapes exercise refinement and coarsening; 64x64 at 0.85
    spans over two padded periods and used to integrate about nine copies.
    """
    source = PixelGrid.focal(24, 0.5, frame)
    x, y = map(np.asarray, source.coordinates())
    density = np.exp(-((x - 1) ** 2 + y**2) / (2 * 0.8**2))
    for dtype in (jnp.float32, jnp.float64):
        image = Image(
            jnp.asarray(density * 0.5**2, dtype=dtype),
            source,
            (),
            "per_pixel",
            "fraction",
            "band_limited",
        )
        for n, pitch in ((64, 0.25), (40, 0.85), (64, 0.85)):
            target = PixelGrid.focal(n, pitch, frame)
            result = np.asarray(resample(image, target).data)
            source_sum = np.asarray(image.data).sum(dtype=np.float64)
            np.testing.assert_allclose(
                result.sum(dtype=np.float64),
                source_sum,
                rtol=16 * np.finfo(np.float32).eps,
                atol=0,
            )
            tx, ty = map(np.asarray, target.coordinates())
            outside_centers = (abs(tx) > 6) | (abs(ty) > 6)
            outside_boxes = (abs(tx) - pitch / 2 >= 6) | (abs(ty) - pitch / 2 >= 6)
            assert outside_boxes.any()
            np.testing.assert_array_equal(result[outside_boxes], 0)
            points = resample(
                replace(image, data=image.data / 0.5**2, measure="per_unit_area"),
                target,
            )
            np.testing.assert_array_equal(np.asarray(points.data)[outside_centers], 0)
            print(
                f"Gaussian {dtype} {n}x{n} pitch={pitch}: relative flux error="
                f"{result.sum(dtype=np.float64) / source_sum - 1:.9g}"
            )


def test_band_limited_zero_clipped_boxes():
    """Fails if a straddling box is masked by its center or uses its full width.

    Basis: pad_factor=1 makes a constant density exactly constant inside;
    its flux is density times the rectangular overlap area, in closed form.
    A separable sine additionally checks the phase of a clipped non-DC mode
    using its cosine antiderivative. 200 eps covers small DFT matrix rounding.
    Non-unit, unequal axis widths, both edges, corners, outside centers,
    a box containing the whole source, and complex values share two shapes.
    """
    source = PixelGrid.regular((4, 4), 0.5, "arcsec")
    x, y = map(np.asarray, source.coordinates())
    for dtype in (jnp.float32, jnp.complex128):
        amplitude = 3 + 2j if dtype == jnp.complex128 else 3
        density = jnp.full(source.shape, amplitude, dtype=dtype)
        image = Image(density * 0.5**2, source, (), "per_pixel", "flux", "band_limited")
        # Static geometry must not round exterior points onto an edge in float32.
        exterior = PixelGrid.regular((2, 2), 2 + 2e-9, "arcsec")
        np.testing.assert_array_equal(
            resample(
                replace(image, data=density, measure="per_unit_area"), exterior
            ).data,
            0,
        )
        for basis in (((0.75, 0), (0, -1.25)), ((5, 0), (0, 6))):
            target = PixelGrid((5, 5), 2.25, 1.75, basis, "arcsec")
            tx, ty = map(np.asarray, target.coordinates())
            wx, wy = abs(basis[0][0]), abs(basis[1][1])
            ax, bx = np.clip(tx - wx / 2, -1, 1), np.clip(tx + wx / 2, -1, 1)
            ay, by = np.clip(ty - wy / 2, -1, 1), np.clip(ty + wy / 2, -1, 1)
            expected = amplitude * (bx - ax) * (by - ay)
            tolerance = 200 * np.finfo(density.real.dtype).eps
            actual = np.asarray(resample(image, target, pad_factor=1).data)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            empty = (bx == ax) | (by == ay)
            np.testing.assert_array_equal(actual[empty], 0)
            if wx < 1:
                straddlers = (abs(tx) > 1) & (bx > ax) & (by > ay)
                assert straddlers.any()
                assert np.all(abs(actual[straddlers]) > 0)
            field = amplitude * (1 + 0.2 * np.sin(np.pi * x) * np.sin(np.pi * y))
            sine_x = (np.cos(np.pi * ax) - np.cos(np.pi * bx)) / np.pi
            sine_y = (np.cos(np.pi * ay) - np.cos(np.pi * by)) / np.pi
            expected += 0.2 * amplitude * sine_x * sine_y
            sampled = resample(
                replace(image, data=jnp.asarray(field * 0.5**2, dtype=dtype)),
                target,
                pad_factor=1,
            )
            np.testing.assert_allclose(
                sampled.data, expected, rtol=tolerance, atol=tolerance
            )
