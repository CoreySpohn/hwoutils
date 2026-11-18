"""Sampling, transmission, photometry and spatial PSD normalization contracts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from hwoutils import constants as const
from hwoutils import conversions as conv
from hwoutils.grids import Image, Measure, PixelGrid, resample
from hwoutils.random_fields import StationaryGaussianField, radial_frequency
from hwoutils.spectral import Band, Transmission, spectral_breaks, trapezoid_weights


@pytest.fixture(autouse=True, scope="module")
def precision():
    with jax.enable_x64():
        yield
    jax.clear_caches()


def test_density_evaluates_centers_without_area_or_box_factors():
    """Fails if point samples are deposited, box-averaged or area-scaled.

    Local linear/Keys kernels reproduce an interior affine density exactly;
    nearest chooses floor(p+1/2). DFT and reflected cosine modes have exact
    trigonometric reconstructions. Absolute tolerance 2e-13 allows float64
    kernel and matrix summation at density <10. Each kernel needs a distinct
    stencil; two Fourier extensions exercise distinct reconstruction bases.
    """
    source = PixelGrid.regular((8, 8), 0.5, "arcsec")
    x, y = source.coordinates()
    image = Image(3 + 2 * x - y, source, (), Measure.density("grid"), "ph/s")
    for step in (0.25, 0.75):
        target = PixelGrid.regular((2, 3), step, "arcsec")
        tx, ty = target.coordinates()
        px, py = source.unit_to_pixel(tx, ty)
        nearest_x, nearest_y = source.pixel_to_unit(
            np.floor(px + 0.5), np.floor(py + 0.5)
        )
        for kernel in ("nearest", "linear", "cubic"):
            expected = (
                3 + 2 * nearest_x - nearest_y
                if kernel == "nearest"
                else 3 + 2 * tx - ty
            )
            result = resample(image, target, reconstruction=kernel)
            np.testing.assert_allclose(result.data, expected, rtol=0, atol=2e-13)
            assert result.measure == Measure.density("grid")
        sy, sx = np.indices(source.shape)
        for outside in ("zero", "reflect"):
            if outside == "zero":
                data = 4 + np.cos(2 * np.pi * sx / 8) * np.cos(2 * np.pi * sy / 8)
                expected = 4 + np.cos(2 * np.pi * px / 8) * np.cos(2 * np.pi * py / 8)
            else:
                data = 4 + np.cos(np.pi * (sx + 0.5) / 8) * np.cos(
                    2 * np.pi * (sy + 0.5) / 8
                )
                expected = 4 + np.cos(np.pi * (px + 0.5) / 8) * np.cos(
                    2 * np.pi * (py + 0.5) / 8
                )
            sampled = replace(
                image, data=data, reconstruction="band_limited", outside=outside
            )
            result = resample(sampled, target, pad_factor=1)
            np.testing.assert_allclose(result.data, expected, rtol=0, atol=2e-13)


def test_transmission_support_moments_band_integrals_and_transforms():
    """Fails if interpolation clamps, knots go missing or band edges are ignored.

    Triangle (400,0),(500,1),(700,0) has area 150 nm and centroid 1600/3.
    Over [450,600], area=225/2 and first moment=58750 from integrating
    (w-400)/100 and (700-w)/200 on either side of 500. Order two is exact
    for a linear source times response. rtol=2e-14 is float64 roundoff;
    An added source kink at 550 contributes integral_0^50 t*(3/4-t/200) dt.
    Derivative finite differences allow 1e-7, including table parameters.
    No prior owner table test exists.
    """
    curve = Transmission([400, 500, 700], [0, 1, 0])
    wavelengths = jnp.array([399, 400, 450, 600, 700, 701.0])
    np.testing.assert_allclose(
        jax.jit(lambda t, w: t(w))(curve, wavelengths),
        [0, 0, 0.5, 0.5, 0, 0],
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_array_equal(spectral_breaks(curve), [400, 500, 700])
    np.testing.assert_allclose(curve.mean_nm, 1600 / 3, rtol=2e-14)
    np.testing.assert_allclose(
        curve.integrate(Band(300, 800), order=1), 150, rtol=2e-14
    )
    band = Band(450, 600)
    np.testing.assert_allclose(curve.integrate(band, order=1), 225 / 2, rtol=2e-14)
    np.testing.assert_allclose(
        curve.integrate(band, lambda w: w, order=2),
        58750,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        curve.integrate(
            band,
            lambda w: jnp.array([1.0, jnp.maximum(w - 550, 0)]),
            order=2,
            breaks_nm=jnp.array([550.0]),
        ),
        [225 / 2, (3 / 4) * 50**2 / 2 - 50**3 / 600],
        rtol=2e-14,
    )
    hat = Transmission.from_band(Band(500, 600))
    np.testing.assert_array_equal(hat(jnp.array([499, 500, 600, 601])), [0, 1, 1, 0])
    assert hat.low_nm == 500 and hat.high_nm == 600
    np.testing.assert_allclose(hat.mean_nm, 550, rtol=2e-14)
    np.testing.assert_allclose(hat.integrate(Band(510, 570)), 60, rtol=2e-14)
    assert hat.integrate(Band(610, 700)) == 0
    opaque = Transmission([400, 700], [0, 0])
    assert opaque.integrate(Band(400, 700)) == 0
    assert np.isnan(opaque.mean_nm)
    for wave, response in (
        ([500], [1]),
        ([500, 400], [1, 1]),
        ([400, 400], [1, 1]),
        ([0, 500], [1, 1]),
        ([400, np.nan], [1, 1]),
        ([400, 500], [-1, 1]),
        ([400, 500], [1, np.inf]),
        ([400, 500], [1]),
    ):
        with pytest.raises(ValueError):
            Transmission(wave, response)
    check_grads(
        lambda t, w: t(w),
        (curve, jnp.array(550.0)),
        order=1,
        modes=("fwd", "rev"),
        atol=1e-7,
        rtol=1e-7,
    )


def test_nonuniform_trapezoid_weights_batched_and_degenerate():
    """Fails if nonuniform PSD line widths or endpoint half-cells are wrong.

    The integral of 2*x+3 is x**2+3*x on each increasing grid. This exact
    algebra allows 2e-15 relative float64 summation error. Singleton span is
    exactly zero; validation assertions require exact failures.
    """
    x = jnp.array([[0, 0.25, 1, 3], [1, 2, 4, 8.0]])
    weights = jax.jit(trapezoid_weights)(x)
    expected = np.asarray(x[:, -1] ** 2 + 3 * x[:, -1] - x[:, 0] ** 2 - 3 * x[:, 0])
    np.testing.assert_allclose(jnp.sum(weights * (2 * x + 3), -1), expected, rtol=2e-15)
    np.testing.assert_array_equal(weights[0], [0.125, 0.5, 1.375, 1.0])
    np.testing.assert_array_equal(trapezoid_weights([2.0]), [0.0])
    for bad in ([], [1, 1], [2, 1], [0, np.inf]):
        with pytest.raises(ValueError):
            trapezoid_weights(bad)


def test_photometric_ratios_diameters_and_literature_zero_point():
    """Fails for a magnitude sign, radius/diameter or mas/arcsec error.

    Five magnitudes is a factor 100 by definition; 2 AU at 1 pc is 2 arcsec.
    rtol=3e-15 covers float64 constants and logarithms. Table A2 of Bessell,
    Castelli & Plez (1998) gives V f_nu=3.636e-20 erg/s/cm^2/Hz: the
    literature constant comparison allows only its unit-conversion roundoff.
    """
    contrasts = jnp.array([1.0, 0.01, 1e-10, 100.0])
    expected = np.array([0, 5, 25, -5.0])
    np.testing.assert_allclose(
        jax.jit(conv.contrast_to_delta_mag)(contrasts), expected, rtol=3e-15, atol=0
    )
    np.testing.assert_allclose(
        conv.delta_mag_to_contrast(expected), contrasts, rtol=3e-15
    )
    assert np.isposinf(conv.contrast_to_delta_mag(0.0))
    assert np.isnan(conv.contrast_to_delta_mag(-1.0))
    radius = jnp.array([0, const.AU2m, 2 * const.AU2m])
    distance = jnp.array([[1], [10.0]])
    expected = np.array([[0, 2, 4], [0, 0.2, 0.4]])
    np.testing.assert_allclose(
        conv.angular_diameter_arcsec(radius, distance), expected, rtol=3e-15, atol=0
    )
    np.testing.assert_allclose(
        jax.jit(conv.angular_diameter_mas)(radius, distance),
        expected * const.arcsec2mas,
        rtol=3e-15,
        atol=0,
    )
    from_cgs_jy = 3.636e-20 / (const.J2erg * const.cm22m2) / const.Jy
    np.testing.assert_allclose(
        const.JOHNSON_V_VEGA_ZERO_POINT_JY, from_cgs_jy, rtol=3e-15
    )


def test_radial_psd_variance_matches_closed_form_integral():
    """Fails for FFT size, spatial spacing, radial Jacobian or half-plane errors.

    S(k)=3*exp(-k**2/(2*sigma**2)), sigma=0.2, integrates to 6*pi*sigma**2.
    With df=1/16, Nyquist >=1, Gaussian tails omit <6e-7 and sampling error
    is negligible, so the deterministic integral allows 2e-6 relative error.
    For R=256 draws, N_eff=(int S)**2/(df**2*int S**2)=4*pi*sigma**2/df**2.
    Mean-square relative SE=sqrt(2/(R*N_eff)); tolerance is four SE plus
    2e-6 discretization error. No sample mean or per-draw RMS is removed.
    """
    sigma, draws, df = 0.2, 256, 1 / 16
    field = StationaryGaussianField.from_radial_psd(
        lambda k: 3 * np.exp(-(k**2) / (2 * sigma**2)),
        np.ones((32, 40), bool),
        spacing=(0.5, 0.4),
    )
    expected = 6 * np.pi * sigma**2
    np.testing.assert_allclose(field.pixel_variance(), expected, rtol=2e-6)
    samples = jax.jit(lambda f, key: f.sample(key, draws))(field, jax.random.key(0))
    n_eff = 4 * np.pi * sigma**2 / df**2
    tolerance = 4 * np.sqrt(2 / (draws * n_eff)) + 2e-6
    measured = np.mean(np.asarray(samples) ** 2)
    np.testing.assert_allclose(measured, expected, rtol=tolerance)
    print(
        f"PSD mean square: {measured:.9g}; expected {expected:.9g}; "
        f"rtol {tolerance:.6g}"
    )


def test_field_coloring_inverse_support_and_psd_validation():
    """Fails if real-field symmetry, support, inverse or white prior breaks.

    Parseval gives log_prob(color(w))=-sum(w**2)/2. With positive moderate
    gains, 2e-13 absolute error covers float64 FFTs and inverses; gradient
    finite differences allow 1e-7. A single tiny rectangular structure also
    checks the cycles-per-unit grid against its explicit DFT indices.
    """
    support = np.ones((4, 6), bool)
    frequency = radial_frequency((4, 6), (0.5, 0.25))
    fy = np.array([0, 1, -2, -1]) / (4 * 0.5)
    fx = np.array([0, 1, 2, -3, -2, -1]) / (6 * 0.25)
    np.testing.assert_allclose(frequency, np.sqrt(fy[:, None] ** 2 + fx**2), rtol=2e-15)
    field = StationaryGaussianField.from_radial_psd(
        lambda k: 1 / (1 + k**2),
        support,
        spacing=(0.5, 0.25),
    )
    white = jax.random.normal(jax.random.key(0), field.shape)
    colored = jax.jit(lambda f, w: f.color(w))(field, white)
    np.testing.assert_allclose(field.whiten(colored), white, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        field.log_prob(colored), -np.sum(np.asarray(white) ** 2) / 2, rtol=0, atol=2e-13
    )
    support[0] = False
    masked = replace(field, support=support)
    np.testing.assert_array_equal(masked.color(white)[0], 0)
    np.testing.assert_allclose(masked.color(white)[1:], colored[1:], rtol=0, atol=0)
    assert field.sample(jax.random.key(0)).shape == field.shape
    check_grads(
        field.color, (white,), order=1, modes=("fwd", "rev"), atol=1e-7, rtol=1e-7
    )
    for bad in (-1.0, np.inf):
        with pytest.raises(ValueError, match="psd"):
            StationaryGaussianField.from_radial_psd(lambda k, bad=bad: bad, support)
    with pytest.raises(ValueError, match="spacing"):
        radial_frequency(4, 0)
    with pytest.raises(ValueError, match="symmetric"):
        StationaryGaussianField(np.arange(24.0).reshape(4, 6), support)
