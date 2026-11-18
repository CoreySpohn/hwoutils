"""Owner APIs checked against independent algebra and radiometric limits."""

import math
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from numpy.polynomial.legendre import leggauss
from scipy.special import zeta

from hwoutils import constants as const
from hwoutils import conversions as conv


def test_lightweight_imports_and_modified_julian_dates():
    """Fails if core modules load JAX or the MJD origin changes; exact algebra."""
    for module in ("constants", "conversions", "epoch", "radial"):
        code = f"import sys; import hwoutils.{module}; assert 'jax' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)
    assert const.MJD_ZERO_JD == 2400000.5
    assert const.AB_ZERO_POINT_JY == conv.AB_ZERO_POINT_JY
    jd = np.array([2400000.5, 2451545.0, 2460000.75])
    expected = np.array([0.0, 51544.5, 60000.25])
    np.testing.assert_array_equal(conv.jd_to_mjd(jd), expected)
    np.testing.assert_array_equal(conv.mjd_to_jd(expected), jd)
    assert conv.jd_to_mjd(const.J2000_JD) == 51544.5
    with jax.enable_x64():
        np.testing.assert_array_equal(jax.jit(conv.jd_to_mjd)(jnp.array(jd)), expected)
    import hwoutils

    assert set(hwoutils.__all__) <= set(dir(hwoutils))
    assert callable(hwoutils.enable_x64)
    assert callable(hwoutils.set_host_device_count)
    assert callable(hwoutils.set_platform)


def test_planck_limits_broadcasting_and_derivatives():
    """Fails for wrong photon units, overflow or broken derivatives.

    Wien x=80 has <2e-35 relative correction; RJ x=1e-5 has <5.1e-6.
    Float64 arithmetic allows 3e-14; float32 and finite differences allow
    3e-5. Analytic log-derivatives also pin float32 second derivatives.
    """
    from hwoutils.planck import planck_photon_radiance

    with jax.enable_x64():
        temperature = 6000.0
        x = np.array([80.0, 1e-5])
        lam_m = const.h * const.c / (const.k_B * temperature * x)
        expected = np.array(
            [
                2 * const.c / lam_m[0] ** 4 * math.exp(-x[0]) * const.nm2m,
                2 * const.k_B * temperature / (const.h * lam_m[1] ** 3) * const.nm2m,
            ]
        )
        actual = jax.jit(planck_photon_radiance)(lam_m * const.m2nm, temperature)
        np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14)
        np.testing.assert_allclose(actual[1], expected[1], rtol=5.1e-6)
        wave = jnp.array([500.0, 10000.0])
        temps = jnp.array([[300.0], [6000.0]])
        lam = np.asarray(wave) * const.nm2m
        exponent = const.h * const.c / (lam * const.k_B * np.asarray(temps))
        truth = 2 * const.c * const.nm2m / lam**4 / np.expm1(exponent)
        np.testing.assert_allclose(
            planck_photon_radiance(wave, temps), truth, rtol=3e-14
        )
        check_grads(
            lambda w, t: jnp.log(planck_photon_radiance(w, t)),
            (jnp.array(10000.0), jnp.array(300.0)),
            order=2,
            modes=("fwd", "rev"),
            atol=3e-5,
            rtol=3e-5,
        )
    with jax.enable_x64(False):
        w, t = jnp.array(500.0), jnp.array(300.0)
        x = const.h * const.c / (const.k_B * 500 * const.nm2m * 300)
        # At x~96 the leading Wien log-slope and curvature are sufficient.
        slope = (x - 4) / 500
        curvature = (4 - 2 * x) / 500**2
        value, first, second = jax.jit(
            lambda w, t: (
                planck_photon_radiance(w, t),
                jax.grad(planck_photon_radiance)(w, t),
                jax.grad(jax.grad(planck_photon_radiance))(w, t),
            )
        )(w, t)
        np.testing.assert_allclose(first / value, slope, rtol=3e-5)
        np.testing.assert_allclose(second / value, slope**2 + curvature, rtol=3e-5)


def test_integrated_planck_photon_rate():
    """Fails for a pi or wavelength-density factor error.

    Integral B_ph d(lambda) = 4*zeta(3)*(k_B*T)^3/(h^3*c^2), per sr.
    A 160-node log-wavelength rule over x=[1e-5,80] omits <2.1e-11
    relative photon rate; allow 1e-9 for quadrature and float64 rounding.
    """
    from hwoutils.planck import planck_photon_radiance

    nodes, weights = leggauss(160)
    with jax.enable_x64():
        temperature = np.array([300.0, 6000.0])
        scale_nm = const.h * const.c / (const.k_B * temperature) * const.m2nm
        lo, hi = np.log(scale_nm / 80), np.log(scale_nm / 1e-5)
        logwave = lo[:, None] + (hi - lo)[:, None] * (nodes + 1) / 2
        wave = np.exp(logwave)
        values = planck_photon_radiance(wave, temperature[:, None])
        integral = np.sum(np.asarray(values) * wave * weights, axis=1) * (hi - lo) / 2
        expected = (
            4 * zeta(3) * (const.k_B * temperature) ** 3 / (const.h**3 * const.c**2)
        )
        np.testing.assert_allclose(integral, expected, rtol=1e-9)


def test_planck_inverse_and_energy_wavenumber_units():
    """Fails for wrong energy, Jacobian or hemisphere normalization.

    Compare cgs pi*B per cm^-1 with 2*pi*h*c^2*nu^3/expm1(h*c*nu/kT)
    in cgs primitives, at 3e-14 floating-point tolerance. Inverse finite
    differences allow 3e-5; a deep Wien input checks log-domain range.
    """
    from hwoutils.planck import brightness_temperature, planck_photon_radiance

    with jax.enable_x64():
        wave = jnp.array([4000.0, 10000.0, 18500.0])
        temperature = jnp.array([[200.0], [350.0]])
        photons = planck_photon_radiance(wave, temperature)
        energy = conv.photons_per_nm_per_m2_to_w_per_nm_per_m2(photons, wave)
        lam_m = np.asarray(wave) * const.nm2m
        expected = (
            2
            * const.h
            * const.c**2
            / lam_m**5
            / np.expm1(
                const.h * const.c / (lam_m * const.k_B * np.asarray(temperature))
            )
            * const.nm2m
        )
        np.testing.assert_allclose(energy, expected, rtol=3e-14)
        np.testing.assert_allclose(
            conv.w_per_nm_per_m2_to_photons_per_nm_per_m2(energy, wave),
            photons,
            rtol=3e-14,
        )
        nu = const.cm2nm / np.asarray(wave)
        h_cgs, c_cgs, k_cgs = (
            const.h * const.J2erg,
            const.c * const.nm2cm / const.nm2m,
            const.k_B * const.J2erg,
        )
        cgs = (
            2
            * math.pi
            * h_cgs
            * c_cgs**2
            * nu**3
            / np.expm1(h_cgs * c_cgs * nu / (k_cgs * np.asarray(temperature)))
        )
        actual = conv.w_per_nm_per_m2_to_cgs_per_wavenumber(math.pi * energy, wave)
        np.testing.assert_allclose(actual, cgs, rtol=3e-14)
        np.testing.assert_allclose(
            conv.cgs_per_wavenumber_to_w_per_nm_per_m2(cgs, wave),
            math.pi * expected,
            rtol=3e-14,
        )
        np.testing.assert_allclose(
            jax.jit(brightness_temperature)(wave, photons),
            np.broadcast_to(temperature, photons.shape),
            rtol=3e-14,
        )
        assert brightness_temperature(500.0, 0.0) == 0
        check_grads(
            lambda w, logb: brightness_temperature(w, jnp.exp(logb)),
            (jnp.array(10000.0), jnp.array(40.0)),
            order=2,
            modes=("fwd", "rev"),
            atol=3e-5,
            rtol=3e-5,
        )
    with jax.enable_x64(False):
        wave, photons = jnp.array(500.0), jnp.array(1e-30)
        lam_m = 500 * const.nm2m
        expected = (
            const.h
            * const.c
            / (lam_m * const.k_B)
            / math.log1p(2 * const.c / lam_m**4 * const.nm2m / 1e-30)
        )
        np.testing.assert_allclose(
            brightness_temperature(wave, photons), expected, rtol=3e-6
        )


def test_radial_center_is_the_geometric_array_center():
    """Fails if the optical-axis center is not (n - 1) / 2; exact arithmetic."""
    from hwoutils.radial import get_center

    for shape in ((3, 6), (4, 5), (1, 1)):
        assert get_center(shape) == ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
