"""Tests for hwoutils.conversions — unit conversion functions."""

import jax.numpy as jnp

from hwoutils import constants as const
from hwoutils import conversions as conv

# =============================================================================
# Flux Conversions
# =============================================================================


class TestFluxConversions:
    """Tests for flux conversion functions."""

    def test_jy_to_photons_roundtrip(self):
        """Verify Jy to photons conversion is reversible."""
        wavelength_nm = jnp.array([500.0, 600.0, 700.0])
        flux_jy = jnp.array([[1.0, 1.0, 1.0]])

        flux_phot = conv.jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm)
        flux_jy_back = conv.photons_per_nm_per_m2_to_jy(flux_phot, wavelength_nm)

        assert jnp.allclose(flux_jy, flux_jy_back, rtol=1e-6)

    def test_jy_to_photons_wavelength_dependence(self):
        """Per-nm flux: longer wavelengths have fewer photons/nm."""
        wavelength_nm = jnp.array([400.0, 800.0])
        flux_jy = jnp.array([[1.0, 1.0]])

        flux_phot = conv.jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm)

        assert flux_phot[0, 0] > flux_phot[0, 1]

    def test_mag_to_jy_conversion(self):
        """Verify AB magnitude to Jy conversion: 0 mag = 3631 Jy."""
        flux_jy = conv.mag_per_arcsec2_to_jy_per_arcsec2(0.0)
        assert jnp.isclose(flux_jy, 3631.0, rtol=1e-3)

    def test_jy_to_photons_consistency(self):
        """Verify Jy → photons/s/m²/nm against independent first-principles calc.

        Physics: F_photons = F_nu * Jy * c / (h * lam^2) (per nm).
        """
        FLUX_JY = 3631.0
        WAVELENGTH_NM = 550.0

        flux_phot = conv.jy_to_photons_per_nm_per_m2(FLUX_JY, WAVELENGTH_NM)

        wavelength_m = WAVELENGTH_NM * 1e-9
        energy_per_photon = const.h * const.c / wavelength_m
        c_over_lambda_sq_per_nm = const.c / (wavelength_m**2) * 1e-9
        flux_power_per_nm = FLUX_JY * const.Jy * c_over_lambda_sq_per_nm
        expected = flux_power_per_nm / energy_per_photon

        assert jnp.isclose(flux_phot, expected, rtol=0.01)

    def test_stellar_flux_order_of_magnitude(self):
        """A 0 mag star should produce ~10⁷ ph/s/m²/nm at V-band."""
        flux_phot = conv.jy_to_photons_per_nm_per_m2(3631.0, 550.0)
        assert 1e6 < flux_phot < 1e9

    def test_spectral_binning_wide_vs_narrow(self):
        """Wide bin vs N narrow bins should give consistent total flux."""
        FLUX_JY = 1000.0
        BANDWIDTH = 100.0

        wide = conv.jy_to_photons_per_nm_per_m2(FLUX_JY, 550.0) * BANDWIDTH

        wavelengths = jnp.linspace(505.0, 595.0, 10)
        narrow_fluxes = conv.jy_to_photons_per_nm_per_m2(FLUX_JY, wavelengths)
        narrow = jnp.sum(narrow_fluxes) * (BANDWIDTH / 10)

        assert jnp.abs(narrow - wide) / wide < 0.05


# =============================================================================
# Length Conversions
# =============================================================================


class TestLengthConversions:
    """Tests for length conversion functions."""

    def test_nm_um_roundtrip(self):
        """Verify nm to um conversion is reversible."""
        assert jnp.isclose(conv.um_to_nm(conv.nm_to_um(550.0)), 550.0)

    def test_nm_to_um_value(self):
        """1000 nm = 1 um."""
        assert jnp.isclose(conv.nm_to_um(1000.0), 1.0)

    def test_au_m_roundtrip(self):
        """Verify AU to m conversion is reversible."""
        assert jnp.isclose(conv.m_to_au(conv.au_to_m(1.0)), 1.0)

    def test_au_to_m_value(self):
        """1 AU ≈ 1.496e11 m."""
        assert jnp.isclose(conv.au_to_m(1.0), const.AU2m, rtol=1e-6)

    def test_rearth_to_m(self):
        """1 Earth radius ≈ 6.371e6 m."""
        assert jnp.isclose(conv.Rearth_to_m(1.0), const.Rearth2m, rtol=1e-3)


# =============================================================================
# Angular Conversions
# =============================================================================


class TestAngularConversions:
    """Tests for angular conversion functions."""

    def test_arcsec_rad_roundtrip(self):
        """Verify arcsec to rad conversion is reversible."""
        assert jnp.isclose(conv.rad_to_arcsec(conv.arcsec_to_rad(1.0)), 1.0)

    def test_arcsec_to_rad_value(self):
        """1 arcsec = pi / (180 * 3600) rad."""
        expected = jnp.pi / (180.0 * 3600.0)
        assert jnp.isclose(conv.arcsec_to_rad(1.0), expected)

    def test_mas_arcsec_roundtrip(self):
        """Verify mas to arcsec conversion is reversible."""
        assert jnp.isclose(conv.arcsec_to_mas(conv.mas_to_arcsec(100.0)), 100.0)

    def test_lambda_d_arcsec_roundtrip(self):
        """Verify λ/D to arcsec conversion is reversible."""
        angle_arcsec = conv.lambda_d_to_arcsec(5.0, 600.0, 6.0)
        angle_lod_back = conv.arcsec_to_lambda_d(angle_arcsec, 600.0, 6.0)
        assert jnp.isclose(angle_lod_back, 5.0, rtol=1e-6)

    def test_lambda_d_wavelength_scaling(self):
        """λ/D angular size must scale linearly with wavelength."""
        lod_blue = conv.lambda_d_to_arcsec(1.0, 500.0, 6.0)
        lod_red = conv.lambda_d_to_arcsec(1.0, 1000.0, 6.0)
        assert jnp.isclose(lod_red / lod_blue, 2.0, rtol=1e-6)


# =============================================================================
# Time Conversions
# =============================================================================


class TestTimeConversions:
    """Tests for time conversion functions."""

    def test_years_days_roundtrip(self):
        """Verify years to days conversion is reversible."""
        t = jnp.array([1.0])
        assert jnp.allclose(conv.days_to_years(conv.years_to_days(t)), t)

    def test_years_to_days_value(self):
        """1 year = 365.25 days."""
        assert jnp.isclose(conv.years_to_days(jnp.array([1.0]))[0], 365.25)

    def test_decimal_year_to_jd(self):
        """J2000.0 = 2451545.0 JD."""
        jd = conv.decimal_year_to_jd(jnp.array([2000.0]))
        assert jnp.isclose(jd[0], 2451545.0, rtol=1e-3)


# =============================================================================
# Distance Conversions
# =============================================================================


class TestDistanceConversions:
    """Tests for distance conversion functions."""

    def test_au_arcsec_roundtrip(self):
        """Verify AU to arcsec conversion is reversible."""
        d = jnp.array([1.0])
        assert jnp.allclose(conv.arcsec_to_au(conv.au_to_arcsec(d, 10.0), 10.0), d)

    def test_au_arcsec_at_10pc(self):
        """1 AU at 10 pc = 0.1 arcsec."""
        assert jnp.isclose(conv.au_to_arcsec(jnp.array([1.0]), 10.0)[0], 0.1)


# =============================================================================
# Mass Conversions
# =============================================================================


class TestMassConversions:
    """Tests for mass conversion functions."""

    def test_msun_to_kg(self):
        """1 Msun ≈ 1.989e30 kg."""
        assert jnp.isclose(conv.Msun_to_kg(1.0), const.Msun2kg, rtol=1e-3)

    def test_mearth_to_kg(self):
        """1 Mearth ≈ 5.972e24 kg."""
        assert jnp.isclose(conv.Mearth_to_kg(1.0), const.Mearth2kg, rtol=1e-3)
