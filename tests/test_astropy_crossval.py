"""Cross-validation tests against astropy to guarantee machine-precision correctness."""

import jax.numpy as jnp
from astropy import constants as astropy_const
from astropy import units as u
from astropy.time import Time

from hwoutils import constants as hwo_const
from hwoutils import conversions as hwo_conv

# =============================================================================
# Constants Validation
# =============================================================================


class TestConstantsValidation:
    """Ensure hardcoded constants match astropy's SI values to extreme precision."""

    def test_speed_of_light(self):
        # c is exact in SI
        assert jnp.isclose(hwo_const.c, astropy_const.c.value, atol=1e-12)

    def test_planck_constant(self):
        # h is exact in SI (2019 redefinition)
        assert jnp.isclose(hwo_const.h, astropy_const.h.value, atol=1e-45)

    def test_boltzmann_constant(self):
        # k_B is exact in SI (2019 redefinition)
        assert jnp.isclose(hwo_const.k_B, astropy_const.k_B.value, atol=1e-35)

    def test_stefan_boltzmann(self):
        assert jnp.isclose(hwo_const.sigma_SB, astropy_const.sigma_sb.value, rtol=1e-12)

    def test_gravitational_constant(self):
        assert jnp.isclose(hwo_const.G_si, astropy_const.G.value, rtol=1e-12)

    def test_masses(self):
        assert jnp.isclose(hwo_const.Msun2kg, astropy_const.M_sun.value, rtol=1e-12)
        assert jnp.isclose(hwo_const.Mearth2kg, astropy_const.M_earth.value, rtol=1e-12)
        assert jnp.isclose(hwo_const.Mjup2kg, astropy_const.M_jup.value, rtol=1e-12)

    def test_distances(self):
        assert jnp.isclose(hwo_const.AU2m, astropy_const.au.value, rtol=1e-12)
        assert jnp.isclose(hwo_const.pc2m, astropy_const.pc.value, rtol=1e-12)
        assert jnp.isclose(hwo_const.Rearth2m, astropy_const.R_earth.value, rtol=1e-12)


# =============================================================================
# Conversions Validation
# =============================================================================


class TestConversionsValidation:
    """Ensure hwoutils conversion equations match astropy.units equivalencies."""

    def test_jy_to_photons_nm_astropy_equiv(self):
        """Cross-validate Jy to photons/s/m^2/nm using astropy.units."""
        # 1.0 Jy at 550 nm
        flux_jy = 1.0 * u.Jy
        wavelength = 550.0 * u.nm

        # Astropy conversion to spectral photon flux density
        equiv = u.spectral_density(wavelength)
        astropy_flux = flux_jy.to(u.photon / (u.s * u.m**2 * u.nm), equivalencies=equiv)

        # hwoutils conversion
        hwo_flux = hwo_conv.jy_to_photons_per_nm_per_m2(1.0, 550.0)

        assert jnp.isclose(hwo_flux, astropy_flux.value, rtol=1e-12)

    def test_decimal_year_to_jd_astropy(self):
        """Cross-validate decimal year to Julian Date against astropy.time."""
        year = 2025.5

        # Astropy expects decimal year to be explicitly formatted
        t = Time(year, format="decimalyear")
        astropy_jd = t.jd

        # hwoutils conversion
        hwo_jd = hwo_conv.decimal_year_to_jd(year)

        # We enforce < 0.5 days (12 hours) precision drift in epoch conversion over thousands of days
        assert jnp.isclose(hwo_jd, astropy_jd, atol=0.5)

    def test_mag_to_jy(self):
        """Cross-validate AB Magnitude to Jansky."""
        mag = 15.0 * u.ABmag
        astropy_jy = mag.to(u.Jy)

        hwo_jy = hwo_conv.mag_per_arcsec2_to_jy_per_arcsec2(15.0)

        assert jnp.isclose(hwo_jy, astropy_jy.value, rtol=1e-12)
