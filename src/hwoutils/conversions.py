"""Unit conversion functions using centralized constants.

Pure JAX implementations — no astropy dependency. Functions are intentionally
NOT JIT-compiled so JAX can fuse them into larger computation graphs.
"""

import jax.numpy as jnp

from hwoutils import constants as const

# ---------------------------------------------------------------------------
# Flux conversions
# ---------------------------------------------------------------------------


def jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm):
    """Convert flux density from Janskys to photons/s/nm/m².

    Args:
        flux_jy: Flux density in Janskys.
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Flux density in photons/s/nm/m².
    """
    return flux_jy * const.Jy / (wavelength_nm * const.h)


def photons_per_nm_per_m2_to_jy(flux_phot, wavelength_nm):
    """Convert flux density from photons/s/nm/m² to Janskys.

    Args:
        flux_phot: Flux density in photons/s/nm/m².
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Flux density in Janskys.
    """
    return flux_phot * (wavelength_nm * const.h) / const.Jy


def mag_per_arcsec2_to_jy_per_arcsec2(mag_per_arcsec2):
    """Convert surface brightness from mag/arcsec² to Jy/arcsec² (AB).

    Args:
        mag_per_arcsec2: Surface brightness in magnitudes per arcsec².

    Returns:
        Surface brightness in Jy/arcsec².
    """
    f0_jy = 3631.0  # AB magnitude zero point
    return f0_jy * 10 ** (-0.4 * mag_per_arcsec2)


# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------


def nm_to_um(length_nm):
    """Convert nanometers to micrometers."""
    return length_nm * const.nm2um


def um_to_nm(length_um):
    """Convert micrometers to nanometers."""
    return length_um * const.um2nm


def au_to_m(length_au):
    """Convert AU to meters."""
    return length_au * const.AU2m


def m_to_au(length_m):
    """Convert meters to AU."""
    return length_m * const.m2AU


def Rearth_to_m(length_Rearth):
    """Convert Earth radii to meters."""
    return length_Rearth * const.Rearth2m


# ---------------------------------------------------------------------------
# Velocity conversions
# ---------------------------------------------------------------------------


def au_per_yr_to_m_per_s(velocity_au_per_yr):
    """Convert AU/yr to m/s."""
    return velocity_au_per_yr * const.AU2m / const.yr2s


# ---------------------------------------------------------------------------
# Angular conversions
# ---------------------------------------------------------------------------


def arcsec_to_rad(angle_arcsec):
    """Convert arcseconds to radians."""
    return angle_arcsec * const.arcsec2rad


def rad_to_arcsec(angle_rad):
    """Convert radians to arcseconds."""
    return angle_rad * const.rad2arcsec


def mas_to_arcsec(angle_mas):
    """Convert milliarcseconds to arcseconds."""
    return angle_mas * const.mas2arcsec


def arcsec_to_mas(angle_arcsec):
    """Convert arcseconds to milliarcseconds."""
    return angle_arcsec * const.arcsec2mas


def arcsec_to_lambda_d(angle_arcsec, wavelength_nm, diameter_m):
    """Convert angular separation to lambda/D units.

    Args:
        angle_arcsec: Angular separation in arcseconds.
        wavelength_nm: Wavelength in nanometers.
        diameter_m: Telescope diameter in meters.

    Returns:
        Angular separation in lambda/D.
    """
    angle_rad = angle_arcsec * const.arcsec2rad
    wavelength_m = wavelength_nm * const.nm2m
    lambda_d_rad = wavelength_m / diameter_m
    return angle_rad / lambda_d_rad


def lambda_d_to_arcsec(angle_lambda_d, wavelength_nm, diameter_m):
    """Convert lambda/D units to angular separation in arcseconds.

    Args:
        angle_lambda_d: Angular separation in lambda/D.
        wavelength_nm: Wavelength in nanometers.
        diameter_m: Telescope diameter in meters.

    Returns:
        Angular separation in arcseconds.
    """
    wavelength_m = wavelength_nm * const.nm2m
    lambda_d_rad = wavelength_m / diameter_m
    angle_rad = angle_lambda_d * lambda_d_rad
    return angle_rad * const.rad2arcsec


# ---------------------------------------------------------------------------
# Mass conversions
# ---------------------------------------------------------------------------


def Msun_to_kg(mass_solar):
    """Convert solar masses to kilograms."""
    return mass_solar * const.Msun2kg


def Mearth_to_kg(mass_earth):
    """Convert Earth masses to kilograms."""
    return mass_earth * const.Mearth2kg


# ---------------------------------------------------------------------------
# Distance conversions
# ---------------------------------------------------------------------------


def au_to_arcsec(distance_au, distance_pc):
    """Convert physical distance in AU to angular separation.

    Args:
        distance_au: Physical distance in AU.
        distance_pc: Distance to system in parsecs.

    Returns:
        Angular separation in arcseconds.
    """
    return distance_au / distance_pc


def arcsec_to_au(angle_arcsec, distance_pc):
    """Convert angular separation to physical distance.

    Args:
        angle_arcsec: Angular separation in arcseconds.
        distance_pc: Distance to system in parsecs.

    Returns:
        Physical distance in AU.
    """
    return angle_arcsec * distance_pc


# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------


def years_to_days(time_years):
    """Convert years to days."""
    return time_years * 365.25


def days_to_years(time_days):
    """Convert days to years."""
    return time_days / 365.25


def is_leap_year(year):
    """Determine if a year is a leap year.

    Args:
        year: The year to check.

    Returns:
        True if the year is a leap year.
    """
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def days_in_year(year):
    """Return the number of days in a year (365 or 366).

    Args:
        year: The year to check.

    Returns:
        Number of days.
    """
    return 365 + is_leap_year(year)


def gregorian_to_jd(year, month, day):
    """Convert a Gregorian date to a Julian day.

    Args:
        year: The year.
        month: The month.
        day: The day.

    Returns:
        The Julian day.
    """
    a = jnp.floor((14 - month) / 12)
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (
        day
        + jnp.floor((153 * m + 2) / 5)
        + 365 * y
        + jnp.floor(y / 4)
        - jnp.floor(y / 100)
        + jnp.floor(y / 400)
        - 32045
    )
    return jdn - 0.5


def jd_to_decimal_year(jd):
    """Convert a Julian day to a decimal year.

    Args:
        jd: The Julian day.

    Returns:
        The decimal year.
    """
    year_approx = 1970.0 + (jd - 2440587.5) / 365.2425
    year = jnp.floor(year_approx)

    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    year = jnp.where(jd < jd_start, year - 1, year)
    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    return year + (jd - jd_start) / (jd_end - jd_start)


def decimal_year_to_jd(decimal_year):
    """Convert a decimal year to a Julian day.

    Args:
        decimal_year: The decimal year.

    Returns:
        The Julian day.
    """
    year = jnp.floor(decimal_year)
    year_fraction = decimal_year - year

    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    return jd_start + year_fraction * (jd_end - jd_start)
