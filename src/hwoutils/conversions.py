"""Unit conversion functions using centralized constants.

Arithmetic functions accept Python scalars, NumPy arrays and JAX arrays.
Only logarithmic and calendar functions load JAX, on first call. Functions
are not JIT-compiled so JAX can fuse them into larger computation graphs.
"""

from hwoutils import constants as const
from hwoutils.constants import AB_ZERO_POINT_JY as AB_ZERO_POINT_JY

# ---------------------------------------------------------------------------
# Flux conversions
# ---------------------------------------------------------------------------


def jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm):
    """Convert flux density from Janskys to photons/s/nm/m^2.

    Args:
        flux_jy: Flux density in Janskys.
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Flux density in photons/s/nm/m^2.
    """
    return flux_jy * const.Jy / (wavelength_nm * const.h)


def photons_per_nm_per_m2_to_jy(flux_phot, wavelength_nm):
    """Convert flux density from photons/s/nm/m^2 to Janskys.

    Args:
        flux_phot: Flux density in photons/s/nm/m^2.
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Flux density in Janskys.
    """
    return flux_phot * (wavelength_nm * const.h) / const.Jy


def photons_per_nm_per_m2_to_w_per_nm_per_m2(flux_phot, wavelength_nm):
    """Convert photons/s/nm/m^2 to W/nm/m^2, retaining any per-sr factor."""
    return flux_phot * (const.h * const.c / const.nm2m) / wavelength_nm


def w_per_nm_per_m2_to_photons_per_nm_per_m2(flux_w, wavelength_nm):
    """Convert W/nm/m^2 to photons/s/nm/m^2, retaining any per-sr factor."""
    return flux_w * wavelength_nm / (const.h * const.c / const.nm2m)


def w_per_nm_per_m2_to_cgs_per_wavenumber(flux_w, wavelength_nm):
    """Convert W/m^2/nm to erg/s/cm^2/(cm^-1), retaining any per-sr factor.

    Apply |d(lambda_nm)/d(nu_cm)| = lambda_nm^2 / cm2nm.
    This changes the density coordinate only, with no solid-angle integral.
    """
    return flux_w * (const.J2erg * const.cm22m2 / const.cm2nm) * wavelength_nm**2


def cgs_per_wavenumber_to_w_per_nm_per_m2(flux_cgs, wavelength_nm):
    """Convert erg/s/cm^2/(cm^-1) to W/m^2/nm, retaining any per-sr factor."""
    return flux_cgs / wavelength_nm**2 / (const.J2erg * const.cm22m2 / const.cm2nm)


def mag_to_flux_jy(mag):
    """AB magnitude to flux density in Jansky."""
    return AB_ZERO_POINT_JY * 10.0 ** (-0.4 * mag)


def flux_jy_to_mag(flux_jy):
    """Flux density in Jansky to AB magnitude."""
    import jax.numpy as jnp

    return -2.5 * jnp.log10(flux_jy / AB_ZERO_POINT_JY)


def contrast_to_delta_mag(contrast):
    """Convert a flux ratio F / F_ref to m - m_ref (same photometric band).

    Zero gives +inf and negative ratios give NaN, without clipping. Callers
    requiring a finite floor must apply it explicitly. JAX loads on first call.
    """
    import jax.numpy as jnp

    return -2.5 * jnp.log10(contrast)


def delta_mag_to_contrast(delta_mag):
    """Convert m - m_ref to the flux ratio F / F_ref in the same band."""
    return 10.0 ** (-0.4 * delta_mag)


def mag_per_arcsec2_to_jy_per_arcsec2(mag_per_arcsec2):
    """Convert surface brightness from mag/arcsec^2 to Jy/arcsec^2 (AB).

    Args:
        mag_per_arcsec2: Surface brightness in magnitudes per arcsec^2.

    Returns:
        Surface brightness in Jy/arcsec^2.
    """
    return mag_to_flux_jy(mag_per_arcsec2)


def mjy_per_sr_to_jy_per_arcsec2(surface_brightness):
    """Convert surface brightness from MJy/sr to Jy/arcsec^2.

    Args:
        surface_brightness: Surface brightness in MJy/sr.

    Returns:
        Surface brightness in Jy/arcsec^2.
    """
    return surface_brightness * 1e6 * const.arcsec2rad**2


def jy_per_arcsec2_to_mjy_per_sr(surface_brightness):
    """Convert surface brightness from Jy/arcsec^2 to MJy/sr.

    Args:
        surface_brightness: Surface brightness in Jy/arcsec^2.

    Returns:
        Surface brightness in MJy/sr.
    """
    return surface_brightness * 1e-6 * const.rad2arcsec**2


def mjy_per_sr_to_jy(surface_brightness, solid_angle_sr):
    """Flux density in Jy of a uniform MJy/sr surface brightness over a solid angle.

    Args:
        surface_brightness: Surface brightness in MJy/sr.
        solid_angle_sr: Solid angle in steradians (e.g. one detector pixel).

    Returns:
        Flux density in Jy.
    """
    return surface_brightness * 1e6 * solid_angle_sr


# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------


def nm_to_um(length_nm):
    """Convert nanometers to micrometers."""
    return length_nm * const.nm2um


def um_to_nm(length_um):
    """Convert micrometers to nanometers."""
    return length_um * const.um2nm


def nm_to_wavenumber_cm(wavelength_nm):
    """Convert wavelength [nm] to wavenumber [cm^-1] (nu = 1 / lambda).

    Args:
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Wavenumber in inverse centimeters.
    """
    return const.cm2nm / wavelength_nm


def wavenumber_cm_to_nm(wavenumber_cm):
    """Convert wavenumber [cm^-1] to wavelength [nm].

    Args:
        wavenumber_cm: Wavenumber in inverse centimeters.

    Returns:
        Wavelength in nanometers.
    """
    return const.cm2nm / wavenumber_cm


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


def angular_diameter_arcsec(radius_m, distance_pc):
    """Small-angle diameter 2 R / d in arcsec, for radius [m], distance [pc].

    Inputs broadcast through ordinary arithmetic; radius may be zero and
    distance must be positive. No finite-distance arcsine correction is used.
    """
    return rad_to_arcsec(2.0 * radius_m / (distance_pc * const.pc2m))


def angular_diameter_mas(radius_m, distance_pc):
    """Small-angle diameter 2 R / d in mas, for radius [m], distance [pc]."""
    return arcsec_to_mas(angular_diameter_arcsec(radius_m, distance_pc))


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


def jd_to_mjd(time_jd):
    """Convert Julian Date to Modified Julian Date without changing time scale.

    Inputs are scalars or arrays; absolute dates need float64 for sub-day
    precision. For two-part dates use Epoch.days with Epoch(MJD_ZERO_JD).
    """
    return time_jd - const.MJD_ZERO_JD


def mjd_to_jd(time_mjd):
    """Convert Modified Julian Date to Julian Date without changing time scale."""
    return time_mjd + const.MJD_ZERO_JD


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
    import jax.numpy as jnp

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
    import jax.numpy as jnp

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
    import jax.numpy as jnp

    year = jnp.floor(decimal_year)
    year_fraction = decimal_year - year

    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    return jd_start + year_fraction * (jd_end - jd_start)


# ---------------------------------------------------------------------------
# Albedo conventions
# ---------------------------------------------------------------------------

# Lambertian-sphere ratio of geometric to spherical (Bond) albedo.
# Seager 2010, eq 3.36: a uniformly Lambertian sphere has A_g = (2/3) A_S.
SPHERICAL_TO_GEOMETRIC_ALBEDO_LAMBERTIAN: float = 2.0 / 3.0


def spherical_to_geometric_albedo(A_S):
    """Convert spherical albedo to geometric albedo (Lambertian sphere).

    Args:
        A_S: Spherical (Bond) albedo, dimensionless.

    Returns:
        Geometric albedo A_g, dimensionless.

    Notes:
        Exact for a uniformly Lambertian-reflecting sphere
        (Seager 2010, eq 3.36): A_g = (2/3) * A_S. For atmospheres
        with anisotropic scattering (Mie clouds, etc.) the actual
        A_g / A_S ratio depends on the scattering phase function;
        this is the leading-order approximation that the ExoJaxPhysicalModel
        uses to convert its 2-stream plane-parallel reflectivity to a
        disk-integrated geometric albedo.
    """
    return SPHERICAL_TO_GEOMETRIC_ALBEDO_LAMBERTIAN * A_S


def geometric_to_spherical_albedo(A_g):
    """Convert geometric albedo to spherical albedo (Lambertian sphere).

    Inverse of :func:`spherical_to_geometric_albedo`:
    A_S = (3/2) * A_g.
    """
    return A_g / SPHERICAL_TO_GEOMETRIC_ALBEDO_LAMBERTIAN
