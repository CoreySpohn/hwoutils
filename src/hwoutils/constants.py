"""Physical constants and unit conversion factors.

Single source of truth for all constants used across the HWO simulation suite
(yippy, orbix, coronagraphoto, photomancy, hwosim). All values are plain floats
-- no array-library or astropy.units dependency.
"""

import math

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------
two_pi = 2 * math.pi
pi_over_2 = math.pi / 2
eps = 2.0**-23  # IEEE 754 binary32 machine epsilon
GAUSSIAN_FWHM_FACTOR = 2 * math.sqrt(2 * math.log(2))  # sigma to FWHM

# ---------------------------------------------------------------------------
# Fundamental physical constants (SI)
# ---------------------------------------------------------------------------
h = 6.62607015e-34  # Planck constant [J*s]
c = 299792458.0  # Speed of light [m/s]
k_B = 1.380649e-23  # Boltzmann constant [J/K]
sigma_SB = 5.6703744191844314e-08  # Stefan-Boltzmann constant [W*m^-2*K^-4]

# ---------------------------------------------------------------------------
# Gravitational constant
# ---------------------------------------------------------------------------
G_si = 6.67430e-11  # [m^3*kg^-1*s^-2]
G = 1.488185170234519e-34  # [AU^3*kg^-1*d^-2] -- for orbital dynamics

# ---------------------------------------------------------------------------
# Flux conversion
# ---------------------------------------------------------------------------
Jy = 1e-26  # 1 Jansky [W*m^-2*Hz^-1]
# Exact AB convention, 10**(-0.4 * 48.60) erg/s/cm^2/Hz, expressed in Jy.
AB_ZERO_POINT_JY = 3630.7805477010028
# Johnson V zero-magnitude flux density in the Vega-based system.
# Bessell, Castelli & Plez (1998), A&A 333, 231, Table A2:
# f_nu = 3.636e-20 erg/s/cm^2/Hz at effective wavelength 0.545 um.
# https://articles.adsabs.harvard.edu/pdf/1998A%26A...333..231B
JOHNSON_V_VEGA_ZERO_POINT_JY = 3636.0
J2erg = 1e7  # Joules to ergs

# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------
nm2m = 1e-9
m2nm = 1e9
um2m = 1e-6
m2um = 1e6
nm2um = 1e-3
um2nm = 1e3
nm2cm = 1e-7
cm2nm = 1e7

# ---------------------------------------------------------------------------
# Distance conversions
# ---------------------------------------------------------------------------
AU2m = 1.495978707e11
m2AU = 6.684587122268445e-12
pc2m = 3.085677581491367e16
m2pc = 1.0 / pc2m
pc2AU = 2.062648062470964e05
Rearth2m = 6.3781e6  # Astropy/IAU default (equatorial)
Rearth2AU = Rearth2m * m2AU
Rsun2m = 6.957e8  # IAU 2015 nominal solar radius (Resolution B3)
Rsun2AU = Rsun2m * m2AU

# ---------------------------------------------------------------------------
# Mass conversions
# ---------------------------------------------------------------------------
Msun2kg = 1.988409870698051e30
Mearth2kg = 5.972167867791379e24
Mjup2kg = 1.8981245973360505e27
GM_SUN_AU3_PER_D2 = G * Msun2kg  # [AU^3*d^-2] solar gravitational parameter

# ---------------------------------------------------------------------------
# Angular conversions
# ---------------------------------------------------------------------------
rad2arcsec = 206264.80624709636
arcsec2rad = 4.84813681109536e-06
mas2arcsec = 1e-3
arcsec2mas = 1e3
deg2rad = math.pi / 180.0
rad2deg = 180.0 / math.pi

# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------
yr2s = 365.25 * 86400.0
s2yr = 1.0 / yr2s
d2s = 86400.0
s2d = 1.157407407407407e-05
J2000_JD = 2451545.0
MJD_ZERO_JD = 2400000.5  # JD origin of Modified Julian Date

# ---------------------------------------------------------------------------
# Area conversions
# ---------------------------------------------------------------------------
m22cm2 = 1e4
cm22m2 = 1e-4
