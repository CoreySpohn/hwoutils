# Constants, conversions and the epoch

Every library in the suite reads its physical constants, unit factors and time
origin from here, so a number such as the Julian date of J2000 or the solar
gravitational parameter has one value everywhere. These three modules need only
NumPy.

## A constant is a plain float

The orbital period of a body at 1 AU around the Sun follows from Kepler's third
law and the solar gravitational parameter, which `hwoutils.constants` holds in
AU and days.

```python
import numpy as np

from hwoutils import constants as const

period_d = 2 * np.pi * np.sqrt(1.0**3 / const.GM_SUN_AU3_PER_D2)
print(f"{period_d:.2f} d")
assert abs(period_d - 365.257) < 0.01
```

`GM_SUN_AU3_PER_D2` is `G * Msun2kg`, the product of the gravitational constant
in AU, kilograms and days and the solar mass. The module groups its names by what
they hold, and each name states its unit in its form:

| Group | Names |
| --- | --- |
| Mathematical | `two_pi`, `pi_over_2`, `eps`, `GAUSSIAN_FWHM_FACTOR` |
| Physical, SI | `h`, `c`, `k_B`, `sigma_SB`, `G_si`, `G`, `Jy` |
| Photometric zero points | `AB_ZERO_POINT_JY`, `JOHNSON_V_VEGA_ZERO_POINT_JY` |
| Length | `nm2m`, `m2nm`, `um2m`, `m2um`, `AU2m`, `m2AU`, `pc2m`, `pc2AU`, `Rearth2m`, `Rsun2m` |
| Mass | `Msun2kg`, `Mearth2kg`, `Mjup2kg`, `GM_SUN_AU3_PER_D2` |
| Angle | `rad2arcsec`, `arcsec2rad`, `mas2arcsec`, `arcsec2mas`, `deg2rad`, `rad2deg` |
| Time | `d2s`, `s2d`, `yr2s`, `s2yr`, `J2000_JD`, `MJD_ZERO_JD` |

A name such as `AU2m` reads "AU to metres": multiply a length in AU by it to get
metres.

## A conversion is a function

`hwoutils.conversions` turns a quantity in one unit into another, where the
factor alone is not enough because the conversion needs a second quantity. Angles
need a wavelength and an aperture to become multiples of $\lambda/D$, and a
separation in AU needs a distance to become an angle.

```python continuation
from hwoutils import conversions as conv

# 0.1 arcsec for a 6 m telescope at 550 nm, in units of lambda over D
separation_lod = conv.arcsec_to_lambda_d(0.1, 550.0, 6.0)
# 1 AU at 10 pc, in arcsec
separation_arcsec = conv.au_to_arcsec(1.0, 10.0)
print(f"{separation_lod:.2f} lambda/D, {separation_arcsec:.2f} arcsec")
assert abs(separation_arcsec - 0.1) < 1e-6
```

Surface brightness in magnitudes per square arcsecond becomes Jy per square
arcsecond through the AB zero point, so the conversion and the constant cannot
disagree.

```python continuation
flux_jy = conv.mag_per_arcsec2_to_jy_per_arcsec2(22.0)
assert abs(flux_jy - const.AB_ZERO_POINT_JY * 10 ** (-0.4 * 22.0)) < 1e-15
```

## An epoch is where time offsets start

Traced code carries times as plain arrays of days from a day zero, so a float32
program keeps sub-second resolution over decades. The day zero is an
`hwoutils.epoch.Epoch`, a two-part Julian date on one time scale, and it turns
Julian dates into offsets once, on the host in float64.

```python continuation
from hwoutils.epoch import Epoch

epoch = Epoch(const.J2000_JD)
time_d = epoch.days(jd=const.J2000_JD + 365.25)
print(time_d)
assert float(time_d) == 365.25
assert float(epoch.jd(time_d)) == const.J2000_JD + 365.25
```

An epoch carries a time scale label, `tdb` by default, and never converts between
scales. Handing it a time object such as an `astropy.time.Time` on a different scale
raises, instead of treating the two scales as one.
