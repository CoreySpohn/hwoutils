"""Photometric ratio and literature zero-point contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import constants as const
from hwoutils import conversions as conv


@pytest.mark.float64_only(
    reason="Photometry and angular diameter references at rtol=3e-15"
)
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
