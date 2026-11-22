"""Planck photon radiance and its inverse, with broadcastable JAX inputs."""

import math

import jax.numpy as jnp

from hwoutils.constants import c, h, k_B, nm2m, nm2um, um2m

# Micrometer arithmetic avoids tiny SI wavelength powers in float32 derivatives.
_EXPONENT_UM_K = h * c / (k_B * um2m)
_PHOTON_PREFACTOR = 2 * c * nm2m / um2m**4


def planck_photon_radiance(wavelength_nm, temperature_k):
    """Return Planck radiance in photons/s/m^2/nm/sr.

    Positive wavelengths [nm] and temperatures [K] broadcast using ordinary
    array rules; insert singleton axes explicitly for an outer product.
    The exact occupation factor exp(-x)/(1-exp(-x)) avoids Wien overflow.
    Half-exponent factors retain representable tails and their derivatives.
    There is no hemispheric pi or observer dilution: multiply by
    pi * (radius_m / distance_m)**2 for an unresolved uniform sphere.
    Multiply by photon energy through hwoutils.conversions for energy units.

    Args:
        wavelength_nm: Wavelength [nm], positive. Broadcasts against
            ``temperature_k``.
        temperature_k: Temperature [K], positive. Broadcasts against
            ``wavelength_nm``.

    Returns:
        Photon radiance [photons/s/m^2/nm/sr], with the broadcast shape of the inputs.
    """
    wavelength_um = jnp.asarray(wavelength_nm) * nm2um
    exponent = _EXPONENT_UM_K / (wavelength_um * jnp.asarray(temperature_k))
    decay = jnp.exp(-exponent / 2)
    amplitude = math.sqrt(_PHOTON_PREFACTOR) / wavelength_um**2 * decay
    return amplitude**2 / -jnp.expm1(-exponent)


def brightness_temperature(wavelength_nm, photon_radiance):
    """Invert photon radiance [photons/s/m^2/nm/sr] to temperature [K].

    Positive wavelength [nm] and nonnegative radiance broadcast. Zero
    radiance gives zero temperature; derivatives require positive radiance.
    Convert energy or wavenumber densities through hwoutils.conversions
    first, and divide hemispheric flux by pi. Log-domain arithmetic avoids
    overflow when radiance is very small. Inputs are not clipped.

    Args:
        wavelength_nm: Wavelength [nm], positive. Broadcasts against
            ``photon_radiance``.
        photon_radiance: Photon radiance [photons/s/m^2/nm/sr], nonnegative.

    Returns:
        Brightness temperature [K], with the broadcast shape of the inputs.
    """
    wavelength_um = jnp.asarray(wavelength_nm) * nm2um
    log_ratio = (
        math.log(_PHOTON_PREFACTOR)
        - 4 * jnp.log(wavelength_um)
        - jnp.log(jnp.asarray(photon_radiance))
    )
    return (_EXPONENT_UM_K / wavelength_um) / jnp.logaddexp(0.0, log_ratio)
