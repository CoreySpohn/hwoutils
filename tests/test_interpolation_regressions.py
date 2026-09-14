"""Independent checks of cubic stencils, dtype handling, and affine gradients."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.map_coordinates import map_coordinates
from hwoutils.transforms import resample_flux


@pytest.fixture(autouse=True)
def enable_x64():
    """Use float64 when comparing two independent polynomial evaluations."""
    with jax.enable_x64():
        yield


def keys_reference(image, coords, cval):
    """Evaluate the original piecewise Keys polynomial with NumPy stencils."""
    result = np.zeros(np.broadcast_shapes(*(c.shape for c in coords)))
    for offsets in itertools.product(range(-1, 3), repeat=2):
        indices = [
            np.floor(c).astype(int) + o for c, o in zip(coords, offsets, strict=True)
        ]
        weight = 1
        valid = np.ones_like(result, dtype=bool)
        for c, i, n in zip(coords, indices, image.shape, strict=True):
            t = np.abs(c - i)
            weight = weight * np.where(
                t <= 1,
                1.5 * t**3 - 2.5 * t**2 + 1,
                np.where(t <= 2, -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2, 0),
            )
            valid = valid & (i >= 0) & (i < n)
        value = image[
            tuple(
                np.clip(i, 0, n - 1) for i, n in zip(indices, image.shape, strict=True)
            )
        ]
        result += weight * np.where(valid, value, cval)
    return result


def test_optimized_cubic_matches_piecewise_reference():
    """Polynomial factoring must preserve the 16-tap stencil at boundaries."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(12, 13))
    coords = [rng.uniform(-2, 14, size=(5, 7)), rng.uniform(-2, 15, size=(5, 7))]
    expected = keys_reference(source, coords, cval=2.3)
    actual = map_coordinates(jnp.asarray(source), coords, order=3, cval=2.3)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("mode", ["constant", "nearest", "mirror", "reflect", "wrap"])
def test_complex_samples_have_real_coordinates(mode):
    """Complex amplitudes interpolate componentwise, without complex indices."""
    source = jnp.array([1 + 2j, 3 - 4j, 5 + 6j])
    coords = [jnp.array([-0.4, 0.6, 1.3, 2.5])]
    actual = map_coordinates(source, coords, order=3, mode=mode)
    expected = map_coordinates(source.real, coords, order=3, mode=mode) + 1j * (
        map_coordinates(source.imag, coords, order=3, mode=mode)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_singleton_mirror_axis():
    """Mirror boundaries on a singleton axis must not divide indices by zero."""
    actual = map_coordinates(
        jnp.array([7.0]), [jnp.array([-1.4, 0.3, 4.0])], order=3, mode="mirror"
    )
    np.testing.assert_allclose(actual, 7, atol=1e-14)


def test_coordinate_precision_does_not_promote_image_output():
    """High precision coordinates must not double a float32 image's storage."""
    source = jnp.array([0.0, 10.0, 20.0], dtype=jnp.float32)
    actual = map_coordinates(source, [jnp.array([0.6], dtype=jnp.float64)], order=1)
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, [6.0])


def test_resampler_angle_gradient_at_zero():
    """Optimizing unrotated evaluation must preserve the infinitesimal rotation."""
    y, x = jnp.mgrid[:32, :32]
    source = jnp.exp(-((x - 19) ** 2 + (y - 13) ** 2) / 18)

    def moment(angle):
        return (
            resample_flux(source, 1.0, 1.3, (24, 24), angle) * jnp.arange(24)[:, None]
        ).sum()

    h = 1e-4
    expected = (moment(h) - moment(-h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(moment)(0.0), expected, rtol=1e-6)
