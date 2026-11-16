"""Band geometry and quadrature contracts, using small independent primitives."""

import sys
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.spectral import Band, Quadrature, bin_mean, integrate

reference_bin_mean = None
reference_spec = find_spec("skyscapes")
if reference_spec is not None and reference_spec.submodule_search_locations:
    # Load the independent spectral module without unrelated package backends.
    spectral_path = Path(reference_spec.submodule_search_locations[0]) / "spectral.py"
    if spectral_path.is_file():
        module_spec = spec_from_file_location("_spectral_reference", spectral_path)
        reference = module_from_spec(module_spec)
        sys.modules[module_spec.name] = reference
        try:
            module_spec.loader.exec_module(reference)
            reference_bin_mean = reference.bin_mean
        except ImportError:
            pass


@pytest.fixture(autouse=True, scope="module")
def precision():
    with jax.enable_x64():
        yield
    jax.clear_caches()


def test_band_partitions_share_exact_edges():
    """Fails if partitions have gaps or change the endpoints; no tolerance.

    Shared edges must be bitwise equal, even for logarithmic channels.
    """
    band = Band(410.1, 987.3)
    for spacing in ("linear", "log"):
        channels = band.channels(7, spacing)
        assert channels[0].lo_nm == band.lo_nm
        assert channels[-1].hi_nm == band.hi_nm
        for left, right in pairwise(channels):
            assert np.float64(left.hi_nm).tobytes() == np.float64(right.lo_nm).tobytes()
        assert all(c.width_nm > 0 for c in channels)
    assert band == Band(np.float64(410.1), 987.3)
    assert hash(band) == hash(Band(410.1, 987.3))
    for lo, hi in ((0, 1), (2, 1), (1, np.inf), (np.nan, 2)):
        with pytest.raises(ValueError):
            Band(lo, hi)
    with pytest.raises(ValueError):
        band.channels(0)
    with pytest.raises(ValueError):
        band.channels(3, "invalid")


def test_knot_quadrature_weights_and_piecewise_linear_integral():
    """Fails if kinks are unsplit or padding carries weight.

    Order one exactly integrates each linear segment. rtol=3e-15, atol=0 allows
    float64 node arithmetic and summation; weight sums allow 4 ULP of width.
    The primitive is independent: integral(2+x+3*max(x-2,0)).
    """
    bands = (Band(1, 6), Band(2, 3), Band(6, 7))
    knots = np.array([1.0, 2.0, 4.0, 6.0])
    widths = np.array([b.width_nm for b in bands])
    expected = np.array([51.5, 6.0, 22.0])
    for order in (1, 3):
        rule = Quadrature.build(bands, knots, order)
        np.testing.assert_array_max_ulp(
            np.asarray(rule.weights_nm.sum(-1)), widths, maxulp=4
        )
        padded = np.asarray(rule.weights_nm) == 0
        centers = np.broadcast_to(
            np.array([b.center_nm for b in bands])[:, None], padded.shape
        )
        np.testing.assert_array_equal(
            np.asarray(rule.nodes_nm)[padded], centers[padded]
        )
        values = 2 + rule.nodes_nm + 3 * jnp.maximum(rule.nodes_nm - 2, 0)
        np.testing.assert_allclose(rule.integrate(values), expected, rtol=3e-15, atol=0)
        vector = jnp.stack((values, -2 * values), axis=-1)
        result = jax.jit(lambda q, v: q.integrate(v))(rule, vector)
        np.testing.assert_allclose(
            result, expected[:, None] * [1, -2], rtol=3e-15, atol=0
        )
    centers = np.array([b.center_nm for b in bands])
    actual = integrate(
        lambda w: 2 + w + 3 * jnp.maximum(w - 2, 0),
        centers,
        widths,
        n_nodes=1,
        knots_nm=knots,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)


def test_smooth_integrand_converges_over_three_orders():
    """Fails if order is ignored or weights have the wrong normalization.

    Truth is exp(2)-exp(1); orders 1,2,4 integrate progressively more terms
    of the exponential series. Each error must fall by >20x, and order four
    is within 1e-9 relative (the degree-eight quadrature remainder).
    """
    errors = []
    exact = np.exp(2) - np.exp(1)
    for order in (1, 2, 4):
        rule = Quadrature.build((Band(1, 2),), (), order)
        value = rule.integrate(jnp.exp(rule.nodes_nm))[0]
        errors.append(abs(float(value) - exact))
    assert errors[1] < errors[0] / 20
    assert errors[2] < errors[1] / 20
    assert errors[2] / exact < 1e-9


def test_integral_matches_optional_bin_mean():
    """Fails if the port changes knot splitting or tensor reduction.

    rtol=3e-14 is a float64 reduction allowance for the integral/width
    conversion and independently compiled kernels, not a model tolerance.
    """
    if reference_bin_mean is None:
        pytest.skip("optional skyscapes.spectral is not importable")
    centers = np.array([1.3, 3.2, 5.0])
    widths = np.array([0.6, 3.0, 2.0])
    knots = np.array([0.0, 2.0, 4.0, 7.0])

    def fn(w):
        f = jnp.sin(w) + jnp.maximum(w - 2, 0)
        return jnp.array([f, f * w])

    options = dict(n_nodes=3, knots_nm=knots)
    expected = reference_bin_mean(fn, centers, widths, **options)
    actual = integrate(fn, centers, widths, **options)
    span = (centers + widths / 2) - (centers - widths / 2)
    np.testing.assert_allclose(actual / span[:, None], expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        bin_mean(fn, centers, widths, **options), expected, rtol=3e-14, atol=3e-14
    )


def test_weighted_point_bins_tracing_and_capacity_errors():
    """Fails if point bins divide by zero, weights drift, or caps truncate kinks.

    Weighted mean of a*x under weight x is a*(c+w^2/(12*c)). rtol=3e-15, atol=0
    covers float64 polynomial arithmetic, including the analytic derivative.
    """
    centers = jnp.array([2.0, 3.0, 2.0])
    widths = jnp.array([1.0, 2.0, 0.0])
    knots = np.array([0.0, 2.0, 4.0])

    @jax.jit
    def evaluate(a, c, w):
        return bin_mean(
            lambda x: a * x,
            c,
            w,
            n_nodes=2,
            knots_nm=knots,
            max_knots=2,
            weight_fn=lambda x: x,
        )

    expected = centers + widths**2 / (12 * centers)
    np.testing.assert_allclose(
        evaluate(1.7, centers, widths), 1.7 * expected, rtol=3e-15, atol=0
    )
    derivative = jax.grad(lambda a: evaluate(a, centers, widths).sum())(1.7)
    np.testing.assert_allclose(derivative, expected.sum(), rtol=3e-15, atol=0)
    np.testing.assert_array_equal(integrate(lambda x: x, centers, jnp.zeros(3)), 0)
    with pytest.raises(ValueError, match="too small"):
        bin_mean(
            lambda x: x, np.array([3.0]), np.array([4.0]), knots_nm=knots, max_knots=1
        )
    with pytest.raises(ValueError, match=r"Traced.*max_knots"):
        jax.jit(lambda c: bin_mean(lambda x: x, c, widths, knots_nm=knots))(centers)
    with pytest.raises(ValueError, match="widths"):
        bin_mean(lambda x: x, np.array([1.0]), np.array([-1.0]))
