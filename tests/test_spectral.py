"""Band geometry and quadrature contracts, using small independent primitives."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.spectral import Band, Quadrature, bin_mean, integrate, spectral_knots


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


def test_spectral_knots_collects_nested_declarations_and_makes_kinks_exact():
    """Fails if a declaration form or a nested component is missed.

    Basis: exact algebra. A piecewise-linear spectrum with a kink at 537 nm is
    integrated exactly (to 64 float64 eps) by a two-node rule split at its
    knots; the unsplit rule over the same band misses the kink by far more.
    """
    from dataclasses import dataclass

    @dataclass
    class Table:
        wavelengths_nm: np.ndarray

    @dataclass
    class Hardware:
        def quadrature_knots_nm(self):
            return np.array([600.0])

    @dataclass
    class Molecule:
        nu_grid: np.ndarray

    @dataclass
    class System:
        star: Table
        parts: tuple

    system = System(
        star=Table(np.array([500.0, 537.0, 580.0])),
        parts=(Hardware(), Molecule(np.array([1e7 / 520.0])), None),
    )
    knots = spectral_knots(system, None, (np.nan,))
    np.testing.assert_allclose(knots, [500.0, 520.0, 537.0, 580.0, 600.0], rtol=1e-12)

    def spectrum(x):
        return np.interp(x, [500.0, 537.0, 580.0], [1.0, 3.0, 2.0])

    exact = 0.5 * (1.0 + 3.0) * 37.0 + 0.5 * (3.0 + 2.0) * 43.0
    with jax.enable_x64(True):
        for knots_nm, close in ((knots, True), (None, False)):
            q = Quadrature.build((Band(500.0, 580.0),), knots_nm, 2)
            value = float(q.integrate(spectrum(np.asarray(q.nodes_nm)))[0])
            if close:
                assert abs(value - exact) <= 64 * np.finfo(float).eps * exact
            else:
                assert abs(value - exact) > 1e-3 * exact
