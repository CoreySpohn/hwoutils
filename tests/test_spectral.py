"""Band geometry and quadrature contracts, using small independent primitives."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hwoutils.grids import Image, Measure, PixelGrid
from hwoutils.spectral import (
    Band,
    Quadrature,
    SpectralBreaks,
    Transmission,
    _overlap_fractions,
    rebin_bands,
    spectral_breaks,
)


@pytest.fixture(autouse=True, scope="module")
def precision():
    with jax.enable_x64():
        yield
    jax.clear_caches()


@given(lo=st.floats(100, 599), hi=st.floats(600, 2000))
def test_band_operations_preserve_supplied_edges(lo, hi):
    """Fails if partitioning, building, slicing or integrating rounds an endpoint."""
    band = Band(lo, hi)
    for spacing in ("linear", "log"):
        channels = band.channels(7, spacing)
        rule = Quadrature.build(channels, order=1)
        assert rule.bands[0].lo_nm.hex() == lo.hex()
        assert rule.bands[-1].hi_nm.hex() == hi.hex()
        for left, right in pairwise(rule.bands):
            assert left.hi_nm.hex() == right.lo_nm.hex()
        for i, channel in enumerate(channels):
            row = rule.band(i)
            assert row.bands == (channel,)
            assert float(row.weights_nm[0, 0]).hex() == channel.width_nm.hex()
    with jax.enable_x64(False):
        # Host support must survive even when device evaluation uses float32.
        hat = Transmission.from_band(band)
        assert float(hat.low_nm).hex() == lo.hex()
        assert float(hat.high_nm).hex() == hi.hex()
    hat = Transmission.from_band(band)
    assert float(hat.low_nm).hex() == lo.hex()
    assert float(hat.high_nm).hex() == hi.hex()
    assert float(hat.integrate(band, order=1)).hex() == (hi - lo).hex()


def test_band_validates_geometry():
    """Fails if invalid edge order or spacing creates unusable quadrature geometry."""
    band = Band(410.1, 987.3)
    assert band == Band(np.float64(410.1), 987.3)
    assert hash(band) == hash(Band(410.1, 987.3))
    for lo, hi in ((0, 1), (2, 1), (1, np.inf), (np.nan, 2)):
        with pytest.raises(ValueError):
            Band(lo, hi)
    with pytest.raises(ValueError):
        band.channels(0)
    with pytest.raises(ValueError):
        band.channels(3, "invalid")


def test_split_quadrature_weights_and_piecewise_linear_integral():
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
        rule = Quadrature.build(bands, order=order, breaks_nm=knots)
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
        padded_values = jnp.where(padded, jnp.nan, values)
        np.testing.assert_allclose(
            rule.integrate(padded_values), expected, rtol=3e-15, atol=0
        )
        assert np.isnan(rule.integrate(values.at[0, 0].set(jnp.nan))[0])
        vector = jnp.stack((values, -2 * values), axis=-1)
        result = jax.jit(lambda q, v: q.integrate(v))(rule, vector)
        np.testing.assert_allclose(
            result, expected[:, None] * [1, -2], rtol=3e-15, atol=0
        )
        mean = rule.mean(vector)
        np.testing.assert_allclose(mean, result / widths[:, None], rtol=3e-15)
        row = rule.band(1)
        assert isinstance(row, Quadrature)
        assert row.bands == (bands[1],)
        np.testing.assert_array_equal(row.weights_nm, rule.weights_nm[1:2])
        np.testing.assert_array_equal(row.nodes_nm, rule.nodes_nm[1:2])
        np.testing.assert_allclose(row.integrate(vector[1:2]), result[1:2], rtol=3e-15)


def test_smooth_integrand_converges_over_three_orders():
    """Fails if order is ignored or weights have the wrong normalization.

    Truth is exp(2)-exp(1); orders 1,2,4 integrate progressively more terms
    of the exponential series. Each error must fall by >20x, and order four
    is within 1e-9 relative (the degree-eight quadrature remainder).
    """
    errors = []
    exact = np.exp(2) - np.exp(1)
    for order in (1, 2, 4):
        rule = Quadrature.build((Band(1, 2),), order=order)
        value = rule.integrate(jnp.exp(rule.nodes_nm))[0]
        errors.append(abs(float(value) - exact))
    assert errors[1] < errors[0] / 20
    assert errors[2] < errors[1] / 20
    assert errors[2] / exact < 1e-9


def test_monochromatic_selects_an_integrated_rate():
    """Fails if a monochromatic selection scales the supplied integrated rate."""
    rule = Quadrature.monochromatic(537.0)
    assert rule.bands == (Band(537.0, 537.0),)
    assert rule.bands[0].width_nm == 0
    np.testing.assert_array_equal(rule.band(0).nodes_nm, [[537.0]])
    np.testing.assert_array_equal(rule.weights_nm, [[1.0]])
    rates = jnp.arange(6.0).reshape((1, 1, 2, 3))
    np.testing.assert_array_equal(rule.integrate(rates), rates[:, 0])
    with pytest.raises(ValueError, match="monochromatic"):
        rule.mean(rates)
    with pytest.raises(ValueError, match="monochromatic"):
        Quadrature.build(rule.bands)
    with pytest.raises(ValueError, match="positive-width"):
        rebin_bands(rule.integrate(rates), rule.bands, rule.bands)


def test_polynomial_exactness_with_declared_breaks():
    """Fails if the order-four rule loses degree-seven exactness on any piece."""
    band = Band(1.0, 6.0)
    curve = Transmission([2, 4], [1, 1])
    rule = Quadrature.build((band,), curve)
    x = rule.nodes_nm
    values = jnp.where(
        x < 2, (x - 1) ** 7, jnp.where(x < 4, 3 * (x - 2) ** 7, 1 - (x - 4) ** 7)
    )
    exact = 1 / 8 + 3 * 2**8 / 8 + 2 - 2**8 / 8
    np.testing.assert_allclose(rule.integrate(values), [exact], rtol=4e-15, atol=0)


def test_spectral_breaks_require_declarations_and_split_one_node_rule_at_kink():
    """Fails if undeclared models are accepted or an interpolation kink is missed."""
    spectrum = Transmission([500, 537, 580], [1, 3, 2])
    overlapping = Transmission([537, 555], [1, 1])
    assert isinstance(spectrum, SpectralBreaks)
    np.testing.assert_array_equal(spectral_breaks(), [])
    np.testing.assert_array_equal(
        spectral_breaks(overlapping, spectrum), [500, 537, 555, 580]
    )

    class Undeclared:
        wavelengths_nm = (515, 530)

    with pytest.raises(TypeError, match="spectral_breaks_nm"):
        spectral_breaks(Undeclared())

    band = Band(500, 580)
    exact = (1 + 3) * 37 / 2 + (3 + 2) * 43 / 2
    split = Quadrature.build((band,), spectrum, order=1)
    np.testing.assert_allclose(
        split.integrate(spectrum(split.nodes_nm)), [exact], rtol=3e-15
    )
    unsplit = Quadrature.build((band,), order=1)
    assert abs(float(unsplit.integrate(spectrum(unsplit.nodes_nm))[0]) - exact) > 1


def test_rebin_conserves_total_flux():
    """Fails if overlap weights lose flux; float64 floor of 8 eps on exact tiling."""
    src_edges = Band(600.0, 700.0).channels(24, "log")
    dst_edges = Band(600.0, 700.0).channels(8, "log")
    q = Quadrature.build(src_edges, order=3)
    values = q.integrate(q.nodes_nm**2)
    exact = (700.0**3 - 600.0**3) / 3
    assert jnp.allclose(values.sum(), exact, rtol=2e-15, atol=0)
    out = rebin_bands(values, src_edges, dst_edges)
    assert out.shape == (8,)
    assert jnp.allclose(out.sum(), values.sum(), rtol=2e-15, atol=0)


def test_rebin_nested_grid_equals_reshape_sum():
    """On an exactly nested grid the overlap rebin reduces to reshape-sum."""
    edges = Band(600.0, 700.0).channels(8, "log")
    n = len(edges)
    edges_hi = tuple(sub for band in edges for sub in band.channels(4, "log"))
    vals = (jnp.arange(n * 4, dtype=float) + 1.0) ** 1.3
    out = rebin_bands(vals, edges_hi, edges)
    assert jnp.allclose(out, vals.reshape(n, 4).sum(axis=1), rtol=1e-10)


def test_rebin_along_cube_axis():
    """Rebinning applies along the chosen axis of a (n_wav, ny, nx) cube."""
    src_edges = Band(600.0, 700.0).channels(12, "log")
    dst_edges = Band(600.0, 700.0).channels(4, "log")
    cube = jax.random.uniform(jax.random.PRNGKey(0), (12, 3, 2))
    out = rebin_bands(cube, src_edges, dst_edges, axis=0)
    assert out.shape == (4, 3, 2)
    assert jnp.allclose(out.sum(axis=0), cube.sum(axis=0), rtol=2e-15, atol=0)


def test_rebin_partial_overlap_takes_fraction():
    """A destination bin covering half a source bin takes half its flux."""
    out = rebin_bands(jnp.array([2.0]), (Band(1.0, 2.0),), (Band(1.0, 1.5),))
    assert jnp.allclose(out, jnp.array([1.0]))


def test_overlap_fractions_irregular_intervals():
    """Fails if widths or empty cells are mishandled; exact binary fractions.

    Zero-width source intervals carry no flux.
    """
    fractions = _overlap_fractions(
        jnp.array([0.0, 2.0, 6.0]),
        jnp.array([2.0, 6.0, 6.0]),
        jnp.array([-1.0, 1.0, 6.0]),
        jnp.array([1.0, 4.0, 8.0]),
    )
    np.testing.assert_array_equal(
        fractions, [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
    )


def test_rebin_image_metadata_and_axes():
    """Fails if rebin changes metadata or spatial axes; exact sums of integers."""
    src = (Band(1.0, 2.0), Band(2.0, 3.0))
    dst = (Band(1.0, 3.0),)
    data = jnp.arange(24.0).reshape(3, 2, 2, 2)
    image = Image(
        data,
        PixelGrid.pupil(2),
        leading_axes=("sample", "band"),
        measure=Measure.integral(),
        quantity="photon",
        reconstruction="nearest",
        outside="reflect",
    )
    out = rebin_bands(image, src, dst, axis=-3)
    np.testing.assert_array_equal(out.data, np.asarray(data).sum(axis=1, keepdims=True))
    assert out.grid == image.grid
    assert out.leading_axes == image.leading_axes
    assert (out.measure, out.quantity, out.reconstruction, out.outside) == (
        image.measure,
        image.quantity,
        image.reconstruction,
        image.outside,
    )
    np.testing.assert_array_equal(
        rebin_bands(data, src, dst, axis=1),
        out.data,
    )
    with pytest.raises(ValueError, match="leading Image axis"):
        rebin_bands(image, src, dst, axis=-1)
    with pytest.raises(ValueError, match="source bands"):
        rebin_bands(data, src, dst, axis=0)
