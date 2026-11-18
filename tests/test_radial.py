"""Radial bin membership and statistics on explicit image grids."""

import jax
import numpy as np
import pytest

from hwoutils.grids import Image, Measure, PixelGrid
from hwoutils.radial import radial_profile


def test_centered_mean_matches_previous_profile():
    """Fails if the default rings or the centered square mean change.

    Reference is the float64 output of the previous implementation on the
    seeded fixture. Tolerance allows float64 accumulation-order rounding.
    """
    with jax.enable_x64():
        data = np.random.default_rng(0).normal(size=(8, 8))
        image = Image(
            data,
            PixelGrid.regular(data.shape, 0.25, "arcsec"),
            (),
            Measure.integral(),
            "test",
        )
        centers, means, counts = radial_profile(image)
    np.testing.assert_allclose(
        centers,
        [
            0.15467960838455727,
            0.4640388251536718,
            0.7733980419227864,
            1.0827572586919008,
        ],
        atol=3e-16,
        rtol=0,
    )
    np.testing.assert_allclose(
        means,
        [
            -0.4194767755760441,
            0.36096114305232346,
            -0.10392928551632763,
            0.2265682987738254,
        ],
        atol=3e-16,
        rtol=0,
    )
    np.testing.assert_array_equal(counts, [4, 12, 28, 20])


@pytest.mark.parametrize(
    ("stat", "expected"),
    [
        ("mean", [6, np.nan, 23 / 3]),
        ("median", [6, np.nan, 4]),
        ("max", [10, np.nan, 20]),
    ],
)
def test_displaced_image_custom_bins(stat, expected):
    """Fails if bins lose the optical center, outer edge, or empty-bin NaN."""
    # x = [-2, -1, 0, 1], y = [0, 1]. The first ring contains
    # (x,y)=(0,0), (1,0), (-1,0), (0,1); the last has three members.
    data = [[4, 3, 10, 9], [99, 20, 2, -1]]
    grid = PixelGrid.regular((2, 4), 1, "pixel", center=(2, 0))
    image = Image(data, grid, (), Measure.integral(), "test")
    edges = np.array([0.0, 1.25, 1.375, 2.0])
    centers, values, counts = jax.jit(
        lambda im, e: radial_profile(im, stat=stat, edges=e)
    )(image, edges)
    np.testing.assert_allclose(centers, [0.625, 1.3125, 1.6875])
    np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(counts, [4, 0, 3])


def test_edges_exclude_samples_outside_both_ends():
    """Fails if underflow or overflow is folded into the first or last bin."""
    grid = PixelGrid.regular((1, 5), 1, "pixel", center=(0, 0))
    image = Image([[8, 4, 2, 16, 32]], grid, (), Measure.integral(), "test")
    _, values, counts = radial_profile(image, edges=[1, 2, 3], nbins=10)
    np.testing.assert_array_equal(values, [4, 9])
    np.testing.assert_array_equal(counts, [1, 2])
