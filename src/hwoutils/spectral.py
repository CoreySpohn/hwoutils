"""Wavelength bands and Gauss-Legendre integration split at declared breaks.

Build quadrature on the host from band edges and concrete model declarations.
Evaluate spectra at its nodes and reduce their values with JAX transformations.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from operator import index
from typing import Protocol, Self, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from hwoutils.grids import Image

__all__ = [
    "Band",
    "Quadrature",
    "SpectralBreaks",
    "Transmission",
    "rebin_bands",
    "spectral_breaks",
    "trapezoid_weights",
]


@runtime_checkable
class SpectralBreaks(Protocol):
    """Model declaring wavelength breaks needed for piecewise integration."""

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return concrete break wavelengths [nm], or an empty 1D array."""
        ...


def _break_array(breaks_nm):
    breaks = np.asarray(breaks_nm, dtype=float)
    if breaks.ndim != 1 or not np.all(np.isfinite(breaks)):
        raise ValueError("spectral breaks must be a finite 1D array")
    return breaks


def spectral_breaks(*models: SpectralBreaks) -> np.ndarray:
    """Return the sorted union of model-declared spectral breaks [nm] on the host."""
    tables = []
    for model in models:
        if not isinstance(model, SpectralBreaks) or not callable(
            model.spectral_breaks_nm
        ):
            raise TypeError(
                f"{type(model).__name__} must implement spectral_breaks_nm()"
            )
        tables.append(_break_array(model.spectral_breaks_nm()))
    return np.unique(np.concatenate(tables)) if tables else np.empty(0)


@dataclass(frozen=True)
class Band:
    """Positive wavelength interval [nm], stored as hashable host floats."""

    lo_nm: float
    hi_nm: float

    def __post_init__(self):
        """Canonicalize concrete edges and reject reversed or unbounded bands."""
        object.__setattr__(self, "lo_nm", float(self.lo_nm))
        object.__setattr__(self, "hi_nm", float(self.hi_nm))
        if not (0 < self.lo_nm <= self.hi_nm < float("inf")):
            raise ValueError("Band edges must be finite and satisfy 0 < lo <= hi")

    @property
    def center_nm(self) -> float:
        """Arithmetic band center [nm]."""
        return self.lo_nm + self.width_nm / 2

    @property
    def geometric_center_nm(self) -> float:
        """Geometric band center sqrt(lo_nm * hi_nm) [nm]."""
        return math.sqrt(self.lo_nm * self.hi_nm)

    @property
    def width_nm(self) -> float:
        """Full band width [nm]."""
        return self.hi_nm - self.lo_nm

    def channels(self, n: int, spacing: str = "linear") -> tuple["Band", ...]:
        """Partition into n bands using one shared edge table, including endpoints."""
        n = index(n)
        if n < 1:
            raise ValueError("n must be a positive integer")
        if spacing == "linear":
            edges = np.linspace(self.lo_nm, self.hi_nm, n + 1)
        elif spacing == "log":
            edges = np.geomspace(self.lo_nm, self.hi_nm, n + 1)
        else:
            raise ValueError("spacing must be 'linear' or 'log'")
        edges[0], edges[-1] = self.lo_nm, self.hi_nm
        return tuple(Band(lo, hi) for lo, hi in pairwise(edges))


def _overlap_fractions(src_lo, src_hi, dst_lo, dst_hi):
    """Fraction of each source interval inside each destination interval.

    All bounds are 1D arrays in a common unit, with nonnegative source widths.
    The result has shape (n_dst, n_src); disjoint or empty intervals give zero.
    """
    src_lo, src_hi, dst_lo, dst_hi = map(jnp.asarray, (src_lo, src_hi, dst_lo, dst_hi))
    lo = jnp.maximum(dst_lo[:, None], src_lo[None])
    hi = jnp.minimum(dst_hi[:, None], src_hi[None])
    widths = src_hi - src_lo
    return jnp.maximum(hi - lo, 0) / jnp.where(widths > 0, widths, 1)[None]


def rebin_bands(values, src_bands, dst_bands, axis=0):
    """Rebin band integrals using exact Band overlaps, preserving units.

    Assumes constant spectral density within each source Band. Full destination
    coverage preserves the total integral to floating-point precision. Source
    and destination bands must have positive widths. An Image
    retains its spatial grid, measure, quantity and reconstruction; axis must
    name a leading (non-spatial) dimension in that case.
    """
    image = values if isinstance(values, Image) else None
    data = image.data if image is not None else jnp.asarray(values)
    axis = axis % data.ndim
    if image is not None and axis >= data.ndim - 2:
        raise ValueError("spectral axis must be a leading Image axis")
    if data.shape[axis] != len(src_bands):
        raise ValueError("source bands must match the spectral axis")
    if any(b.width_nm == 0 for b in (*src_bands, *dst_bands)):
        raise ValueError("overlap rebinning requires positive-width bands")
    src_lo = jnp.asarray([b.lo_nm for b in src_bands])
    src_hi = jnp.asarray([b.hi_nm for b in src_bands])
    dst_lo = jnp.asarray([b.lo_nm for b in dst_bands])
    dst_hi = jnp.asarray([b.hi_nm for b in dst_bands])
    frac = _overlap_fractions(src_lo, src_hi, dst_lo, dst_hi)
    out = jnp.moveaxis(
        jnp.tensordot(frac, jnp.moveaxis(data, axis, 0), axes=1), 0, axis
    )
    return replace(image, data=out) if image is not None else out


def trapezoid_weights(x):
    """Integration weights along the last axis of an increasing grid x.

    ``sum(weights * y, axis=-1)`` is the composite trapezoid integral, with
    half-width endpoint cells and centered interior widths. Leading axes
    are independent grids. A singleton has zero weight (zero integration
    span); an empty axis raises. Concrete inputs must be finite and strictly
    increasing; under JAX tracing the caller must maintain these conditions.
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, 1.0))
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty last axis")
    if not isinstance(x, jax.core.Tracer):
        values = np.asarray(x)
        if not np.all(np.isfinite(values)) or np.any(np.diff(values, axis=-1) <= 0):
            raise ValueError("x must be finite and strictly increasing")
    half = jnp.diff(x, axis=-1) / 2
    zero = jnp.zeros_like(x[..., :1])
    return jnp.concatenate((half, zero), axis=-1) + jnp.concatenate(
        (zero, half), axis=-1
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Transmission:
    """Tabulated dimensionless response, linear in wavelength, zero outside.

    Construct with concrete, matching 1D arrays of at least two samples:
    positive strictly increasing ``wavelength_nm`` and finite nonnegative
    ``transmission``. Values above one and identically zero responses are
    allowed (relative responses and opaque elements). Values at endpoints
    retain the tabulated response, with no extrapolated wings.

    The wavelength table retains host edge precision. Evaluation uses the
    configured JAX precision. Both arrays are dynamic PyTree leaves for
    jit, vmap and parameter gradients.
    Transformed tables must preserve the construction constraints. A photon
    spectrum multiplied by this response integrates detected photons; no
    energy-to-photon weighting or normalization is implicit.
    """

    wavelength_nm: jax.Array
    transmission: jax.Array

    def __post_init__(self):
        """Preserve host wavelength geometry and validate the response table."""
        wave = np.asarray(self.wavelength_nm, dtype=float)
        response = np.asarray(jnp.asarray(self.transmission, dtype=float))
        if wave.ndim != 1 or wave.size < 2 or response.shape != wave.shape:
            raise ValueError("tables must share shape (N,) with N >= 2")
        if (
            not np.all(np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(np.diff(wave) <= 0)
        ):
            raise ValueError("wavelength_nm must be finite, positive and increasing")
        if not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ValueError("transmission must be finite and nonnegative")
        # Retain the supplied break coordinates even when evaluations use float32.
        with jax.enable_x64():
            object.__setattr__(self, "wavelength_nm", jnp.asarray(wave))
        object.__setattr__(self, "transmission", jnp.asarray(response, dtype=float))

    @classmethod
    def from_band(cls, band: Band) -> "Transmission":
        """Unit response between the two supplied wavelength edges [nm]."""
        return cls([band.lo_nm, band.hi_nm], [1.0, 1.0])

    def __call__(self, wavelength_nm):
        """Evaluate at scalar or array wavelengths [nm], with zero extrapolation."""
        return jnp.interp(
            wavelength_nm,
            jnp.asarray(self.wavelength_nm, dtype=float),
            jnp.asarray(self.transmission, dtype=float),
            left=0.0,
            right=0.0,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """All breakpoints [nm], including support endpoints for zero extension."""
        return np.asarray(self.wavelength_nm)

    @property
    def low_nm(self):
        """Lower edge of the tabulated support [nm]."""
        return self.wavelength_nm[0]

    @property
    def high_nm(self):
        """Upper edge of the tabulated support [nm]."""
        return self.wavelength_nm[-1]

    @property
    def mean_nm(self):
        """Response-weighted mean wavelength [nm], NaN for an opaque table.

        Exact linear-segment area and quadratic first moment; this is the
        detected mean for a flat incident photon flux density.
        """
        wave = jnp.asarray(self.wavelength_nm, dtype=float)
        response = jnp.asarray(self.transmission, dtype=float)
        a, b = wave[:-1], wave[1:]
        ta, tb = response[:-1], response[1:]
        area = jnp.sum((b - a) * (ta + tb) / 2)
        moment = jnp.sum((b - a) * ((2 * a + b) * ta + (a + 2 * b) * tb) / 6)
        return moment / area

    def integrate(self, band: Band, fn=None, *, order: int = 4, breaks_nm=()):
        """Integrate response times fn over the supplied edges, in fn units times nm.

        Construct the rule on the host from this table and any additional
        breaks in fn. With no fn the result is the equivalent width [nm].
        A scalar-wavelength fn may return any fixed shape.
        """
        rule = Quadrature.build((band,), self, order=order, breaks_nm=breaks_nm)
        if fn is None:
            return rule.integrate(self(rule.nodes_nm))[0]
        values = jax.vmap(lambda w: self(w) * fn(w))(rule.nodes_nm[0])
        return rule.integrate(values[None])[0]

    def tree_flatten(self):
        """Expose both table arrays as dynamic leaves."""
        return (self.wavelength_nm, self.transmission), None

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Rebuild transformed tables without host validation."""
        result = object.__new__(cls)
        object.__setattr__(result, "wavelength_nm", arrays[0])
        object.__setattr__(result, "transmission", arrays[1])
        return result


class Quadrature(eqx.Module):
    """Host band geometry and JAX quadrature nodes and weights of shape (W, Q).

    Build with x64 enabled for float64 nodes and weights. Padding has zero
    weight and sits at the band center. A monochromatic channel carries one
    integrated rate with unit weight.
    """

    bands: tuple[Band, ...] = eqx.field(static=True, converter=tuple)
    nodes_nm: jax.Array = eqx.field(converter=jnp.asarray)
    weights_nm: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate the rectangular node table against its band geometry."""
        if self.nodes_nm.ndim != 2 or self.weights_nm.shape != self.nodes_nm.shape:
            raise ValueError("nodes_nm and weights_nm must share shape (W, Q)")
        if self.nodes_nm.shape[0] != len(self.bands) or not self.nodes_nm.shape[1]:
            raise ValueError("each band must have a nonempty row of nodes and weights")

    @classmethod
    def build(
        cls,
        bands: Sequence[Band],
        *models: SpectralBreaks,
        order: int = 4,
        breaks_nm=(),
    ) -> Self:
        """Split each band at its interior breaks and apply an order-point rule.

        Model declarations and optional explicit breaks are concrete host data.
        The dimensional weights on each row sum to the corresponding width [nm].
        """
        order = index(order)
        if order < 1:
            raise ValueError("order must be a positive integer")
        bands = tuple(bands)
        if not bands:
            raise ValueError("at least one band is required")
        breaks = np.union1d(spectral_breaks(*models), _break_array(breaks_nm))
        nodes, weights = leggauss(order)
        rules = []
        for band in bands:
            if band.width_nm == 0:
                raise ValueError("use Quadrature.monochromatic for a zero-width band")
            start = np.searchsorted(breaks, band.lo_nm, side="right")
            stop = np.searchsorted(breaks, band.hi_nm, side="left")
            edges = np.concatenate(([band.lo_nm], breaks[start:stop], [band.hi_nm]))
            lengths = np.diff(edges)
            wave = edges[:-1, None] + lengths[:, None] * ((nodes + 1) / 2)
            weight = lengths[:, None] * weights / 2
            rules.append((wave.ravel(), weight.ravel()))
        q = max(wave.size for wave, _ in rules)
        waves = np.array([[band.center_nm] * q for band in bands])
        weights = np.zeros_like(waves)
        for i, (wave, weight) in enumerate(rules):
            waves[i, : wave.size] = wave
            weights[i, : wave.size] = weight
        return cls(bands, waves, weights)

    @classmethod
    def monochromatic(cls, wavelength_nm: float) -> Self:
        """Select an already integrated rate at one wavelength, with unit weight."""
        band = Band(wavelength_nm, wavelength_nm)
        return cls((band,), [[band.lo_nm]], [[1.0]])

    def integrate(self, values: jax.Array) -> jax.Array:
        """Sum values of shape (W, Q, ...) into band integrals (W, ...)."""
        values = jnp.asarray(values)
        if values.shape[:2] != self.nodes_nm.shape:
            raise ValueError("values must have leading shape (W, Q)")
        active = (self.weights_nm != 0).reshape(
            self.weights_nm.shape + (1,) * (values.ndim - 2)
        )
        return jnp.einsum(
            "wq,wq...->w...", self.weights_nm, jnp.where(active, values, 0)
        )

    def mean(self, values: jax.Array) -> jax.Array:
        """Return the integral divided by the width of each positive-width band."""
        widths = tuple(band.width_nm for band in self.bands)
        if 0 in widths:
            raise ValueError("a monochromatic integrated rate has no band-width mean")
        integral = self.integrate(values)
        return integral / jnp.asarray(widths).reshape(
            (-1,) + (1,) * (integral.ndim - 1)
        )

    def band(self, i: int) -> Self:
        """Select a single band, preserving its row, padding, and wavelength axis."""
        return type(self)(
            (self.bands[i],), self.nodes_nm[i][None], self.weights_nm[i][None]
        )
