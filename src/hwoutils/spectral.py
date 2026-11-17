"""Static wavelength bands and knot-split Gauss-Legendre integration.

``integrate`` returns bin integrals in nm times the integrand's unit.
``bin_mean`` retains the scalar-function bin-mean interface, including optional
normalized weights. ``Quadrature.build`` prepares concrete geometry on the host;
its arrays and ``integrate`` method can then participate in JAX transformations.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import pairwise
from operator import index

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from hwoutils.conversions import wavenumber_cm_to_nm


@dataclass(frozen=True)
class Band:
    """Positive wavelength interval [nm], stored as hashable host floats."""

    lo_nm: float
    hi_nm: float

    def __post_init__(self):
        """Canonicalize concrete edges and reject empty or unbounded bands."""
        object.__setattr__(self, "lo_nm", float(self.lo_nm))
        object.__setattr__(self, "hi_nm", float(self.hi_nm))
        if not (0 < self.lo_nm < self.hi_nm < float("inf")):
            raise ValueError("Band edges must be finite and satisfy 0 < lo < hi")

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

    The arrays are dynamic PyTree leaves for jit, vmap and parameter gradients.
    Transformed tables must preserve the construction constraints. A photon
    spectrum multiplied by this response integrates detected photons; no
    energy-to-photon weighting or normalization is implicit.
    """

    wavelength_nm: jax.Array
    transmission: jax.Array

    def __post_init__(self):
        """Validate concrete tables and store arrays at configured precision."""
        wave = np.asarray(jnp.asarray(self.wavelength_nm, dtype=float))
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
        object.__setattr__(self, "wavelength_nm", jnp.asarray(wave, dtype=float))
        object.__setattr__(self, "transmission", jnp.asarray(response, dtype=float))

    @classmethod
    def tophat(cls, center_nm: float, width_nm: float) -> "Transmission":
        """Unit response over a positive full width [nm] about center_nm."""
        return cls([center_nm - width_nm / 2, center_nm + width_nm / 2], [1.0, 1.0])

    def __call__(self, wavelength_nm):
        """Evaluate at scalar or array wavelengths [nm], with zero extrapolation."""
        return jnp.interp(
            wavelength_nm, self.wavelength_nm, self.transmission, left=0.0, right=0.0
        )

    @property
    def quadrature_knots_nm(self):
        """All breakpoints [nm], including support endpoints for zero extension."""
        return self.wavelength_nm

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
        a, b = self.wavelength_nm[:-1], self.wavelength_nm[1:]
        ta, tb = self.transmission[:-1], self.transmission[1:]
        area = jnp.sum((b - a) * (ta + tb) / 2)
        moment = jnp.sum((b - a) * ((2 * a + b) * ta + (a + 2 * b) * tb) / 6)
        return moment / area

    def integrate(self, band: Band, fn=None, *, n_nodes: int = 4, knots_nm=None):
        """Integrate response * fn over a Band, in fn units times nm.

        fn defaults to one, giving equivalent width [nm]. Scalar-wavelength
        fn may return any fixed shape. Splits at all table knots and optional
        additional fn breakpoints. Order one integrates the response exactly;
        order two integrates its product with a linear spectrum exactly.
        Band and n_nodes are static under jit; tables may be traced.
        """
        knots = self.wavelength_nm
        if knots_nm is not None:
            knots = jnp.sort(jnp.concatenate((knots, jnp.asarray(knots_nm))))
        return integrate(
            self if fn is None else lambda w: self(w) * fn(w),
            jnp.array([band.center_nm]),
            jnp.array([band.width_nm]),
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=knots.size + 1,
        )[0]

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


def bin_mean(
    fn: Callable,
    centers_nm: jax.Array,
    widths_nm: jax.Array,
    *,
    n_nodes: int = 4,
    knots_nm: jax.Array | None = None,
    max_knots: int | None = None,
    weight_fn: Callable | None = None,
) -> jax.Array:
    """Mean of a scalar-wavelength function in bins, shape ``(W, ...)``.

    ``centers_nm`` and nonnegative ``widths_nm`` have shape ``(W,)``.
    ``fn`` returns any fixed shape; ``weight_fn``, if supplied, is scalar and
    gives integral(weight * fn) / integral(weight). Its denominator must be
    nonzero. Sorted 1D ``knots_nm`` split bins at strictly interior knots.
    ``n_nodes`` is the positive Gauss-Legendre order on each sub-interval.

    ``max_knots`` caps SUB-INTERVALS (interior knots plus one), despite its
    historical name. It is inferred on concrete geometry and required on
    traced geometry. Concrete undersized caps and negative widths raise;
    callers must supply valid widths and sufficient capacity under tracing.
    Selection uses bounded gathers, without a bins-by-knots matrix.

    Zero represented width returns fn(center), even with weight_fn. Inactive
    intervals evaluate at the center. Normalizing by the represented edge
    span retains narrow-bin accuracy. JIT, vmap and parameter gradients work
    with a static cap; knot membership changes are discrete.
    """
    n_nodes = index(n_nodes)
    if n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer.")
    concrete = not any(
        isinstance(x, jax.core.Tracer)
        for x in jax.tree.leaves((centers_nm, widths_nm, knots_nm))
    )
    if concrete and np.any(np.asarray(widths_nm) < 0):
        raise ValueError("Bin widths must be >= 0.")
    if knots_nm is not None:
        required = None
        if concrete:
            c, w, k = map(np.asarray, (centers_nm, widths_nm, knots_nm))
            counts = np.searchsorted(k, c + w / 2, side="left") - np.searchsorted(
                k, c - w / 2, side="right"
            )
            required = int(np.max(counts, initial=0)) + 1
        if max_knots is None:
            if required is None:
                raise ValueError("Traced bin geometry requires a static max_knots.")
            max_knots = required
        max_knots = index(max_knots)
        if max_knots < 1:
            raise ValueError("max_knots must be a positive sub-interval cap.")
        if required is not None and max_knots < required:
            raise ValueError(f"max_knots={max_knots} is too small; need {required}.")

    centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
    low, high = centers - widths / 2, centers + widths / 2
    if knots_nm is None or len(knots_nm) == 0:
        edges = jnp.stack((low, high), axis=1)
    else:
        knots = jnp.asarray(knots_nm)
        start = jnp.searchsorted(knots, low, side="right")
        offsets = start[:, None] + jnp.arange(max_knots - 1)
        interior = knots[jnp.minimum(offsets, knots.size - 1)]
        interior = jnp.where(offsets < knots.size, interior, high[:, None])
        interior = jnp.clip(interior, low[:, None], high[:, None])
        edges = jnp.concatenate((low[:, None], interior, high[:, None]), axis=1)

    lo, hi = edges[:, :-1], edges[:, 1:]
    span = high - low
    point_bin = span == 0
    lengths = (hi - lo) / jnp.where(point_bin, 1.0, span)[:, None]
    lengths = jnp.where(point_bin[:, None], 0.0, lengths)
    lengths = lengths.at[:, 0].set(jnp.where(point_bin, 1.0, lengths[:, 0]))
    nodes, weights = leggauss(n_nodes)
    wavelengths = lo[..., None] + (hi - lo)[..., None] * ((jnp.asarray(nodes) + 1) / 2)
    # Evaluate inactive intervals at the center, avoiding 0 * NaN at endpoints.
    wavelengths = jnp.where(
        ((hi == lo) | point_bin[:, None])[..., None],
        centers[:, None, None],
        wavelengths,
    )
    wavelengths = wavelengths.reshape((centers.size, -1))
    weights = (lengths[..., None] * jnp.asarray(weights) / 2).reshape(wavelengths.shape)
    values = jax.vmap(jax.vmap(fn))(wavelengths)
    if weight_fn is not None:
        throughput = jax.vmap(jax.vmap(weight_fn))(wavelengths)
        weights = weights * jnp.where(point_bin[:, None], 1.0, throughput)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    return jnp.einsum("wn,wn...->w...", weights, values)


def integrate(
    fn: Callable,
    centers_nm: jax.Array,
    widths_nm: jax.Array,
    *,
    n_nodes: int = 4,
    knots_nm: jax.Array | None = None,
    max_knots: int | None = None,
) -> jax.Array:
    """Integrate fn over each bin, shape ``(W, ...)``, with units fn times nm.

    Arguments follow ``bin_mean`` without normalized weighting. Multiply
    factors into fn for a weighted integral. Zero represented width gives zero.
    """
    means = bin_mean(
        fn,
        centers_nm,
        widths_nm,
        n_nodes=n_nodes,
        knots_nm=knots_nm,
        max_knots=max_knots,
    )
    centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
    span = (centers + widths / 2) - (centers - widths / 2)
    return jnp.einsum("w,w...->w...", span, means)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Quadrature:
    """Wavelength nodes and dimensional integration weights, both ``(W, Q)``.

    Build with x64 enabled for float64 weights; otherwise arrays follow JAX's
    configured precision. Padding has zero weight and sits at the band center.
    """

    nodes_nm: jax.Array
    weights_nm: jax.Array

    def __post_init__(self):
        """Convert array inputs and reject inconsistent quadrature shapes."""
        object.__setattr__(self, "nodes_nm", jnp.asarray(self.nodes_nm))
        object.__setattr__(self, "weights_nm", jnp.asarray(self.weights_nm))
        if self.nodes_nm.ndim != 2 or self.weights_nm.shape != self.nodes_nm.shape:
            raise ValueError("nodes_nm and weights_nm must share shape (W, Q)")

    @classmethod
    def build(cls, bands: Sequence[Band], knots_nm, order: int) -> "Quadrature":
        """Build host Gauss-Legendre rules split at unique interior knots.

        Use ``order`` nodes per nonempty sub-interval and pad to the largest
        node count Q. Padding nodes are at each band's arithmetic center.
        Weights sum to each band's width to the configured floating-point floor.
        """
        order = index(order)
        if order < 1:
            raise ValueError("order must be a positive integer")
        bands = tuple(bands)
        if not bands:
            raise ValueError("at least one band is required")
        knots = np.asarray(() if knots_nm is None else knots_nm, dtype=np.float64)
        if knots.ndim != 1 or not np.all(np.isfinite(knots)):
            raise ValueError("knots_nm must be a finite 1D array")
        knots = np.unique(knots)
        nodes, weights = leggauss(order)
        rules = []
        for band in bands:
            interior = knots[(knots > band.lo_nm) & (knots < band.hi_nm)]
            edges = np.concatenate(([band.lo_nm], interior, [band.hi_nm]))
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
        return cls(waves, weights)

    def integrate(self, values: jax.Array) -> jax.Array:
        """Sum values of shape ``(W, Q, ...)`` into integrals ``(W, ...)``."""
        values = jnp.asarray(values)
        if values.shape[:2] != self.nodes_nm.shape:
            raise ValueError("values must have leading shape (W, Q)")
        return jnp.einsum("wq,wq...->w...", self.weights_nm, values)

    def tree_flatten(self):
        """Expose just the two arrays to JAX."""
        return (self.nodes_nm, self.weights_nm), None

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Reconstruct transformed arrays without rerunning input validation."""
        result = object.__new__(cls)
        object.__setattr__(result, "nodes_nm", arrays[0])
        object.__setattr__(result, "weights_nm", arrays[1])
        return result


def spectral_knots(*models) -> np.ndarray:
    """Collect declared spectral breakpoints [nm] from models, on the host.

    A model declares breakpoints through ``quadrature_knots_nm`` (an array, or
    a method returning one), a tabulated ``wavelengths_nm``, or a wavenumber
    grid ``nu_grid`` [cm^-1]. Dataclass fields (Equinox modules are
    dataclasses) and tuples of them are searched recursively, so a system or an
    optical path contributes the tables of every component. Pass the result to
    ``Quadrature.build`` so that no Gauss-Legendre interval spans a kink of a
    tabulated spectrum. Call before tracing: the tables must be concrete.

    Returns:
        Sorted unique finite knots, shape ``(K,)``.
    """
    knots, seen = [], set()

    def visit(model):
        if model is None or id(model) in seen:
            return
        seen.add(id(model))
        if isinstance(model, (tuple, list)):
            for item in model:
                visit(item)
            return
        declared = getattr(model, "quadrature_knots_nm", None)
        if callable(declared):
            declared = declared()
        if declared is not None:
            knots.append(np.asarray(declared, dtype=float).ravel())
        wavelengths = getattr(model, "wavelengths_nm", None)
        if wavelengths is not None:
            knots.append(np.asarray(wavelengths, dtype=float).ravel())
        wavenumbers = getattr(model, "nu_grid", None)
        if wavenumbers is not None:
            converted = wavenumber_cm_to_nm(np.asarray(wavenumbers, dtype=float))
            knots.append(np.asarray(converted, dtype=float).ravel())
        if is_dataclass(model) and not isinstance(model, type):
            for field in fields(model):
                child = getattr(model, field.name, None)
                if is_dataclass(child) or isinstance(child, (tuple, list)):
                    visit(child)

    for model in models:
        visit(model)
    if not knots:
        return np.zeros(0)
    values = np.concatenate(knots)
    return np.unique(values[np.isfinite(values)])
