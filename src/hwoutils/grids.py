"""Static image geometry and arrays with explicit sampling metadata.

Arrays are (..., y, x); named coordinates are (x, y). Pixel (row j, column i)
is centered at (i, j), so the array corner is (-0.5, -0.5). Coordinates obey
u = B (p - c). No array or device value is stored in a PixelGrid.
"""

import math
from dataclasses import dataclass, fields
from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hwoutils.conversions import lambda_d_to_arcsec
from hwoutils.map_coordinates import map_coordinates
from hwoutils.radial import get_center


class _XY(NamedTuple):
    x: object
    y: object


@dataclass(frozen=True)
class Frame:
    """Definition of one lambda/D, including the declared diameter convention.

    The convention labels which physical diameter is supplied; no conversion
    between inscribed and circumscribed diameters is inferred.
    coordinate_model='paraxial_entrance_tilt' (the only supported model)
    identifies small entrance-pupil angular tilts, theta = s * lambda / D
    in radians for a coordinate s in lambda/D. No tangent or sine projection
    is applied. The default retains this linear angular conversion.
    """

    reference_wavelength_nm: float
    diameter_m: float
    diameter_convention: str
    coordinate_model: str = "paraxial_entrance_tilt"

    def __post_init__(self):
        """Store finite positive host floats and validate the diameter label."""
        for name in ("reference_wavelength_nm", "diameter_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.diameter_convention not in ("circumscribed", "inscribed"):
            raise ValueError("diameter_convention must be circumscribed or inscribed")
        if self.coordinate_model not in ("paraxial_entrance_tilt",):
            raise ValueError("coordinate_model must be 'paraxial_entrance_tilt'")

    def lod_to_arcsec(self, wavelength_nm):
        """Size of one lambda/D in arcsec; wavelength_nm may be traced."""
        return lambda_d_to_arcsec(1.0, wavelength_nm, self.diameter_m)


@dataclass(frozen=True)
class SkyOrientation:
    """Position angle of detector +y, north through east, and handedness.

    Let c=cos(pa_y_deg), s=sin(pa_y_deg). Detector +y has sky direction
    (E,N)=(s,c); the orthogonal +x direction is parity*(c,-s). Thus
    x=parity*(E*c-N*s), y=E*s+N*c, with determinant parity. For parity +1,
    x=E*c-N*s; for the inside-sphere parity -1, x=-E*c+N*s. The inverse
    is the transpose since both detector axes are orthonormal.
    """

    pa_y_deg: float
    parity: int = -1

    def __post_init__(self):
        """Canonicalize host values and reject undefined orientations."""
        object.__setattr__(self, "pa_y_deg", float(self.pa_y_deg))
        object.__setattr__(self, "parity", index(self.parity))
        if not math.isfinite(self.pa_y_deg) or self.parity not in (-1, 1):
            raise ValueError("orientation needs finite pa_y_deg and parity -1 or +1")

    def sky_to_detector(self, east, north):
        """Map east/north offsets to named detector (x, y) offsets."""
        c, s = (
            math.cos(math.radians(self.pa_y_deg)),
            math.sin(math.radians(self.pa_y_deg)),
        )
        east, north = jnp.asarray(east), jnp.asarray(north)
        return _XY(self.parity * (east * c - north * s), east * s + north * c)

    def detector_to_sky(self, x, y):
        """Return named (east, north) offsets in the sky plane."""
        c, s = (
            math.cos(math.radians(self.pa_y_deg)),
            math.sin(math.radians(self.pa_y_deg)),
        )
        x, y = jnp.asarray(x), jnp.asarray(y)
        return _Sky(self.parity * c * x + s * y, -self.parity * s * x + c * y)


class _Sky(NamedTuple):
    east: object
    north: object


@jax.tree_util.register_static
@dataclass(frozen=True)
class PixelGrid:
    """Hashable host geometry, with basis mapping pixel offsets to unit.

    shape is (ny, nx), center_x/center_y are pixel-center coordinates, and
    basis is a nonsingular 2 by 2 tuple. frame is mandatory for unit='lod'.
    orientation, when present, relates the unit plane to the east/north sky.
    """

    shape: tuple[int, int]
    center_x: float
    center_y: float
    basis: tuple[tuple[float, float], tuple[float, float]]
    unit: str
    frame: Frame | None = None
    orientation: SkyOrientation | None = None

    def __post_init__(self):
        """Canonicalize host geometry and validate units and invertibility."""
        shape = tuple(index(n) for n in self.shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("shape must contain positive (ny, nx)")
        basis = np.asarray(self.basis, dtype=np.float64)
        if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
            raise ValueError("basis must be finite with shape (2, 2)")
        determinant = basis[0, 0] * basis[1, 1] - basis[0, 1] * basis[1, 0]
        if not np.isfinite(determinant) or determinant == 0:
            raise ValueError("basis must be nonsingular with finite determinant")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "basis", tuple(tuple(map(float, row)) for row in basis)
        )
        for name in ("center_x", "center_y"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("centers must be finite")
            object.__setattr__(self, name, value)
        if self.unit not in ("pupil_diameter", "meter", "lod", "arcsec"):
            raise ValueError("unknown grid unit")
        if self.unit == "lod" and self.frame is None:
            raise ValueError("a lod grid requires a frame")
        if self.frame is not None and not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if self.orientation is not None and not isinstance(
            self.orientation, SkyOrientation
        ):
            raise TypeError("orientation must be a SkyOrientation")

    @classmethod
    def regular(cls, shape, step, unit, *, center=None, frame=None, orientation=None):
        """Make an axis-aligned isotropic grid; center defaults to geometric (x,y)."""
        if center is None:
            cy, cx = get_center(shape)
            center = (cx, cy)
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step must be positive and finite")
        return cls(
            shape,
            center[0],
            center[1],
            ((step, 0.0), (0.0, step)),
            unit,
            frame,
            orientation,
        )

    @classmethod
    def pupil(cls, npix):
        """Square grid spanning one pupil diameter, with geometric center."""
        npix = index(npix)
        if npix < 1:
            raise ValueError("npix must be positive")
        return cls.regular((npix, npix), 1 / npix, "pupil_diameter")

    @classmethod
    def focal(cls, npix, step_lod, frame):
        """Square focal grid in lambda/D, with geometric center."""
        return cls.regular((npix, npix), step_lod, "lod", frame=frame)

    @classmethod
    def from_corner_count(
        cls, shape, center_x, center_y, basis, unit, frame=None, orientation=None
    ):
        """Construct from corner counts, subtracting 0.5 from each center count.

        Arguments match PixelGrid; only center_x and center_y change meaning
        here. For example, a header count 128 becomes center coordinate 127.5.
        """
        return cls(
            shape,
            float(center_x) - 0.5,
            float(center_y) - 0.5,
            basis,
            unit,
            frame,
            orientation,
        )

    def coordinates(self):
        """Return named (x, y) arrays of shape (ny, nx), in this grid's unit."""
        rows, columns = jnp.indices(
            self.shape, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
        )
        return self.pixel_to_unit(columns, rows)

    def pixel_to_unit(self, x, y):
        """Map named pixel-center coordinates (x, y) to offsets in unit."""
        dx, dy = jnp.asarray(x) - self.center_x, jnp.asarray(y) - self.center_y
        (a, b), (c, d) = self.basis
        return _XY(a * dx + b * dy, c * dx + d * dy)

    def unit_to_pixel(self, x, y):
        """Invert the basis and center, returning named pixel coordinates (x,y)."""
        (a, b), (c, d) = self.basis
        det = a * d - b * c
        x, y = jnp.asarray(x), jnp.asarray(y)
        return _XY(
            (d * x - b * y) / det + self.center_x, (a * y - c * x) / det + self.center_y
        )

    @property
    def is_geometric(self) -> bool:
        """Whether the center is exactly ((nx-1)/2, (ny-1)/2), in pixels.

        Tolerance is zero: integer and half-integer centers are exactly
        representable, and even a small displacement breaks pixel-flip
        symmetry. This does not constrain the basis or orientation.
        """
        cy, cx = get_center(self.shape)
        return self.center_x == cx and self.center_y == cy

    @property
    def step(self) -> float:
        """Positive scalar spacing; raise for rotated or anisotropic bases."""
        (a, b), (c, d) = self.basis
        if b != 0 or c != 0 or a != d or a <= 0:
            raise ValueError("step requires a positive isotropic axis-aligned basis")
        return a

    @property
    def pixel_area(self) -> float:
        """Absolute basis determinant, in unit squared."""
        (a, b), (c, d) = self.basis
        return abs(a * d - b * c)

    def to_arcsec(self, wavelength_nm) -> "PixelGrid":
        """Convert a lod grid at a concrete wavelength to static arcsec geometry.

        The basis scales by lambda/D at wavelength_nm, preserving centers and
        retaining the frame. Static grid creation is a host operation; use
        Frame.lod_to_arcsec for traced wavelength calculations.
        """
        if self.unit != "lod":
            raise ValueError("to_arcsec requires a lod grid")
        wavelength_nm = float(wavelength_nm)
        if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive and finite")
        scale = self.frame.lod_to_arcsec(wavelength_nm)
        basis = tuple(tuple(value * scale for value in row) for row in self.basis)
        return PixelGrid(
            self.shape,
            self.center_x,
            self.center_y,
            basis,
            "arcsec",
            self.frame,
            self.orientation,
        )

    def is_native(self, wavelength_nm):
        """Whether this is a lod grid at its frame's reference wavelength."""
        return (
            self.unit == "lod" and wavelength_nm == self.frame.reference_wavelength_nm
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class GridArray:
    """Array (..., y, x) with a static grid and names for every leading axis.

    Construction validates shape and names. PyTree reconstruction bypasses
    construction checks, as JAX may insert placeholders or map away an axis.
    When returning a container with a changed rank, supply the new axis names.
    """

    data: jax.Array
    grid: PixelGrid
    leading_axes: tuple[str, ...]

    def __post_init__(self):
        """Convert data and validate spatial shape and leading-axis labels."""
        object.__setattr__(self, "data", jnp.asarray(self.data))
        object.__setattr__(self, "leading_axes", tuple(self.leading_axes))
        if self.data.ndim < 2 or self.data.shape[-2:] != self.grid.shape:
            raise ValueError("data spatial shape must match grid.shape")
        if len(self.leading_axes) != self.data.ndim - 2:
            raise ValueError("leading_axes count must match data leading dimensions")
        if any(not isinstance(name, str) or not name for name in self.leading_axes):
            raise ValueError("leading_axes must contain nonempty strings")

    def tree_flatten(self):
        """Expose data, retaining all other dataclass fields as static metadata."""
        metadata = tuple(getattr(self, field.name) for field in fields(self)[1:])
        return (self.data,), metadata

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Rebuild transformed containers without converting JAX placeholders."""
        result = object.__new__(cls)
        for field, value in zip(fields(cls), (arrays[0], *metadata), strict=True):
            object.__setattr__(result, field.name, value)
        return result


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Image(GridArray):
    """Image values with static measure, reconstruction, outside, and unit label.

    measure='per_pixel' means integrated pixel flux; 'per_unit_area' means
    density in the grid unit squared. 'point_density' explicitly declares
    density samples at pixel centers, never cell integrals or cell means.
    Both density measures resample by evaluating the chosen reconstruction
    at target centers, without area scaling or footprint averaging.
    quantity labels the stored data unit.
    reconstruction declares how samples define a continuous density: nearest,
    linear (default), Keys cubic, or band_limited. For band_limited per-pixel
    data, data / pixel_area are density samples, not deconvolved box averages.
    outside declares the extension beyond the array: zero (default) or reflect
    (even symmetry about the half-sample edges, repeated with period 2N).
    """

    measure: str
    quantity: str
    reconstruction: str = "linear"
    outside: str = "zero"

    def __post_init__(self):
        """Validate array geometry and the interpretation of image values."""
        super().__post_init__()
        if self.measure not in ("per_pixel", "per_unit_area", "point_density"):
            raise ValueError(
                "measure must be 'per_pixel', 'per_unit_area', or 'point_density'"
            )
        _validate_reconstruction(self.reconstruction)
        if self.outside not in ("zero", "reflect"):
            raise ValueError("outside must be 'zero' or 'reflect'")
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise ValueError("quantity must be a nonempty unit string")


def _validate_reconstruction(reconstruction):
    """Reject unknown reconstruction names before tracing numeric operations."""
    if reconstruction not in ("nearest", "linear", "cubic", "band_limited"):
        raise ValueError(
            "reconstruction must be nearest, linear, cubic, or band_limited"
        )


def _pixel_map(source, target):
    """Compose the static target-to-source linear map through the sky plane."""

    def orientation_matrix(orientation):
        angle = math.radians(orientation.pa_y_deg)
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[orientation.parity * c, -orientation.parity * s], [s, c]])

    transform = np.eye(2)
    if source.orientation != target.orientation:
        transform = (
            orientation_matrix(source.orientation)
            @ orientation_matrix(target.orientation).T
        )
    return np.linalg.solve(np.asarray(source.basis), transform @ target.basis)


def _band_limited_matrix(
    n_source, positions, width, pad_factor, dtype, integrate, outside
):
    """Build one target-by-source DFT interpolation or box-integration matrix."""
    positions = np.asarray(positions)
    samples = np.arange(n_source, dtype=dtype)
    if outside == "reflect":
        positions = positions.astype(dtype)
        # Pair opposite DFT modes of [d, d[::-1]]: its Nyquist mode is zero.
        frequencies = samples / np.asarray(2 * n_source, dtype=dtype)
        phase = np.asarray(2 * np.pi, dtype=dtype)
        evaluate = np.cos(phase * (positions[:, None] + 0.5) * frequencies)
        if integrate:
            evaluate *= np.sinc(frequencies * np.asarray(width, dtype=dtype))
        analyze = np.cos(phase * frequencies[:, None] * (samples + 0.5))
        analyze[0] *= 0.5
        return (evaluate @ analyze) * np.asarray(2 / n_source, dtype=dtype)
    n_padded = math.ceil(pad_factor * n_source)
    frequencies = np.fft.fftfreq(n_padded).astype(dtype)
    phase = np.asarray(2j * np.pi, dtype=np.result_type(dtype, np.complex64))
    if integrate:
        left = np.clip(positions - width / 2, -0.5, n_source - 0.5)
        right = np.clip(positions + width / 2, -0.5, n_source - 0.5)
        length = (right - left).astype(dtype)
        positions = ((left + right) / 2).astype(dtype)
        weights = (length / width)[:, None] * np.sinc(frequencies * length[:, None])
    else:
        weights = ((positions >= -0.5) & (positions <= n_source - 0.5))[:, None]
        positions = np.clip(positions, -0.5, n_source - 0.5).astype(dtype)
    evaluate = np.exp(phase * positions[:, None] * frequencies)
    evaluate *= weights
    analyze = np.exp(-phase * frequencies[:, None] * samples)
    return (evaluate @ analyze) / n_padded


def resample(
    image: Image,
    target_grid: PixelGrid,
    *,
    reconstruction: str | None = None,
    pad_factor: float = 2,
) -> Image:
    """Resample in one affine interpolation, respecting centers and full bases.

    Both grids must have the same unit (and equal frames for lod). Convert
    units explicitly first. If both orientations are supplied, coordinates
    pass through the sky plane; if only one is supplied, orientation is unknown
    and must be reconciled explicitly before resampling.

    reconstruction overrides the image's declared reconstruction for this
    operation only. nearest, linear, and cubic use the existing kernels.
    reconstruction and pad_factor must be static when supplied under jit.
    Per-unit-area and point-density values sample the source reconstruction
    at target pixel centers, extending indices according to image.outside: zero or
    half-sample reflection, d[-1-n]=d[n] and d[N+n]=d[N-1-n], periodically.

    For these three kernels with outside='zero', per-pixel flux uses two forms,
    chosen on the host from the pixel areas. When target pixels are smaller
    than source pixels (refinement) the flux is converted to a density, sampled at
    target centers and multiplied by the target pixel area, which is the
    model of ``transforms.resample_flux`` with explicit centers and bases:
    the result is smooth, and the sum is conserved up to sampling error
    (exactly for an interior source at an integer refinement). When the
    target pixels are as large or larger (coarsening, equal scale, or a
    rotation at equal scale) each source pixel's flux is deposited onto the
    target with the transpose of the interpolation kernel; its partition of
    unity conserves the sum to roundoff whenever the target contains every
    nonzero source pixel's stencil. Cropping loses flux in both forms.
    Neither is a pixel-footprint integral; use ``transforms.rebin_flux``
    for that on centered, axis-aligned grids.
    With outside='reflect', these kernels always sample the extended density
    and multiply by target pixel area, including when coarsening. Deposition
    of only the finite source would omit the reflected exterior. This center
    approximation is not an exact box integral or a flux-conservation guarantee.

    band_limited with outside='zero' uses the interpolant of a zero-padded array:
    data are assumed zero outside the source array. Each padded axis has
    P=ceil(pad_factor*N) samples (default 2; 1 means no padding). Padding
    controls zero-extension accuracy inside the footprint [-1/2,N-1/2] on
    each axis. The interpolant is restricted to that footprint and is exactly
    zero outside it, regardless of the padded period. Point samples at the
    boundary retain the interpolant value; box integrals clip at the boundary.
    With outside='reflect', use the even array [d, d[::-1]] of period P=2N;
    pad_factor applies only to zero extension and is ignored for reflection.
    Its paired DFT modes give cos(pi*k*(t+1/2)/N), k=0,...,N-1.
    The analysis weights are 1/N for DC and 2/N for the other cosine modes.
    These modes are even about both -1/2 and N-1/2 and periodic with 2N,
    so every exterior value is the repeated edge reflection, without an
    independent translated copy or a zero-padded gap.
    At an edge a, the smooth even extension f(a+u)=f(a-u) matches values.
    Its one-sided slopes have opposite signs, leaving a slope jump in general.
    Integrating Fourier coefficients by parts twice gives O(k^-2) edge-error
    amplitudes, versus O(k^-1) from a value jump under zero extension. These
    are spectral rates; a summed tail right at a slope corner can be O(K^-1).
    Both bases and the composed target-to-source map must be diagonal;
    rotation or shear raises NotImplementedError. Reflections are supported.

    Per-axis derivation, in source-pixel coordinates (sinc(z)=sin(pi*z)/(pi*z)):
    c_k = sum_n d_n exp(-2*pi*i*k*n/P) / P, with FFT-ordered k.
    A_jn = sum_k exp(2*pi*i*k*(t_j-n)/P) / P evaluates the density.
    For zero outside, intersect [t-w/2,t+w/2] with [-1/2,N-1/2] to get [a,b].
    Let ell=max(b-a,0), m=(a+b)/2; empty intersections contribute zero.
    Integral_a^b exp(2*pi*i*k*u/P) du = ell*exp(2*pi*i*k*m/P)*sinc(k*ell/P).
    Thus B_jn=sum_k (ell/w)*sinc(k*ell/P)*exp(2*pi*i*k*(m-n)/P)/P.
    For reflection use the full box: ell=w, m=t, and the cosine modes above.
    For per_unit_area and point_density return A_y @ data @ A_x.T at target centers.
    For per_pixel, density samples are data / source_pixel_area; integrating
    yields target_pixel_area/source_pixel_area * B_y @ data @ B_x.T.
    Real inputs use the cosine Nyquist convention separately on each even
    axis; complex inputs retain the FFT-ordered signed Nyquist mode.

    Matrices are built on the host from static geometry, in the input's real
    and complex precision, then applied with JAX; integer inputs are promoted
    to floating point. float32 has a rounding floor (typically 1e-5 for small
    grids, increasing with phase arguments and matrix size); use float64 for
    precision photometry. No positivity clamp or flux renormalization is used.
    Footprint clipping creates a value jump unless the interpolant vanishes
    at the edge: the restricted field is no longer globally band-limited.
    A well-sampled Gaussian with negligible edges makes this cost small; for
    sigma=0.8, center=(1,0), a 24x24 grid at 0.5 has its nearest edge 6.25 sigma
    away (analytic edge/peak <3.3e-9, omitted Gaussian mass <2.1e-10).
    With pad_factor=2 in float64, the reconstructed jump at (6,0) is about
    -5.15e-7 of peak; the clipped integral differs from the sample sum by
    3.58e-9 relative. float32 adds its rounding floor.
    This is not a bound for undersampled PSFs or bright/truncated wings:
    their edge jump and discarded interpolant tails can appreciably change
    flux. Total flux is the integral over the footprint, not in general the
    source sample sum. Coarsening aliases the density, so subsequent
    reconstruction need not conserve flux.

    The output always retains outside. It retains the source reconstruction,
    regardless of an override, except that a band_limited source becomes linear
    when the target Nyquist region cannot contain its source frequency box.
    If p_source=M*p_target+b,
    target frequencies are M.T*f_source; the condition is each column's
    absolute sum <= 1 (for a diagonal M, each absolute spacing ratio <= 1).
    Coarsening either axis fails this guarantee even if pixel area decreases.
    """
    source_grid = image.grid
    if source_grid.unit != target_grid.unit:
        raise ValueError("resampling requires matching grid units")
    if source_grid.unit == "lod" and source_grid.frame != target_grid.frame:
        raise ValueError("lod resampling requires equal frames")
    if (source_grid.orientation is None) != (target_grid.orientation is None):
        raise ValueError("resampling requires both orientations or neither")
    selected = image.reconstruction if reconstruction is None else reconstruction
    _validate_reconstruction(selected)
    mapping = _pixel_map(source_grid, target_grid)
    output_reconstruction = image.reconstruction
    if output_reconstruction == "band_limited" and np.any(
        np.sum(np.abs(mapping), axis=0) > 1 + 32 * np.finfo(float).eps
    ):
        output_reconstruction = "linear"
    metadata = (
        target_grid,
        image.leading_axes,
        image.measure,
        image.quantity,
        output_reconstruction,
        image.outside,
    )
    area_ratio = target_grid.pixel_area / source_grid.pixel_area
    data = jnp.asarray(image.data, dtype=jnp.result_type(image.data, 1.0))
    if selected == "band_limited":
        for matrix in (source_grid.basis, target_grid.basis, mapping):
            matrix = np.asarray(matrix)
            tolerance = 32 * np.finfo(float).eps * np.max(np.abs(matrix))
            if np.any(np.abs(matrix[[0, 1], [1, 0]]) > tolerance):
                raise NotImplementedError(
                    "band_limited reconstruction does not support rotated or sheared "
                    "grid bases or composed maps"
                )
        if image.outside == "zero":
            pad_factor = float(pad_factor)
            if not math.isfinite(pad_factor) or pad_factor < 1:
                raise ValueError("pad_factor must be finite and at least 1")
        matrices = []
        for axis, source_center, target_center in (
            (0, source_grid.center_x, target_grid.center_x),
            (1, source_grid.center_y, target_grid.center_y),
        ):
            ratio = mapping[axis, axis]
            positions = (
                np.arange(target_grid.shape[1 - axis]) - target_center
            ) * ratio + source_center
            matrix = _band_limited_matrix(
                source_grid.shape[1 - axis],
                positions,
                abs(ratio),
                pad_factor,
                data.real.dtype,
                image.measure == "per_pixel",
                image.outside,
            )
            # A real cosine choice on EACH axis avoids a mixed Nyquist term.
            if not jnp.iscomplexobj(data):
                matrix = matrix.real
            matrices.append(jnp.asarray(matrix, dtype=data.dtype))
        wx, wy = matrices
        data = wy @ data @ wx.T
        if image.measure == "per_pixel":
            data = data * area_ratio
        return Image.tree_unflatten(metadata, (data,))
    order = {"nearest": 0, "linear": 1, "cubic": 3}[selected]
    deposit_flux = (
        image.outside == "zero" and image.measure == "per_pixel" and area_ratio >= 1.0
    )
    scale = area_ratio if image.measure == "per_pixel" and not deposit_flux else 1.0
    # Sampling maps target centers into source pixels; deposition maps
    # source centers into target pixels and applies the kernel's transpose.
    from_grid, to_grid = (
        (source_grid, target_grid) if deposit_flux else (target_grid, source_grid)
    )
    x, y = from_grid.coordinates()
    if from_grid.orientation != to_grid.orientation:
        east, north = from_grid.orientation.detector_to_sky(x, y)
        x, y = to_grid.orientation.sky_to_detector(east, north)
    pixels = to_grid.unit_to_pixel(x, y)

    def interpolate(plane):
        if deposit_flux:

            def sample_target(target):
                return map_coordinates(target, (pixels.y, pixels.x), order=order)

            _, deposit = jax.vjp(
                sample_target, jnp.zeros(target_grid.shape, dtype=plane.dtype)
            )
            return deposit(plane)[0]
        return map_coordinates(
            plane,
            (pixels.y, pixels.x),
            order=order,
            mode="reflect" if image.outside == "reflect" else "constant",
        )

    for _ in range(data.ndim - 2):
        interpolate = jax.vmap(interpolate)
    data = interpolate(data) * scale
    return Image.tree_unflatten(metadata, (data,))
