# Interpolation in `hwoutils`

`hwoutils.map_coordinates` provides three interpolation orders:

| `order` | Kernel | Taps | True interpolant? | Partition of unity? |
|---|---|---|---|---|
| 0 | Nearest neighbor | 1 | yes | yes |
| 1 | Linear | 2 | yes | yes |
| 3 | Keys cubic convolution, `a = -0.5` | 4 | yes | yes |

Naming note: the integer `order` follows SciPy's convention loosely.
Orders 0 and 1 match SciPy exactly. Order 3 uses the **Keys cubic
convolution** kernel rather than a cubic B-spline interpolant. Both
are degree-3 (cubic) piecewise polynomials with `O(h^3)` accuracy, but
they are different kernels. See "Why Keys, not cubic B-spline" below.

## Keys cubic convolution

### The kernel

Keys' cubic convolution kernel with parameter `a = -0.5` (also known as
Catmull-Rom) is a piecewise cubic with compact support on `[-2, 2]`:

```
         | 1.5*|t|^3 - 2.5*|t|^2 + 1              if |t| <= 1
K(t) =  -| -0.5*|t|^3 + 2.5*|t|^2 - 4*|t| + 2     if 1 < |t| <= 2
         | 0                                      otherwise
```

Key values:

```
K(0)   =  1
K(0.5) =  0.5625
K(1)   =  0
K(1.5) = -0.0625
K(2)   =  0
```

### Properties that matter

**True interpolant.** `K(0) = 1` and `K(k) = 0` for every non-zero integer
`k`, so evaluating the convolution at an integer grid point returns the
sample value exactly. A uniform image resampled at its own grid spacing
comes back unchanged on interior pixels (where the stencil does not touch
the boundary).

**Partition of unity at integer spacing.** For any real offset `t in [0,
1)`, the four kernel weights evaluated at `{t + 1, t, t - 1, t - 2}` sum
to exactly `1`. This is what makes the kernel flux-preserving when the
output grid samples the input at integer spacing (see below).

**Reproduces constants and linear functions exactly** under translation,
and reproduces quadratics to `O(h^3)`. This matches the standard
cubic-convolution order of accuracy.

**Small negative lobes.** The outer piece dips to `K(1.5) = -0.0625`, so
non-negative inputs can produce slightly negative outputs near isolated
bright delta-like features. For smooth (PSF-convolved) inputs the negative
values are negligible. If a strictly non-negative output is required
downstream -- for example, before a Poisson draw -- clip with
`jnp.maximum(img, 0.0)`.

**No prefilter required.** Convolved directly with pixel values, Keys acts
as a true interpolant. This is why it is both cheap and JAX-friendly: there
is no IIR filter pass.

### 2D separable evaluation

In 2D we evaluate the kernel separately along each axis. For a target point
with continuous coordinates `(y, x)` in the source grid:

1. Let `iy = floor(y)`, `ix = floor(x)`.
2. Sample the source at the 4x4 stencil
   `{iy-1, iy, iy+1, iy+2} x {ix-1, ix, ix+1, ix+2}`.
3. Compute the four Keys weights along each axis at the offsets
   `{y - (iy-1), y - iy, y - (iy+1), y - (iy+2)}` (and likewise for `x`).
4. The output is the sum over the 16 stencil points of
   `W_y[i] * W_x[j] * source[iy-1+i, ix-1+j]`.

The 2D weight field `W_y[i] * W_x[j]` is itself a partition of unity because
the 1D weights each sum to 1.

### Boundary handling

All boundary modes supported by `map_coordinates` (`constant`, `nearest`,
`wrap`, `mirror`, `reflect`) apply to the Keys stencil in the usual way: the
per-axis index fixer determines what sample value appears at indices that
fall outside `[0, size-1]`. For the default `mode='constant'` with
`cval=0.0`, out-of-bounds samples contribute zero flux. Near the boundary
(within 2 pixels of an edge) the stencil reaches the padding and the
reproduction properties above hold only on the interior.

## Flux conservation in `resample_flux`

`resample_flux(f_src, pixscale_src, pixscale_tgt, shape_tgt, rotation_deg)`
conserves total flux when the target grid is an integer-ratio downsample of
a band-limited source. The procedure is:

1. Convert integrated flux per source pixel to surface brightness:
   `s_src = f_src / pixscale_src^2`.
2. Sample the surface brightness at each target pixel center using Keys
   interpolation. Call the result `s_tgt`.
3. Convert back to integrated flux per target pixel:
   `f_tgt = s_tgt * pixscale_tgt^2`.

Total flux conservation reduces to the statement that

```
sum_{target pixels p}  s_tgt(p) * pixscale_tgt^2
  = sum_{source pixels q}  s_src(q) * pixscale_src^2
```

With `scale = pixscale_tgt / pixscale_src` an integer, every target pixel
center lies on the source grid at integer multiples of the source pixel
spacing, so along each axis the per-axis Keys weights at any target point
sum to 1 across the contributing source indices. Summing `s_tgt` over
target pixels and re-ordering the sum gives each interior source pixel a
total weight of exactly `scale^2`, the number of target-pixel areas that
cover it. The `pixscale_tgt^2 / pixscale_src^2 = scale^2` factor cancels
that weight, returning total flux to `sum s_src * pixscale_src^2 = sum
f_src`.

### When it conserves flux exactly (modulo boundary)

Flux is conserved to `O(h^3)` or better when:

- The target-to-source scale ratio is an integer (2x, 3x, ...).
- The rotation is a multiple of 90 degrees.
- The source is band-limited relative to the target grid (its power above
  the target Nyquist is small). HWO PSFs and post-PSF scenes satisfy this.
- The flux that matters lies inside an interior strip 2 pixels from each
  edge of the source image, so the Keys 4-tap stencil does not reach the
  boundary.

In practice the test suite (`hwoutils/tests/test_transforms.py`) verifies
`<1%` relative flux error on compact Gaussians at 2x and 3x integer
downsampling.

### When flux is lost deliberately

When the affine map takes target pixels outside the source array, flux that
was genuinely outside the source footprint cannot be recovered. A uniform
`(64, 64)` image rotated 45 degrees, for instance, loses roughly the corner
triangles -- this is physically correct. The test suite expects
`0.6 < f_tgt / f_src < 1.0` for this case.

### When flux conservation is only approximate

- **Non-integer scale.** At scale 1.5, target centers land at half-integer
  offsets in the source grid, so the per-column sum of Keys weights is
  generally not exactly 1. Total flux error is small (`O(h^3)` for
  band-limited inputs) but non-zero.
- **Rotation by a non-right angle.** Same reason: target centers fall at
  generic offsets in the source grid.
- **Undersampled source.** If the source has content above the target
  Nyquist (sharp delta-like peaks on the source grid), aliasing will
  redistribute flux and the small Keys negative lobes can make it visible.
  A brief Gaussian smoothing of the source at `sigma ~ scale / 2` before
  resampling restores conservation.

### Why surface brightness, not flux, is the quantity interpolated

Interpolating the raw flux-per-pixel values and then summing would
double-count or under-count depending on the target-to-source area ratio.
By converting to surface brightness (flux per unit area) before
interpolation and back to flux after, the area factor enters only through
the explicit `pixscale_tgt^2` / `pixscale_src^2` rescaling, and the Keys
partition-of-unity property handles the rest.

## Why Keys, not cubic B-spline

The most common "order=3" interpolator in other libraries is the cubic
B-spline *interpolant*. That is a two-step procedure:

1. A prefilter pass (solving a tridiagonal IIR system) computes B-spline
   coefficients `c_k` such that the B-spline basis expansion
   `f(x) = sum_k c_k * B3(x - k)` reproduces the sample values at integer
   `x`.
2. An evaluation pass computes `f(x)` at arbitrary points.

Both Keys and the cubic B-spline interpolant are degree-3 piecewise
polynomials with `O(h^3)` accuracy. Keys is preferred here because:

- **No IIR prefilter.** Keys acts as an interpolant directly on pixel
  values. The cubic B-spline needs a tridiagonal solve per row and per
  column, which is awkward to jit and differentiate.
- **Compact 4-tap support.** Keys evaluates with a bounded 4x4 stencil in
  2D, no matter how large the input. The B-spline interpolant's prefilter
  is formally non-local.
- **Partition of unity at integer grid spacing.** Both kernels sum to 1 on
  integer grids, but Keys needs no prefilter to achieve it.

If you evaluate the cubic B-spline *basis* directly on pixel values without
the prefilter -- as an earlier version of this code did -- you get a
smoothing operator, not an interpolant. `B3(0) = 2/3`, so the output at an
integer grid point is `(2/3) * sample` along each axis, or `(2/3)^2 ~=
0.4444 * sample` in 2D. This is the bug that prompted moving to Keys.

## Pipeline integration notes

### Clip before Poisson draw

If a resampled image feeds a Poisson noise draw, clip negatives first:

```python
img = resample_flux(psf, pix_src, pix_tgt, shape_tgt)
img = jnp.maximum(img, 0.0)
# ... Poisson draw ...
```

The negatives come from Keys' `K(1.5) = -0.0625` lobe and are small
(`<= 6%` of the peak for delta-like inputs, smaller for smooth inputs).
They are harmless in noiseless flux-conservation accounting but would break
the Poisson sampler.

### Non-integer scale ratios

If the scale ratio is not an integer and you care about absolute flux
conservation at the `0.01%` level, either pick an integer ratio, or perform
a two-step resample (first to an integer-ratio common multiple, then an
integer downsample/upsample). For typical HWO science (1%-level flux
accuracy is usually enough), the Keys default handles non-integer ratios
within the error budget.

### Band-limit the source first

For a resample that must preserve the integral of a peaked feature exactly,
ensure the source has no power above the target Nyquist: a brief Gaussian
smoothing at `sigma ~ scale / 2` is usually sufficient. Smooth PSFs from
`coronagraphoto` already satisfy this.

### Rotation-only or shift-only cases

`rotation_deg = 0` and `pixscale_src = pixscale_tgt` with `shape_tgt =
f_src.shape` is the identity map on the interior. Boundary pixels within 2
of the edge are still attenuated by the `cval=0` padding reaching the
stencil; if that matters, apply `mode='nearest'` or `mode='reflect'`.

## API reference

See `hwoutils.map_coordinates.map_coordinates` and
`hwoutils.transforms.resample_flux` for full signatures. Quick check:

```python
from hwoutils.transforms import resample_flux
import jax.numpy as jnp

img = jnp.ones((64, 64))
out = resample_flux(img, 1.0, 2.0, (32, 32), rotation_deg=0.0)
assert jnp.isclose(out.sum(), img.sum(), rtol=1e-2)
```

## References

- **Keys, R.G. (1981).** *Cubic convolution interpolation for digital image
  processing.* IEEE Transactions on Acoustics, Speech, and Signal
  Processing, 29(6), 1153-1160.
