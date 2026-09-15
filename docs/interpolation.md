# Interpolation, rebinning, and rotation

Choose the operation according to what the samples mean:

| Operation | Input model | Flux behavior |
| --- | --- | --- |
| `map_coordinates` | Point samples reconstructed with nearest, linear, or Keys cubic interpolation | No pixel integration |
| `resample_flux` | Surface brightness sampled at target pixel centers, then scaled by target area | Approximate target-pixel flux; not strictly conservative |
| `rebin_flux` | Uniform surface brightness within each source pixel | Conserves covered source-pixel flux by integrating overlaps |
| `downsample_psf(s)` | Integrated source pixels, same field of view | Exact block sums for integer reductions; overlap integration otherwise |
| `rotate_image` | Periodic Fourier interpolation on a zero-padded grid | Finite-grid, Nyquist, and cropping errors require convergence checks |

## Keys cubic interpolation

For `order=3`, `map_coordinates` uses the four-tap Keys cubic convolution
kernel with a=-0.5 (Catmull-Rom), without a spline prefilter. It reproduces
samples at integer coordinates, including boundary pixels, and reproduces
constant, linear, and quadratic fields where the stencil is wholly inside
the input. The four weights are computed from the fractional coordinate's
square and cube and combined over the tensor-product stencil.

The weights sum to one at each target point. This is a condition for
reproducing constants, not for conserving the sum across a different target
grid. In particular, integer-ratio downsampling is not generally conservative:
a 3x center-sampling reduction can miss a single-pixel source entirely or
multiply its flux by nine. Smooth, adequately sampled PSFs usually have much
smaller errors, but their photometry must be checked for the chosen sampling.

Boundary modes are `constant`, `nearest`, `wrap`, `mirror`, and `reflect`.
`constant` substitutes `cval` for individual out-of-bounds stencil samples.
Coordinates are real and retain fractional values even for integer input.
Integer results are rounded back to the input dtype; cast the input to float
when fractional outputs are required. Complex samples are supported with real
coordinates. Floating and complex outputs retain the input dtype.

Cubic interpolation has negative lobes. Clipping negative values changes the
integral and should be an explicit downstream decision.

## Smooth resampling versus pixel integration

`resample_flux` centers both grids geometrically and performs inverse affine
mapping for scaling and rotation. Its output equals interpolation of the
source pixel values multiplied by `(pixscale_tgt / pixscale_src)**2`.
This is algebraically equivalent to conversion to surface brightness before
interpolation and back afterward. Neither formulation integrates the target
pixel footprint or suppresses aliasing automatically.

Use this operation when a smooth reconstructed image is needed, such as
aperture-boundary oversampling. Check convergence of the aperture measurement;
conserving a global sum alone does not establish correct local photometry.

Use `rebin_flux` for conservative axis-aligned changes of pixel grid:

```python
from hwoutils.transforms import rebin_flux

rebinned = rebin_flux(image, pixscale_src=1.0, pixscale_tgt=2.5, shape_tgt=(40, 40))
```

Each output is the sum of source-pixel flux multiplied by the fraction of each
source pixel overlapped by the target pixel. The grids share their geometric
center. Positive pixel scales are required. Outside the input footprint the
brightness is zero. If the output covers the whole input, all source fractions
sum to one; if it crops the input, uncovered flux is lost without renormalization.
This piecewise-constant reconstruction preserves positivity. It does not infer
sub-pixel optical structure, provide ideal low-pass filtering, or implement
rotated polygon overlaps.

`downsample_psf` and `downsample_psfs` use this model with fixed field of view.
They reject upsampling and aspect-ratio changes because a single returned
pixel scale cannot describe unequal axis scaling. For aligned integer ratios,
block summation avoids forming overlap matrices.

## Fourier rotation

`rotate_image(image, rotation_deg, pad_factor=0.5, center=None)` rotates
counter-clockwise when displayed with `origin="lower"`. The default center is
`((N-1)/2, (N-1)/2)`. FITS optical centers can differ, so pass `center=(row, column)`
explicitly when rotating around the optical axis.

The implementation applies right-angle array turns followed by three shears
with native FFT frequency ordering. Horizontal shears pair column frequencies
with row distances, and vertical shears pair row frequencies with column
distances. All shears share one padded image and only the final result is cropped.

For real input, each shear retains the real part of its inverse transform.
On even FFT grids this uses the real cosine interpolant of the Nyquist mode;
fractional shifts can attenuate that mode. Complex input retains both parts.
No clipping or flux renormalization is applied. This choice is explicitly tested
with an alternating-pixel pattern and is not a promise of exact unitarity.

`pad_factor` is the padding **per side** divided by the input width. The default
0.5 produces width 2N for even images and 2N-1 for odd images; 1.5 gives 4N or
4N-1. Larger padding remains available for precision wings and edge content.
A geometric buffer suppresses wrap-around but does not guarantee zero error
from finite periodic interpolation. Compare increasing padding for the science
measurement of interest. Translation helpers retain their existing 1.5 default.

The residual shear path executes even at zero angle, preserving the local angle
derivative at multiples of 90 degrees. Identity outputs therefore agree to FFT
roundoff rather than necessarily bitwise. At decomposition boundaries, finite
sampled images can differ between the two shear factorizations. JAX derivatives
are local to the chosen factorization. Rebinning similarly has piecewise scale
derivatives at pixel boundaries; JAX compatibility does not imply global smoothness.
