# hwoutils

Shared JAX-based utilities for the HWO direct imaging simulation suite.

## Overview

hwoutils provides common building blocks used across the HWO simulation
libraries (yippy, orbix, coronagraphoto, coronalyze):

- **JAX configuration** -- helpers to set precision, platform, and device
  count before JAX initializes
- **Constants** -- physical and mission constants
- **Conversions** -- unit and coordinate transforms
- **Grids and images** -- `PixelGrid`, `Image`, `Measure` and
  `SkyOrientation`, with conservative pixel-footprint integration
- **Spectral quadrature** -- `Band`, `Quadrature` and `spectral_breaks`
- **Image transforms** -- smooth resampling, rotation, and subpixel shifts
  via JAX
- **Radial profiles** -- azimuthal averaging for 2D images
- **Map coordinates** -- nearest, linear and Keys cubic interpolation on
  regular grids

## Spectral integration

Supply wavelength intervals by their endpoints in nm. Build the quadrature
once on the host, then evaluate each spectrum on the same array of nodes.
For example, the following integrates a piecewise-linear transmission curve
and returns its mean response in two bands.

```python
from hwoutils.spectral import Band, Quadrature, Transmission

bands = (Band(400, 550), Band(550, 700))
response = Transmission([400, 500, 700], [0, 1, 0])
rule = Quadrature.build(bands, response)
samples = response(rule.nodes_nm)
equivalent_widths_nm = rule.integrate(samples)
mean_transmissions = rule.mean(samples)
```

Models declare slope changes and jumps with a host method
`spectral_breaks_nm()` returning a one-dimensional NumPy array.
Smooth models return an empty array. Each band is split at all interior
breaks before applying Gauss-Legendre integration. The first two axes of
the samples index bands and nodes. Integration preserves all trailing axes.
`rule.band(i)` selects a single band while retaining the band axis.

`Quadrature.monochromatic(wavelength_nm)` represents an already integrated
rate at one wavelength. Its single node has unit weight. A mean density
is undefined for this zero-width channel.

## Installation

```bash
pip install hwoutils
```

## Guides

```{toctree}
:maxdepth: 2

jax_configuration
interpolation
testbudget
internal_refs
```

## API Reference

```{toctree}
:maxdepth: 2

autoapi/index
```
