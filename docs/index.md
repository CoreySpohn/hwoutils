# hwoutils

Shared JAX-based utilities for the HWO direct imaging simulation suite.

## Overview

hwoutils provides common building blocks used across the HWO simulation
libraries (yippy, orbix, coronagraphoto, coronalyze):

- **JAX configuration** -- helpers to set precision, platform, and device
  count before JAX initializes
- **Constants** -- physical and mission constants
- **Conversions** -- unit and coordinate transforms
- **Image transforms** -- flux-conserving resampling, downsampling, and
  subpixel interpolation via JAX
- **Radial profiles** -- azimuthal averaging for 2D images
- **Map coordinates** -- B-spline interpolation on regular grids

## Installation

```bash
pip install hwoutils
```

## Guides

```{toctree}
:maxdepth: 2

jax_configuration
```

## API Reference

```{toctree}
:maxdepth: 2

autoapi/index
```
