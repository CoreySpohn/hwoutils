# hwoutils

Shared utilities for the HWO direct imaging simulation suite.

## Overview

hwoutils provides common building blocks used across the HWO simulation
libraries (yippy, orbix, coronagraphoto, photomancy):

- **JAX configuration** -- helpers to set precision, platform, and device
  count before JAX initializes
- **Constants** -- physical and mission constants (NumPy only)
- **Conversions** -- unit and coordinate transforms (NumPy only)
- **Epoch** -- the reference day zero for time offsets
- **Image transforms** -- flux-conserving resampling, downsampling, and
  subpixel interpolation via JAX
- **Array centers** -- the zero-indexed geometric center of an array
- **Map coordinates** -- B-spline interpolation on regular grids
- **Model paths and blocks** -- dotted paths into a model, and the contract of a
  free entry that owns a node of a model and writes it from its own coordinates

## Installation

```bash
pip install hwoutils          # constants, conversions, Epoch: NumPy only
pip install "hwoutils[jax]"   # adds the JAX modules
```

## Guides

```{toctree}
:maxdepth: 2

jax_configuration
interpolation
model_paths_and_blocks
testbudget
internal_refs
```

## API Reference

```{toctree}
:maxdepth: 2

autoapi/index
```
