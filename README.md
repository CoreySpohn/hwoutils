<p align="center">
  <a href="https://pypi.org/project/hwoutils/"><img src="https://img.shields.io/pypi/v/hwoutils.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://hwoutils.readthedocs.io"><img src="https://readthedocs.org/projects/hwoutils/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/hwoutils/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/hwoutils/tests.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/hwoutils/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/CoreySpohn/hwoutils"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg?style=flat-square&logo=python" alt="Python Versions"></a>
</p>

---

# hwoutils

**hwoutils** is the shared utility foundation for the HWO direct imaging simulation suite. It provides JAX-native physical constants, unit conversions, and flux-conserving image transforms used across:

- **[yippy](https://github.com/CoreySpohn/yippy)** — Coronagraph performance modeling
- **[orbix](https://github.com/CoreySpohn/orbix)** — Orbital dynamics and target scheduling
- **[coronagraphoto](https://github.com/CoreySpohn/coronagraphoto)** — Image simulation for coronagraphic observations
- **[coronalyze](https://github.com/CoreySpohn/coronalyze)** — Post-processing and SNR analysis
- **[hwosim](https://github.com/CoreySpohn/hwosim)** — End-to-end mission simulations

## Key Features

- **Physical Constants** — Single source of truth for SI constants, conversion factors, and astronomical quantities
- **Unit Conversions** — Pure JAX conversion functions (angular, flux, distance, time) with zero astropy overhead
- **Image Transforms** — Flux-conserving resampling, sub-pixel shifts, and cubic spline interpolation
- **JAX-Native** — All operations are JIT-compilable, differentiable, and GPU-accelerated

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install hwoutils
```

Or with pip:

```bash
pip install hwoutils
```

## Quick Start

```python
from hwoutils import constants as const
from hwoutils import conversions as conv
from hwoutils.transforms import resample_flux

# Convert 5 arcsec to lambda/D for a 6m telescope at 550nm
sep_lod = conv.arcsec_to_lambda_d(5.0, 550.0, 6.0)

# Flux-conserving PSF resampling
resampled = resample_flux(psf, pixscale_src=0.01, pixscale_tgt=0.1, shape_tgt=(64, 64))
```

## Documentation

Full documentation is available at [hwoutils.readthedocs.io](https://hwoutils.readthedocs.io).
