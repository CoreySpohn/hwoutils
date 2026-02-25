# JAX Configuration Guide

hwoutils provides three functions for configuring the JAX runtime.
All three are re-exported from the top-level `hwoutils` namespace.

```python
from hwoutils import enable_x64, set_platform, set_host_device_count
```

---

## The Golden Rule: Configure Before Computing

JAX initializes its backends on the **first JAX operation**, which can
happen as early as the first `import` that triggers array creation.
Configuration set after initialization is either silently ignored or
raises a `RuntimeError`.

```python
# CORRECT -- configure immediately after import
import jax
from hwoutils import enable_x64, set_platform

enable_x64()
set_platform("cpu")

import jax.numpy as jnp  # safe: config is already set
x = jnp.ones(10)         # uses float64 on CPU
```

```python
# WRONG -- too late, JAX already initialized
import jax.numpy as jnp
x = jnp.ones(10)          # triggers backend init with float32

from hwoutils import enable_x64
enable_x64()               # may be silently ignored!
```

> [!CAUTION]
> Library imports can trigger JAX initialization. If your project imports
> a JAX-based library (orbix, coronagraphoto, etc.) before calling
> `enable_x64()`, the flag may not take effect. Always configure JAX
> at the very top of your entry point.

### Environment variables (safest approach)

Setting environment variables guarantees the configuration is active
before Python even starts:

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python my_script.py
```

Or in a shell profile / CI config:

```bash
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

---

## `enable_x64`

Switches JAX from 32-bit (default) to 64-bit floating-point precision.

```python
from hwoutils import enable_x64

enable_x64()        # enable float64
enable_x64(False)   # revert to float32 (or read JAX_ENABLE_X64 env var)
```

### The `jax.enable_x64()` context manager (JAX >= 0.8.0)

As of JAX 0.8.0 (Oct 2025), there is a built-in context manager for
temporarily enabling 64-bit precision:

```python
import jax
import jax.numpy as jnp

# Default 32-bit
x = jnp.ones(3)  # float32

with jax.enable_x64():
    y = jnp.ones(3)  # float64 inside the block

z = jnp.ones(3)  # back to float32
```

This is useful for tests or isolated calculations that need double
precision without affecting the rest of the program.

> [!NOTE]
> The context manager replaces the deprecated
> `jax.experimental.enable_x64()`. If you see the experimental
> version in old code, update it to `jax.enable_x64()`.

### When to use which

| Scenario | Approach |
|---|---|
| Entire program needs float64 | `enable_x64()` at top of entry point |
| One test or function needs float64 | `with jax.enable_x64():` block |
| CI / batch jobs | `JAX_ENABLE_X64=True` env var |

---

## `set_platform`

Selects the compute backend (CPU, GPU, or TPU).

```python
from hwoutils import set_platform

set_platform("cpu")   # force CPU even if GPU is available
set_platform("gpu")   # use GPU
set_platform()        # read JAX_PLATFORMS env var, default "cpu"
```

> [!IMPORTANT]
> This uses the modern `jax_platforms` config key. Older code using
> `jax_platform_name` (singular) will trigger a deprecation warning
> in recent JAX versions.

### Common gotcha: "No GPU/TPU found"

If you call `set_platform("gpu")` but no GPU is available, JAX will
raise an error at the first computation rather than falling back to
CPU. To allow fallback:

```bash
# Allow GPU with CPU fallback
export JAX_PLATFORMS=gpu,cpu
```

Or in code:

```python
set_platform("gpu,cpu")
```

---

## `set_host_device_count`

Exposes multiple CPU cores as separate XLA devices, enabling
`jax.pmap` on CPU:

```python
from hwoutils import set_host_device_count

set_host_device_count(4)  # expose 4 CPU devices
```

> [!WARNING]
> This must be called **before any JAX operation**. It works by
> setting the `XLA_FLAGS` environment variable, which XLA reads
> only once during initialization.

---

## Quick Reference

```python
# Standard preamble for scripts in this workspace
import jax
from hwoutils import enable_x64, set_platform

enable_x64()
set_platform("cpu")

# Now safe to import JAX-based libraries and do work
import jax.numpy as jnp
from yippy import Coronagraph
from orbix import ...
```

### Debugging checklist

| Symptom | Likely cause |
|---|---|
| Arrays are float32 despite `enable_x64()` | Called too late (after first JAX op) |
| `RuntimeError: Couldn't initialize backend` | `set_platform("gpu")` with no GPU, use `"gpu,cpu"` |
| `jax.pmap` says only 1 device | `set_host_device_count` called after JAX init |
| `DeprecationWarning: jax_platform_name` | Old code using deprecated key, switch to `hwoutils.set_platform` |
| `DeprecationWarning: jax.experimental.enable_x64` | Replace with `jax.enable_x64()` context manager |
