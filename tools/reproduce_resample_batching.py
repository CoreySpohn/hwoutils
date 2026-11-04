"""Compare scalar and batched resampling with identical, explicit inputs.

Run with PYTHONPATH=src. The source matches the compact-source property test:
14x14, center (7, 7), sigma 1.5 pixels, total flux 100, and >=4 sigma support.
"""

import jax
import jax.numpy as jnp

from hwoutils.transforms import resample_flux


def resample(source, scale, angle):
    """Keep only the target shape static; pass all numerical inputs explicitly."""
    return resample_flux(source, jnp.asarray(1.0, source.dtype), scale, (30, 30), angle)


def main():
    """Print all eight fluxes, output dtypes, and maximum pixel differences."""
    print(f"JAX {jax.__version__}; devices: {jax.devices()}")
    compiled = jax.jit(resample)
    batched = jax.vmap(resample)
    compiled_batch = jax.jit(batched)
    for dtype in (jnp.float64, jnp.float32):
        with jax.enable_x64(dtype == jnp.float64):
            y, x = jnp.mgrid[:14, :14].astype(dtype)
            source = jnp.exp(-((y - 7) ** 2 + (x - 7) ** 2) / (2 * 1.5**2))
            source = source / source.sum() * 100
            args = source, jnp.asarray(1.75, dtype), jnp.asarray(45.0, dtype)
            batch = tuple(value[None] for value in args)
            outputs = (
                resample(*args),
                compiled(*args),
                batched(*batch)[0],
                compiled_batch(*batch)[0],
            )
            for name, output in zip(
                ("scalar", "jit(scalar)", "vmap", "jit(vmap)"), outputs, strict=True
            ):
                output.block_until_ready()
                error = float(jnp.max(jnp.abs(output - outputs[0])))
                print(
                    f"{output.dtype} {name:12} {float(output.sum()):.12f} {error=:.3e}"
                )


if __name__ == "__main__":
    main()
