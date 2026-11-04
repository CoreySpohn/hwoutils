"""Resampling must preserve each batch member's geometry and pixel values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.transforms import resample_flux


@pytest.mark.parametrize("x64", [True, False])
def test_mixed_resampling_batch_matches_scalar(x64):
    # Catch per-call geometry being captured or broadcast across batch members.
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        y, x = jnp.mgrid[:14, :14].astype(dtype)
        source = jnp.exp(-((y - 7) ** 2 + (x - 7) ** 2) / (2 * 1.5**2))
        source = source / source.sum() * 100
        scales = jnp.asarray([1.75, 0.8, 1.25, 1.0], dtype)
        angles = jnp.asarray([45.0, 0.0, -30.0, 90.0], dtype)
        source_scales = jnp.asarray([0.125, 0.25, 0.5, 1.0], dtype)
        target_scales = source_scales * scales

        def resample(image, src_scale, tgt_scale, angle):
            return resample_flux(image, src_scale, tgt_scale, (30, 30), angle)

        args = list(zip(source_scales, target_scales, angles, strict=True))
        scalar = jnp.stack([resample(source, *arg) for arg in args])
        compiled = jax.jit(resample)
        jit_scalar = jnp.stack([compiled(source, *arg) for arg in args])
        rtol, atol = (1e-10, 1e-13) if x64 else (1e-5, 2e-5)
        np.testing.assert_allclose(jit_scalar, scalar, rtol=rtol, atol=atol)
        # Independent NumPy evaluation: sum the separable Keys basis over
        # every source pixel at inverse-rotated target centers, then scale
        # by pixel area. Store its results, not a second live implementation.
        np.testing.assert_allclose(
            scalar.sum(axis=(1, 2)),
            [98.35233398292655, 99.90533705906861, 99.94704197124867, 100.0],
            rtol=rtol,
            atol=0,
        )
        np.testing.assert_allclose(
            scalar[0, 14:16, 14:16],
            [[10.35654217154022] * 2, [17.800321723675186] * 2],
            rtol=rtol,
            atol=atol,
        )
        for image_axis in (None, 0):
            amplitudes = jnp.asarray([1.0, 2.0, 0.5, 1.25], dtype)[:, None, None]
            images = source if image_axis is None else source * amplitudes
            expected = scalar if image_axis is None else scalar * amplitudes
            batch_args = images, source_scales, target_scales, angles
            batched = jax.vmap(resample, in_axes=(image_axis, 0, 0, 0))
            for run in (batched, jax.jit(batched)):
                actual = run(*batch_args).block_until_ready()
                assert actual.dtype == dtype
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    actual.sum(axis=(1, 2)),
                    expected.sum(axis=(1, 2)),
                    rtol=rtol,
                    atol=0,
                )
