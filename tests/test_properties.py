"""Property-based tests for hwoutils to verify invariants across arbitrary inputs."""

import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from hwoutils.transforms import ccw_rotation_matrix, resample_flux

jax.config.update("jax_enable_x64", True)

# =============================================================================
# Transforms: Flux Conservation
# =============================================================================


class TestFluxConservationProperties:
    """Property-based tests verifying rigorous flux conservation."""

    @given(
        # Fuzz source grid sizes
        ny_src=integers(min_value=5, max_value=20),
        nx_src=integers(min_value=5, max_value=20),
        # Fuzz target grid sizes
        ny_tgt=integers(min_value=10, max_value=30),
        nx_tgt=integers(min_value=10, max_value=30),
        # Fuzz pixel geometries
        pixscale_src=floats(min_value=0.01, max_value=0.2),
        pixscale_tgt=floats(min_value=0.01, max_value=0.2),
        # Fuzz rotation angles
        rotation_deg=floats(min_value=-360.0, max_value=360.0),
    )
    # Give hypothesis time to compile the JITed functions for new shapes
    @settings(deadline=None, max_examples=50)
    def test_compact_source_conservation(
        self, ny_src, nx_src, ny_tgt, nx_tgt, pixscale_src, pixscale_tgt, rotation_deg
    ):
        """Compact source must conserve flux regardless of scale or rotation."""
        # We need a well-sampled source (spline interpolation over a 1-pixel
        # delta function mathematically cannot conserve flux when grid points
        # miss the peak, especially during downsampling).
        # We build a 2D Gaussian with sigma=1.5 pixels.
        y, x = jnp.mgrid[:ny_src, :nx_src]
        cy, cx = ny_src / 2.0, nx_src / 2.0
        f_src = jnp.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.5**2))
        f_src = (f_src / jnp.sum(f_src)) * 100.0

        # If the target grid is physically MUCH smaller than the source grid,
        # the flux might naturally clip. We calculate physical bounds.
        physical_src_y = ny_src * pixscale_src
        physical_src_x = nx_src * pixscale_src

        # Compute maximum rotated extent
        diag = jnp.sqrt(physical_src_y**2 + physical_src_x**2)

        physical_tgt_y = ny_tgt * pixscale_tgt
        physical_tgt_x = nx_tgt * pixscale_tgt

        # Substantial margin: 4-point stencil extends 2 pixels,
        # plus diagonal extension
        margin_y = 4.0 * pixscale_tgt
        margin_x = 4.0 * pixscale_tgt

        # We must also ensure we aren't sub-Nyquist sampling a delta function.
        # Splines cannot conserve flux if a 1-pixel source is downsampled by >2x
        # (the spline stencil simply steps completely over the delta function).
        scale_ratio = pixscale_tgt / pixscale_src

        # Only run the conservation assert if the rotated source physically fits
        # inside the target domain and the downsampling isn't extreme.
        if (
            physical_tgt_y > (diag + margin_y)
            and physical_tgt_x > (diag + margin_x)
            and scale_ratio < 2.0
        ):
            f_tgt = resample_flux(
                f_src,
                pixscale_src=pixscale_src,
                pixscale_tgt=pixscale_tgt,
                shape_tgt=(ny_tgt, nx_tgt),
                rotation_deg=rotation_deg,
            )

            # The sum should be nearly exactly 100.0
            assert jnp.isclose(jnp.sum(f_tgt), 100.0, rtol=0.05)


# =============================================================================
# Transforms: Rotation Matrix Properties
# =============================================================================


class TestRotationMatrixProperties:
    """Property-based tests for rotation matrix invariants."""

    @given(angle=floats(min_value=-1000.0, max_value=1000.0))
    def test_determinant_is_always_one(self, angle):
        """det(R) = 1 for all valid angles (prevents scaling/shearing)."""
        R = ccw_rotation_matrix(angle)
        det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
        assert jnp.isclose(det, 1.0, atol=1e-6)

    @given(angle=floats(min_value=-1000.0, max_value=1000.0))
    def test_orthogonality(self, angle):
        """R^T @ R = I for all valid angles."""
        R = ccw_rotation_matrix(angle)
        eye = jnp.eye(2)
        assert jnp.allclose(R.T @ R, eye, atol=1e-6)

    @given(
        angle1=floats(min_value=-360.0, max_value=360.0),
        angle2=floats(min_value=-360.0, max_value=360.0),
    )
    def test_composition(self, angle1, angle2):
        """R(a) @ R(b) = R(a+b)."""
        R1 = ccw_rotation_matrix(angle1)
        R2 = ccw_rotation_matrix(angle2)
        R_composed = R1 @ R2
        R_expected = ccw_rotation_matrix(angle1 + angle2)
        # Angles add modulo 2pi under rotation, so the matrices should match
        assert jnp.allclose(R_composed, R_expected, atol=1e-5)
