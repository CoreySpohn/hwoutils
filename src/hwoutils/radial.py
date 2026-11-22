"""Array-center helper shared by the image libraries."""


def get_center(shape: tuple[int, int]) -> tuple[float, float]:
    """Return the zero-indexed geometric center (cy, cx) of shape (ny, nx).

    Args:
        shape: Image shape ``(ny, nx)`` in pixels.

    Returns:
        The center ``(cy, cx)`` in pixels, zero-indexed, so ``(ny - 1) / 2`` along
        the first axis and ``(nx - 1) / 2`` along the second.
    """
    return ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
