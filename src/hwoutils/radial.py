"""Array-center helper shared by the image libraries."""


def get_center(shape: tuple[int, int]) -> tuple[float, float]:
    """Return the zero-indexed geometric center (cy, cx) of shape (ny, nx)."""
    return ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
