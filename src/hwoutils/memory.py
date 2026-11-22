"""Size checks before a large allocation.

One rule for every library that builds a table or runs a map whose size scales with
the problem: estimate the bytes in closed form before allocating; if the work is a
map over independent items, run the largest batch that fits; if it is one object,
refuse. The limit is one keyword, ``max_bytes`` (``None`` disables the check), and
the refusal is one exception, :class:`MemoryLimit`. NumPy-free.
"""

__all__ = ["DEFAULT_MAX_BYTES", "MemoryLimit", "admit", "largest_batch"]

DEFAULT_MAX_BYTES = 2 * 2**30
"""Default ``max_bytes`` of the libraries that refuse large builds (2 GiB)."""


class MemoryLimit(ValueError):
    """An estimate exceeded ``max_bytes``.

    A refusal of an argument before anything is allocated, so it is a
    :class:`ValueError`. The message names the object, the estimate and the limit.
    """


def _gib(nbytes: float) -> str:
    return f"{nbytes / 2**30:.3g} GiB"


def admit(nbytes: int, max_bytes: int | None, what: str) -> None:
    """Refuse a build whose estimated size exceeds ``max_bytes``.

    Args:
        nbytes: Estimated bytes of the object, computed before allocating it.
        max_bytes: The limit, or ``None`` to admit everything.
        what: Name of the object, used in the message.

    Raises:
        MemoryLimit: If ``max_bytes`` is not ``None`` and ``nbytes > max_bytes``.
    """
    if max_bytes is not None and nbytes > max_bytes:
        raise MemoryLimit(
            f"{what} needs an estimated {_gib(nbytes)} ({nbytes} bytes), above "
            f"max_bytes={max_bytes} ({_gib(max_bytes)}); raise max_bytes or pass "
            "max_bytes=None to build it anyway"
        )


def largest_batch(
    n_items: int,
    bytes_per_item: int,
    max_bytes: int | None,
    *,
    fixed_bytes: int = 0,
) -> int:
    """Largest batch of independent items whose estimated size fits ``max_bytes``.

    The estimate of a batch of ``b`` items is ``fixed_bytes + b * bytes_per_item``.

    Args:
        n_items: Number of items to process, at least 1.
        bytes_per_item: Estimated bytes per item in flight.
        max_bytes: The limit, or ``None`` for all items at once.
        fixed_bytes: Bytes held regardless of the batch (tables, outputs).

    Returns:
        A batch size in ``[1, n_items]``.

    Raises:
        ValueError: If ``n_items`` is below 1 or a byte count is negative.
        MemoryLimit: If not even one item fits.
    """
    if n_items < 1:
        raise ValueError(f"n_items must be at least 1, got {n_items}")
    if bytes_per_item < 0 or fixed_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if max_bytes is None or bytes_per_item == 0:
        admit(fixed_bytes, max_bytes, "the fixed part of the batch")
        return n_items
    admit(fixed_bytes + bytes_per_item, max_bytes, "one item of the batch")
    return min(n_items, (max_bytes - fixed_bytes) // bytes_per_item)
