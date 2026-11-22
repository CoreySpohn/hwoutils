"""Tests for the size checks in hwoutils.memory."""

import pytest

from hwoutils.memory import DEFAULT_MAX_BYTES, MemoryLimit, admit, largest_batch


def test_admit_refuses_above_the_limit_and_names_the_keyword():
    admit(DEFAULT_MAX_BYTES, DEFAULT_MAX_BYTES, "a table")
    with pytest.raises(MemoryLimit, match=r"needs an estimated 3 GiB.*max_bytes"):
        admit(3 * 2**30, DEFAULT_MAX_BYTES, "a table")


def test_admit_none_admits_everything_and_the_refusal_is_a_value_error():
    admit(10**15, None, "a table")
    assert issubclass(MemoryLimit, ValueError)


@pytest.mark.parametrize(
    ("n_items", "per_item", "max_bytes", "fixed", "expected"),
    [
        (100, 10, 1000, 0, 100),  # everything fits
        (100, 10, 255, 50, 20),  # floor((255 - 50) / 10)
        (100, 10, None, 0, 100),  # no limit
        (100, 0, 5, 0, 100),  # free items
        (3, 10, 10, 0, 1),  # exactly one fits
    ],
)
def test_largest_batch(n_items, per_item, max_bytes, fixed, expected):
    assert largest_batch(n_items, per_item, max_bytes, fixed_bytes=fixed) == expected


def test_largest_batch_refuses_when_one_item_does_not_fit():
    with pytest.raises(MemoryLimit, match="one item"):
        largest_batch(10, 100, 150, fixed_bytes=60)
    with pytest.raises(MemoryLimit, match="fixed part"):
        largest_batch(10, 0, 5, fixed_bytes=6)
    with pytest.raises(ValueError, match="n_items"):
        largest_batch(0, 1, 10)
