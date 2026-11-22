# Size checks

`hwoutils.memory` holds the one rule the libraries follow before a large
allocation: estimate the bytes in closed form, then either run the largest batch
that fits or refuse. The limit is the keyword `max_bytes` (default
`DEFAULT_MAX_BYTES`, 2 GiB; `None` disables the check) and the refusal is
`MemoryLimit`, a `ValueError` raised before anything is allocated.

## Refuse one object

```python
from hwoutils.memory import DEFAULT_MAX_BYTES, MemoryLimit, admit

n_sources, n_pixels = 128 * 128, 256 * 256
nbytes = n_sources * n_pixels * 8  # a dense float64 PSF cube: 8.6 GB

try:
    admit(nbytes, DEFAULT_MAX_BYTES, "a dense PSF cube")
except MemoryLimit as err:
    print(err)
```

## Batch a map over independent items

```python
from hwoutils.memory import largest_batch

# 10 000 items of 4 MB each, a 100 MB table held throughout, 1 GiB allowed
batch = largest_batch(10_000, 4 * 2**20, 2**30, fixed_bytes=100 * 2**20)
print(batch)  # 231
```

A batch of `None` in a library call means all items at once; it is never chosen
behind the caller's back from an estimate.
