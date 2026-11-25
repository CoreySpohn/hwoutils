# Leaves of a model

An Equinox model is a tree. Its leaves are what `jax.jit` traces, `jax.grad` differentiates and a fit
frees, so they must be arrays of one precision. `hwoutils.leaves` states that rule and gives the helpers
that keep a model to it. It needs the `jax` extra.

## The rule

Every leaf of an owned model is a `jax.Array`. Floating leaves have one strong working precision per model,
float32 or float64, with a matching complex dtype. Integer and boolean leaves keep their kind. A NumPy array, a NumPy
scalar or a Python number is never a leaf: a Python number lives on a `static=True` field, or it is converted.

## Convert on the field

Declare the conversion on the field, so every construction obeys the rule.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hwoutils.leaves import as_float_leaf, as_leaf, check_tree

jax.config.update("jax_enable_x64", True)


class Disk(eqx.Module):
    radius_au: jax.Array = eqx.field(converter=as_float_leaf)
    mask: jax.Array = eqx.field(converter=as_leaf)


disk = Disk(1.5, np.array([True, False]))
print(disk.radius_au.dtype, disk.mask.dtype)
print(check_tree(disk))
```

`as_float_leaf` keeps the dtype of a strong JAX float, including a tracer. Python numbers, NumPy
values, nested sequences, weak arrays and integers take the construction default (float64 with x64
enabled, float32 otherwise). `as_complex_leaf` applies the same rule with the matching complex dtype.
`as_leaf` keeps the numeric kind and checks that an integer fits JAX's integer width.

Declare a large real or complex table with `as_bulk_leaf` to preserve narrow storage. A float32 or
complex64 table stays narrow under x64. Its reader uses `hwoutils.like(table, owner)` to cast it to
the owner's working precision at use. Bulk fields do not determine that precision.

A concrete finite value that would become infinite in the target dtype is refused:

```python
import jax
import numpy as np

from hwoutils.leaves import as_float_leaf

with jax.enable_x64(False):
    try:
        as_float_leaf(np.array([1e300]))
    except ValueError as error:
        print(error)
```

A nonzero value that would underflow to zero (1e-50 under x32) and a Python integer beyond the float range
are refused the same way. A typed PRNG key is not a number: `as_leaf` and `as_float_leaf` raise `TypeError`
for one, so declare a key field without a converter (a key is already a valid leaf).

Tracers pass through unchecked.

## Check a model

`eqx.tree_at` bypasses converters, so a replacement made through it can break the rule. `check_tree` lists
the leaves that do, and `require_tree` raises on them. Call `require_tree` where a model enters a fit or an
exposure.

```python
import equinox as eqx
import numpy as np

from hwoutils.leaves import check_tree, require_tree


class Held(eqx.Module):
    value: object


model = Held(np.ones(2))
for violation in check_tree(model):
    print(violation.path, violation.reason)
try:
    require_tree(model)
except TypeError as error:
    print(str(error).splitlines()[0][:40])
```

A model can use float32 with x64 enabled. Both checks raise `hwoutils.PrecisionError` when non-bulk
leaves disagree on precision. Use `canonical(model, precision="float32")` or
`canonical(model, precision="float64")` from `hwoutils.leaves` to cast the entire model, including
bulk tables. Omitting `precision` chooses the construction default. Narrowing checks each real and
imaginary component for range loss. Widening float32 values to float64 is exact and requires x64.

The check flags NumPy leaves, Python numbers, weakly typed arrays, and unsupported working dtypes
(including float16 and bfloat16). Typed PRNG keys and integer and boolean arrays
are valid. `check_tree` enters only Modules owned by the workspace libraries; `require_tree` enters every
Module except those of third-party packages (the Equinox `nn` layers, Diffrax, Interpax), whose own fields are not
ours to constrain. The public `is_foreign(node, foreign=())` predicate identifies these opaque Modules.

## Three predicates

`is_data` selects what to slice, stack, copy or serialize, `is_param` what a fit may free, and `is_traced`
what host-only guards must skip. Nothing else should test array types.

## Static grids

Data that a host build depends on, such as a wavelength grid an operator is built on, is held as a
`HostGrid`: an immutable float64 host array, compared by a digest and stored on a static field. A fit
cannot free it, and nothing built from it can go stale.

```python
from hwoutils.leaves import HostGrid

grid = HostGrid([400.0, 500.0, 700.0])
print(grid, grid.array().dtype)
```

## The pytest plugin

A library suite can record every violation of the rule that its tests construct. Load the plugin from the
suite's `tests/conftest.py`:

```python notest
pytest_plugins = ["hwoutils.leaves_contract"]
```

The terminal summary lists each violating field once with the number of constructions, and
`--leaf-report PATH` writes the records as JSON. The plugin records and never raises, because a raise during
construction would be swallowed by a library `try` or matched by `pytest.raises`. Set the packages to check
with the `leaf_contract_packages` ini option; the default is the project name.
