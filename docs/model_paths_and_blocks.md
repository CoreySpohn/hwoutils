# Model paths and blocks

A fit in the simulation suite frees some quantities of a model and holds the rest. Two
small modules give the libraries one vocabulary for that: `hwoutils.model_paths` names a
node of a model by a string, and `hwoutils.blocks` states what a free entry that owns a
node must provide. Both need the `jax` extra.

## A path names a node

A tuple or list member with a unique string `name` can be reached by that name.
Names are case-sensitive. They start with an ASCII letter or underscore and
can then contain ASCII letters, digits, underscores and hyphens. Keys have
no quoting or escaping. Use an index for an empty, duplicated or otherwise
unaddressable name. A key consisting entirely of digits is always an index.
Keys chain from left to right, as in `groups[inner][2]`.

```python
import equinox as eqx
import jax
from hwoutils.leaves import as_float_leaf
from hwoutils.model_paths import (
    assign_many,
    canonical_path,
    display_path,
    resolve,
    scale_at,
)


class Shift(eqx.Module):
    shift: jax.Array = eqx.field(converter=as_float_leaf)


class Stage(eqx.Module):
    name: str = eqx.field(static=True)
    op: Shift


class OpticalPath(eqx.Module):
    stages: tuple[Stage, ...]


optics = OpticalPath(
    (
        Stage("entrance", Shift(1.0)),
        Stage("mask", Shift(2.0)),
        Stage("stop", Shift(3.0)),
    )
)
path = "stages[mask].op.shift"
assert float(resolve(optics, path)) == 2.0
assert canonical_path(optics, path) == "stages[1].op.shift"
assert display_path(optics, "stages[01].op.shift") == path

scaled = scale_at(optics, path, 2.0)
assert float(resolve(scaled, path)) == 4.0
changed = assign_many(scaled, {"stages[0].op.shift": 5.0, path: 6.0}, convert=True)
assert float(resolve(changed, "stages[entrance].op.shift")) == 5.0
assert float(resolve(changed, path)) == 6.0

try:
    assign_many(optics, {path: 5.0, "stages[1].op.shift": 6.0})
except ValueError as error:
    assert path in str(error) and "stages[1].op.shift" in str(error)
else:
    raise AssertionError("both paths select the same node")
```

| Function | Use |
|---|---|
| `canonical_path` | Resolve names and numeric aliases to plain indices before comparing paths. |
| `display_path` | Show unique names where the model can resolve them. |
| `assign` | Replace one node, with optional field conversion. Use `None` to clear an optional slot. |
| `assign_many` | Replace disjoint nodes together using the original model for all lookups. |
| `apply_at` | Read a node, call a function with it, and assign the result. |
| `scale_at` | Multiply an inexact JAX array leaf while keeping its precision. |

A model is an Equinox module, often nested. Here a star and a tuple of planets stand in for
the scene of a real fit.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from hwoutils.model_paths import assign, join, paths_overlap, resolve

jax.config.update("jax_enable_x64", True)


class Star(eqx.Module):
    mass_Msun: jax.Array


class Planet(eqx.Module):
    radius_Rearth: jax.Array


class Scene(eqx.Module):
    star: Star
    planets: tuple[Planet, ...]


scene = Scene(
    Star(jnp.asarray(1.0)),
    (Planet(jnp.asarray(1.0)), Planet(jnp.asarray(2.5))),
)
print(resolve(scene, "planets[1].radius_Rearth"))
```

A path is a dotted chain of attribute names, and `[k]` indexes a tuple or a list (never an
array axis). The empty path `""` names the model itself. `resolve` reads the node at a
path and `assign` returns a copy of the model with that node replaced.

```python continuation
bigger = assign(scene, "planets[1].radius_Rearth", jnp.asarray(3.0))
print(
    resolve(bigger, "planets[1].radius_Rearth"),
    resolve(scene, "planets[1].radius_Rearth"),
)
```

`join` builds a path from a prefix and a path relative to it, so that code written for a
star at the root also works on a star nested inside a larger model. `paths_overlap` says
whether two paths name the same node or one contains the other.

```python continuation
print(
    join("", "star.mass_Msun"), join("scene", "star.mass_Msun"), join("planets", "[1]")
)
print(paths_overlap("planets", "planets[1].radius_Rearth"))
print(paths_overlap("star", "planets"))
```

## A block owns a node

A free entry is either a law over one array leaf or a block. A block owns a node of the
model, carries its own real coordinates, and writes the node from them. It declares the
paths it owns, reads and writes, names its coordinates, and gives a normalized density
over them. The block below frees one positive leaf through its logarithm, with a Gaussian
density on the log.

```python continuation
from jax.scipy.stats import norm

from hwoutils.blocks import AbstractBlock, block_order, check_writers


class LogNormalBlock(AbstractBlock):
    """One positive scalar leaf, freed through its log with a Gaussian density."""

    owns: str = eqx.field(static=True)
    mean: float
    sd: float
    reads: tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def writes(self):
        return (self.owns,)

    @property
    def coordinate_names(self):
        return ("ln_value",)

    def forward(self, x, model):
        return {self.owns: jnp.exp(x["ln_value"])}

    def inverse(self, model):
        value = resolve(model, self.owns)
        return {"ln_value": jnp.where(value > 0, jnp.log(value), jnp.nan)}

    def log_prob(self, x, model):
        return norm.logpdf(x["ln_value"], self.mean, self.sd)

    def sample(self, key, model):
        return {"ln_value": self.mean + self.sd * jax.random.normal(key)}

    def same_layout(self, other):
        return type(other) is type(self) and other.owns == self.owns

    def same_map(self, other):
        return self.same_layout(other)


block = LogNormalBlock("planets[1].radius_Rearth", mean=0.9, sd=0.3)
x = block.sample(jax.random.key(0), scene)
written = block.forward(x, scene)
print(x, written, block.log_prob(x, scene))
```

The coordinates are a dictionary of arrays keyed by `coordinate_names`, and every
coordinate takes any real value, so a sampler that moves them never meets a boundary.
`forward` returns one value per written path, and `inverse` goes the other way: it gives
the coordinates that rebuild the node of a model, NaN where the model's value lies outside
the support.

```python continuation
print(block.inverse(scene))
print(block.inverse(assign(scene, "planets[1].radius_Rearth", jnp.asarray(-1.0))))
```

## Several blocks in one fit

A host that frees several blocks needs two guarantees: no path has two writers, and a
block that reads a path runs after the block that writes it. `check_writers` refuses the
first failure and `block_order` returns an order that satisfies the second. Below, a block
that reads the star mass is listed before the block that writes it, and `block_order` puts
the writer first. Supply `model=` to compare name keys and numeric aliases
against the same model.

```python continuation
mass = LogNormalBlock("star.mass_Msun", mean=0.0, sd=0.1)
radius = LogNormalBlock(
    "planets[0].radius_Rearth", mean=0.0, sd=0.5, reads=("star.mass_Msun",)
)
blocks = {radius.owns: radius, mass.owns: mass}

check_writers(blocks)
print(block_order(blocks))

try:
    check_writers({mass.owns: mass}, leaf_paths=("star",))
except ValueError as error:
    print(error)
```

The orbit blocks of orbix (`OrbitBlock` and `ThieleInnesBlock`) and the fits of
photomancy are built on this contract.
