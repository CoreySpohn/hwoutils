"""Dotted paths into a model pytree: read, replace and compare.

A path is a dotted attribute chain. Bracket keys select tuple or list members
by index or by their unique string name, for example ``"stages[mask].op.shift"``.
The empty path ``""`` names the model itself. Needs the ``hwoutils[jax]`` extra.
"""

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from itertools import combinations
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from hwoutils.precision import like, require_float64

_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_-]*"
_KEY_PATTERN = rf"(?:\d+|{_NAME_PATTERN})"
_NAME = re.compile(_NAME_PATTERN)
_STEP = re.compile(rf"([A-Za-z_]\w*)((?:\[{_KEY_PATTERN}\])*)")
_INDEX = re.compile(rf"\[{_KEY_PATTERN}\]")


def _segments(path: str) -> Iterator[str]:
    for part in path.split(".") if path else ():
        found = _STEP.fullmatch(part)
        if found is None:
            raise ValueError(f"malformed model path {path!r}")
        yield found.group(1)
        yield from _INDEX.findall(found.group(2))


def _key_index(sequence, key: str, path: str) -> int:
    if not isinstance(sequence, (tuple, list)):
        raise ValueError(
            f"model path {path!r}: [k] indexes a tuple or list, not an array axis"
        )
    if key.isdecimal():
        return int(key)
    names = [getattr(item, "name", None) for item in sequence]
    matches = [
        index
        for index, name in enumerate(names)
        if isinstance(name, str) and name == key
    ]
    if not matches:
        present = [name for name in names if isinstance(name, str)]
        # Keep the refusal recognizable for paths on sequences with no names.
        title = "model path" if present else "malformed model path"
        raise ValueError(
            f"{title} {path!r}: no key {key!r}; names present: {present!r}"
        )
    if len(matches) > 1:
        raise ValueError(f"model path {path!r}: ambiguous name key {key!r}")
    return matches[0]


def _follow(value, segment: str, path: str) -> tuple[Any, int | None]:
    if segment.startswith("["):
        index = _key_index(value, segment[1:-1], path)
        return value[index], index
    return getattr(value, segment), None


def resolve(model, path: str) -> Any:
    """Read the value at a dotted path of ``model``.

    Args:
        model: The pytree to read.
        path: Dotted attribute path with bracket indices or name keys into
            tuples or lists. The empty path names the model.

    Returns:
        The value at the path.

    Raises:
        ValueError: If the path is malformed, a key follows a non-sequence,
            or a name key is missing or ambiguous.
    """
    for segment in _segments(path):
        model, _ = _follow(model, segment, path)
    return model


def _display_key(sequence, value, index: int) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and _NAME.fullmatch(name):
        names = (getattr(item, "name", None) for item in sequence)
        if sum(isinstance(other, str) and other == name for other in names) == 1:
            return name
    return str(index)


def _path_spelling(model, path: str, *, display: bool) -> str:
    spelling = ""
    for segment in _segments(path):
        parent = model
        model, index = _follow(parent, segment, path)
        if index is not None:
            key = _display_key(parent, model, index) if display else str(index)
            segment = f"[{key}]"
        spelling = join(spelling, segment)
    return spelling


def canonical_path(model, path: str) -> str:
    """Resolve a path and spell every key as a plain decimal index.

    Args:
        model: The pytree that gives the path meaning.
        path: Dotted attribute path, as for :func:`resolve`.

    Returns:
        The path with name keys and numeric aliases normalized.
    """
    return _path_spelling(model, path, display=False)


def display_path(model, path: str) -> str:
    """Spell each key with its element's name when that name is addressable.

    Args:
        model: The pytree that gives the path meaning.
        path: Dotted attribute path, as for :func:`resolve`.

    Returns:
        The equivalent path with unique valid names where available.
    """
    return _path_spelling(model, path, display=True)


def _converted(model, path: str, value):
    prefix, _, attr = path.rpartition(".")
    if not attr or attr.endswith("]"):
        return value
    owner = resolve(model, prefix)
    if is_dataclass(owner):
        for field in fields(owner):
            if field.name == attr:
                converter = field.metadata.get("converter")
                return converter(value) if converter is not None else value
    return value


def _is_none(value):
    return value is None


def assign(model, path: str, value, *, convert: bool = False):
    """Return a copy of the model with one node replaced.

    Args:
        model: The pytree to update.
        path: Dotted attribute path, as for :func:`resolve`.
        value: Replacement node, including None for an optional slot.
        convert: Apply the final attribute's field converter if it has one.

    Returns:
        The updated model, or the value itself for an empty path.
    """
    if not path:
        return value
    canonical = canonical_path(model, path)
    replacement = _converted(model, canonical, value) if convert else value
    return eqx.tree_at(
        lambda tree: resolve(tree, canonical), model, replacement, is_leaf=_is_none
    )


def assign_many(model, mapping: Mapping[str, Any], *, convert: bool = False):
    """Replace disjoint nodes together, selecting every node on the input model.

    Args:
        model: The pytree to update.
        mapping: Paths and their replacement nodes.
        convert: Apply field converters to replacements where available.

    Returns:
        The updated model, or the input model for an empty mapping.

    Raises:
        ValueError: If two paths name the same node or overlapping nodes.
    """
    if not mapping:
        return model
    entries = [
        (path, canonical_path(model, path), value) for path, value in mapping.items()
    ]
    for (first, a, _), (second, b, _) in combinations(entries, 2):
        if paths_overlap(a, b):
            raise ValueError(f"replacement paths {first!r} and {second!r} overlap")
    replacements = tuple(
        _converted(model, path, value) if convert else value
        for _, path, value in entries
    )
    return eqx.tree_at(
        lambda tree: tuple(resolve(tree, path) for _, path, _ in entries),
        model,
        replacements,
        is_leaf=_is_none,
    )


def apply_at(model, path: str, fn: Callable[[Any], Any]):
    """Apply a function to a node and replace it with the result.

    Args:
        model: The pytree to update.
        path: Dotted attribute path, as for :func:`resolve`.
        fn: Function from the original node to its replacement.

    Returns:
        The updated model.
    """
    return assign(model, path, fn(resolve(model, path)))


def scale_at(model, path: str, factor):
    """Scale an inexact JAX leaf at its existing precision.

    Args:
        model: The pytree to update.
        path: Dotted attribute path naming the array.
        factor: Multiplier, which may be traced. A real leaf needs a real factor.

    Returns:
        The updated model with a strong array at the selected path.

    Raises:
        TypeError: If the target is not an inexact JAX array, or the factor is
            complex and the target is real.
    """
    leaf = resolve(model, path)
    if not isinstance(leaf, jax.Array) or not jnp.issubdtype(leaf.dtype, jnp.inexact):
        raise TypeError(f"model path {path!r} must select an inexact JAX array")
    if jnp.iscomplexobj(factor) and not jnp.iscomplexobj(leaf):
        raise TypeError(f"model path {path!r} needs a real scale factor")
    if leaf.weak_type:
        if jax.dtypes.canonicalize_dtype(leaf.dtype) != leaf.dtype:
            require_float64("scale_at")
        leaf = jnp.asarray(leaf, dtype=leaf.dtype)
    return assign(model, path, leaf * like(factor, leaf))


def paths_overlap(a: str, b: str) -> bool:
    """Whether two model paths are equal or one names a node containing the other.

    The empty path (the model) overlaps every path.
    ``"dynamics.orbits"`` overlaps ``"dynamics.orbits"`` and ``"dynamics.orbits[0].ex"``
    but not ``"dynamics.orbits_extra"``.

    Args:
        a: First model path.
        b: Second model path.

    Returns:
        True when the paths are equal or one contains the other.
    """
    short, long = sorted((a, b), key=len)
    if not short:
        return True
    return long == short or (long.startswith(short) and long[len(short)] in (".", "["))


def ancestors(model, path: str) -> Iterator[tuple[str, object]]:
    """Yield ``(prefix, node)`` for every proper prefix of ``path``, root first.

    The root comes first as ``("", model)``; ``[k]`` steps count as steps, so
    ``"planets[1].mass"`` yields ``""``, ``"planets"`` and ``"planets[1]"``. The
    path itself is not yielded.

    Args:
        model: The pytree the path reads.
        path: Dotted attribute path, as for :func:`resolve`.

    Yields:
        Pairs of a proper prefix of ``path`` and the node of ``model`` at it.

    Raises:
        ValueError: As for :func:`resolve`.
    """
    segments = tuple(_segments(path))
    if not segments:
        return
    prefix = ""
    yield prefix, model
    for segment in segments[:-1]:
        model, _ = _follow(model, segment, path)
        prefix = join(prefix, segment)
        yield prefix, model


def join(prefix: str, path: str) -> str:
    """Join two model paths, where either may be empty or start with an index.

    ``join("", "star.mass")`` is ``"star.mass"``, ``join("scene", "")`` is ``"scene"``,
    ``join("planets", "[1].mass")`` is ``"planets[1].mass"`` and
    ``join("scene", "star.mass")`` is ``"scene.star.mass"``.

    Args:
        prefix: Leading model path, possibly empty.
        path: Model path appended to ``prefix``, possibly empty or starting with an
            index such as ``"[1]"``.

    Returns:
        The joined model path.
    """
    if not prefix:
        return path
    if not path:
        return prefix
    return prefix + path if path.startswith("[") else f"{prefix}.{path}"
