"""Dotted paths into a model pytree: read, replace and compare.

A path is a dotted attribute chain in which ``[k]`` indexes a tuple or list, for
example ``"dynamics.orbits"`` or ``"planets[1].Rp_Rearth"``. ``[k]`` never indexes an
array axis. Needs the ``hwoutils[jax]`` extra.
"""

import re

import equinox as eqx

_STEP = re.compile(r"([A-Za-z_]\w*)((?:\[\d+\])*)")
_INDEX = re.compile(r"\[(\d+)\]")


def resolve(model, path: str):
    """Read the value at a dotted path of ``model``.

    Args:
        model: The pytree to read.
        path: Dotted attribute path, with ``[k]`` to index a tuple or list.

    Returns:
        The value at the path.

    Raises:
        ValueError: If the path is malformed.
    """
    value = model
    for part in path.split("."):
        found = _STEP.fullmatch(part)
        if found is None:
            raise ValueError(f"malformed model path {path!r}")
        value = getattr(value, found.group(1))
        for index in _INDEX.findall(found.group(2)):
            value = value[int(index)]
    return value


def assign(model, path: str, value):
    """Return ``model`` with the node at ``path`` replaced by ``value``.

    Args:
        model: The pytree to update; it is not modified.
        path: Dotted attribute path, as for :func:`resolve`.
        value: The replacement node.

    Returns:
        A copy of ``model`` holding ``value`` at the path.
    """
    return eqx.tree_at(lambda m: resolve(m, path), model, value)


def paths_overlap(a: str, b: str) -> bool:
    """Whether two model paths are equal or one names a node containing the other.

    ``"dynamics.orbits"`` overlaps ``"dynamics.orbits"`` and ``"dynamics.orbits[0].ex"``
    but not ``"dynamics.orbits_extra"``.
    """
    short, long = sorted((a, b), key=len)
    return long == short or (long.startswith(short) and long[len(short)] in (".", "["))
