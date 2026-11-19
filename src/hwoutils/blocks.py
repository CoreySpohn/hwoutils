"""The block contract: a free entry that owns a node of a model and writes it.

A block turns coordinates ``x`` (a dict of named arrays) into values for one or more
model paths, carries a normalized density over those coordinates, and declares
statically which model paths it reads and writes. A host that frees several blocks uses
:func:`check_writers` to refuse two writers of one path and :func:`block_order` to apply
the blocks in an order that respects their reads. The module knows no domain: orbit,
disk and composition blocks subclass :class:`AbstractBlock` in the libraries that own
them. Needs the ``hwoutils[jax]`` extra.
"""

from abc import abstractmethod
from collections.abc import Iterable, Mapping

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike

from hwoutils.model_paths import paths_overlap


class AbstractBlock(eqx.Module):
    """Coordinates that write a node of a model, with a normalized density.

    Coordinates ``x`` are unconstrained reals, one named array per coordinate. Every
    method is per draw; a host batches with ``jax.vmap``. ``reads`` and ``writes`` are
    static tuples of model paths. ``writes`` starts with ``owns`` and lists any further
    path the block sets, such as a mass the coordinates imply. A block never reads a
    model path outside ``reads``, ``owns`` and its own fields.

    Attributes:
        owns: Path of the owned node; the key of the block's entry in a fit.
        reads: Model paths read besides the owned node.
        writes: Model paths written, ``owns`` first.
    """

    owns: eqx.AbstractVar[str]
    reads: eqx.AbstractVar[tuple[str, ...]]
    writes: eqx.AbstractVar[tuple[str, ...]]

    @abstractmethod
    def forward(self, x: dict[str, ArrayLike], model) -> dict[str, object]:
        """Return ``{path: value}`` for every path in ``writes``."""

    @abstractmethod
    def inverse(self, model) -> dict[str, Array]:
        """Return coordinates whose ``forward`` rebuilds the owned node of ``model``."""

    @abstractmethod
    def log_prob(self, x: dict[str, ArrayLike], model) -> Array:
        """Return the scalar log density of ``x`` against Lebesgue ``dx``.

        The density is normalized given the values the block reads from ``model``;
        the volume of the coordinate map, any gauge and every constant are inside.
        """

    @abstractmethod
    def sample(self, key, model) -> dict[str, Array]:
        """Return one draw of coordinates from the density of :meth:`log_prob`."""

    @abstractmethod
    def same_map(self, other) -> bool:
        """Whether ``other`` writes the same paths from the same coordinates.

        True when everything but the law is equal, so that swapping ``other`` in
        changes the density and nothing else. Evaluated on the host.
        """


def check_writers(
    blocks: Mapping[str, AbstractBlock], leaf_paths: Iterable[str] = ()
) -> None:
    """Refuse two writers of one path.

    ``blocks`` maps each block's key to the block and ``leaf_paths`` are the paths of
    free leaf laws.

    Raises:
        ValueError: Naming the path and both writers when a block's key is not its
            owned path, two blocks write overlapping paths, or a leaf law overlaps a
            path a block writes.
    """
    writers: list[tuple[str, str]] = []
    for key, block in blocks.items():
        if key != block.owns:
            raise ValueError(
                f"block entry {key!r} must be keyed by the path it owns, {block.owns!r}"
            )
        writers.extend((path, f"block {key!r}") for path in block.writes)
    writers.extend((path, f"free leaf {path!r}") for path in leaf_paths)
    for i, (path_a, who_a) in enumerate(writers):
        for path_b, who_b in writers[i + 1 :]:
            if who_a != who_b and paths_overlap(path_a, path_b):
                raise ValueError(
                    f"path {path_a!r} has two writers: {who_a} and {who_b}"
                )


def block_order(blocks: Mapping[str, AbstractBlock]) -> tuple[str, ...]:
    """Keys ordered so each block runs after the blocks that write what it reads.

    The order is stable (ties keep the mapping's order).

    Raises:
        ValueError: If the blocks read each other's writes in a cycle.
    """
    keys = list(blocks)
    after = {
        k: {
            j
            for j in keys
            if j != k
            and any(
                paths_overlap(w, r) for w in blocks[j].writes for r in blocks[k].reads
            )
        }
        for k in keys
    }
    order: list[str] = []
    while len(order) < len(keys):
        ready = next(
            (k for k in keys if k not in order and after[k] <= set(order)), None
        )
        if ready is None:
            raise ValueError("blocks read each other's writes in a cycle")
        order.append(ready)
    return tuple(order)
