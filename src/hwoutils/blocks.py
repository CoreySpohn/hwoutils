"""The block contract: a free entry that owns a node of a model and writes it.

A block turns coordinates ``x`` (a dict of named arrays) into values for one or more
model paths, carries a normalized density over those coordinates, and declares
statically which model paths it reads and writes and which coordinates repeat. A
host that frees several blocks uses :func:`check_writers` to refuse two writers of one
path and :func:`block_order` to apply the blocks in an order that respects their
reads.

A model node may declare that some of its fields are derived from others, through a
class attribute ``tied_fields: ClassVar[dict[str, tuple[str, ...]]]`` that maps each
derived field to the fields it is derived from (paths relative to the node). A host
calls :func:`check_ties` to refuse a free entry that would break such a tie: a path
inside a derived field, or a source written while nothing writes the derived field.
The node's library declares the tie and the host checks it, so neither names the
other. The module knows no domain: orbit, disk and composition blocks subclass
:class:`AbstractBlock` in the libraries that own them. Needs the ``hwoutils[jax]``
extra.
"""

from abc import abstractmethod
from collections.abc import Iterable, Mapping

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike

from hwoutils.model_paths import ancestors, join, paths_overlap


class AbstractBlock(eqx.Module):
    """Coordinates that write a node of a model, with a normalized density.

    Coordinates ``x`` are real numbers, one named array per coordinate, with no
    boundary: every coordinate takes any real value; a zero-density region inside
    those values is a wall and is listed in :attr:`walls`. A coordinate listed in
    :attr:`periods` is an angle-like one: the model depends on it only through its value
    modulo the period, and the density is that of one fundamental cell
    ``[0, period)``, repeated outside it. Such a density is normalized per cell and is
    not integrable over the real line, so a host that proposes from a density with
    support beyond one cell must wrap the proposal onto the cell. Every other coordinate
    is non-periodic and :meth:`log_prob` is normalized over the real line.

    :meth:`log_prob` returns ``-inf``, never NaN or ``+inf``, where the density is zero
    or the arithmetic overflows (coordinates of 1e5, say).

    Every method is per draw; a host batches with ``jax.vmap``. ``reads`` and ``writes``
    are static tuples of model paths. ``writes`` starts with ``owns`` and lists any
    further path the block sets, such as a mass the coordinates imply. A block never
    reads a model path outside ``reads``, ``owns`` and its own fields.

    A block may offer a deterministic map of the unit cube, for a host that needs one
    (nested sampling, low-discrepancy sequences): :attr:`unit_size` is the number of
    cube coordinates and :meth:`from_unit` the map. A block without one keeps the
    defaults, ``unit_size`` of ``None`` and a refusing ``from_unit``.

    Attributes:
        owns: Path of the owned node; the key of the block's entry in a fit.
        reads: Model paths read besides the owned node.
        writes: Model paths written, ``owns`` first.
        coordinate_names: Names of the coordinates, the keys of ``x``.
    """

    owns: eqx.AbstractVar[str]
    reads: eqx.AbstractVar[tuple[str, ...]]
    writes: eqx.AbstractVar[tuple[str, ...]]
    coordinate_names: eqx.AbstractVar[tuple[str, ...]]

    @property
    def periods(self) -> tuple[float | None, ...]:
        """Period of each coordinate in ``coordinate_names`` order, ``None`` if none.

        All ``None`` by default.
        """
        return (None,) * len(self.coordinate_names)

    @property
    def walls(self) -> tuple[str, ...]:
        """Names of the bounded quantities whose bounds the coordinates do not map.

        A block cannot see the model, so a name is listed when for some model
        :meth:`log_prob` is ``-inf`` on a set of positive measure at finite
        coordinates. Empty (the default) means that for every model the density is
        positive almost everywhere in the coordinates. Evaluated on the host; a bound
        that is not mapped and whose value is traced is listed.
        """
        return ()

    @property
    def unit_size(self) -> int | None:
        """Length of the unit-cube vector of :meth:`from_unit`; ``None`` if absent.

        The number of unit-cube coordinates :meth:`from_unit` consumes.
        """
        return None

    @abstractmethod
    def forward(self, x: dict[str, ArrayLike], model) -> dict[str, object]:
        """Return ``{path: value}`` for every path in ``writes``.

        Args:
            x: Coordinates, one scalar array per name in ``coordinate_names``.
            model: The model whose paths the block reads.

        Returns:
            Mapping from each written model path to the value the block sets.
        """

    @abstractmethod
    def inverse(self, model) -> dict[str, Array]:
        """Return coordinates whose ``forward`` rebuilds the owned node of ``model``.

        Where the model value lies outside the support, every coordinate of that
        entry is NaN. A value outside by no more than rounding is held at the
        boundary; one outside by more is never clipped onto it. NaN propagates through
        every consumer, so a host marks the draw dead with ``isfinite``.

        Args:
            model: The model holding the owned node to invert.

        Returns:
            Mapping from coordinate name to a scalar array, in the units of the
            coordinate.
        """

    @abstractmethod
    def log_prob(self, x: dict[str, ArrayLike], model) -> Array:
        """Return the scalar log density of ``x`` against Lebesgue ``dx``.

        The density is normalized given the values the block reads from ``model``;
        the volume of the coordinate map, any gauge and every constant are inside.

        Args:
            x: Coordinates, one scalar array per name in ``coordinate_names``.
            model: The model whose paths the block reads.

        Returns:
            Scalar natural-log density, ``-inf`` where the density is zero.
        """

    @abstractmethod
    def sample(self, key, model) -> dict[str, Array]:
        """Return one draw of coordinates from the density of :meth:`log_prob`.

        Args:
            key: JAX PRNG key for the draw.
            model: The model whose paths the block reads.

        Returns:
            Mapping from coordinate name to a scalar array.
        """

    def from_unit(self, c: ArrayLike, model) -> dict[str, Array]:
        """Map a point ``c`` of the unit cube to coordinates, deterministically.

        ``c`` has shape ``(unit_size,)`` with values in ``(0, 1)``. The map is
        piecewise continuous and preserves locality (nearby points of the cube go to
        nearby coordinates except across a seam), and uniform ``c`` gives coordinates
        with the density of :meth:`log_prob`, the law of :meth:`sample`.

        Args:
            c: Point of the unit cube, shape ``(unit_size,)``.
            model: The model whose paths the block reads.

        Returns:
            Mapping from coordinate name to a scalar array.

        Raises:
            NotImplementedError: For a block with no unit-cube map (``unit_size`` is
                ``None``).
        """
        raise NotImplementedError(f"{type(self).__name__} has no unit-cube map")

    @abstractmethod
    def same_layout(self, other) -> bool:
        """Whether ``other`` is equal to this block in everything but the law.

        True when the two have the same type, ownership and reads, chart type and
        rows (whatever the concrete block's coordinates are made of), so that
        ``other`` can replace this block in a fit. The law, and any bound it
        carries, is not compared. Evaluated on the host.

        Args:
            other: The block to compare with.

        Returns:
            True when the layouts match.
        """

    @abstractmethod
    def same_map(self, other) -> bool:
        """Whether ``same_layout`` holds and the mapped support is the same.

        True when swapping ``other`` in changes the density and nothing else, so
        coordinates drawn for this block stay valid for ``other``. False when that
        cannot be decided statically (a traced bound). Evaluated on the host.

        Args:
            other: The block to compare with.

        Returns:
            True when the layouts match and the mapped support is the same.
        """


def check_writers(
    blocks: Mapping[str, AbstractBlock], leaf_paths: Iterable[str] = ()
) -> None:
    """Refuse two writers of one path.

    ``blocks`` maps each block's key to the block and ``leaf_paths`` are the paths of
    free leaf laws.

    Args:
        blocks: Mapping from block key to block.
        leaf_paths: Model paths written by free leaf laws.

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

    Args:
        blocks: Mapping from block key to block.

    Returns:
        The keys of ``blocks`` in an order where every writer of a path comes before
        the blocks that read it.

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


def check_ties(model, writes: Mapping[str, Iterable[str]]) -> None:
    """Refuse free entries that break a tie a model node declares.

    A node whose class declares ``tied_fields`` (derived field to source fields,
    paths relative to the node) is found at every proper prefix of each written
    path, the model root included. The check refuses two things: (a) a written path
    strictly inside a derived field, whose value would no longer agree with its
    sources; and (b) a written source of a derived field when no entry writes that
    field or a node containing it, which leaves the derived field at the old source
    values.

    Args:
        model: The model the entries write.
        writes: Mapping from each entry's key to the model paths it writes (a leaf
            law writes its key, a block its ``writes``).

    Raises:
        ValueError: Naming the path, the derived field and the way out.
    """
    written = [path for paths in writes.values() for path in paths]
    for path in written:
        for prefix, node in ancestors(model, path):
            for derived, sources in getattr(node, "tied_fields", {}).items():
                target = join(prefix, derived)
                if len(path) > len(target) and paths_overlap(path, target):
                    raise ValueError(
                        f"free path {path!r} lies inside {target!r}, whose fields "
                        f"are tied together: free it through a block at {target!r}"
                    )
                touches = any(
                    paths_overlap(path, join(prefix, source)) for source in sources
                )
                covered = any(
                    paths_overlap(other, target) and len(other) <= len(target)
                    for other in written
                )
                if touches and not covered:
                    raise ValueError(
                        f"free path {path!r} is a source of {target!r}, which no "
                        f"entry writes: add a block that writes {target!r}"
                    )
