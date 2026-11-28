"""The block contract: a free entry that owns a node of a model and writes it.

A block turns coordinates ``x`` (a dict of named arrays) into values for one or more
model paths, carries a normalized density over those coordinates, and declares
statically which model paths it reads and writes and which coordinates repeat. A
host that frees several blocks uses :func:`check_writers` to refuse two writers of one
path and :func:`block_order` to apply the blocks in an order that respects their
reads.

A model node may declare that some of its fields are derived from others, through a
class attribute ``tied_fields``, a read-only mapping from each derived field to the
fields it is derived from (paths relative to the node). A host
calls :func:`check_ties` to refuse a free entry that would break such a tie: a path
inside a derived field, or a source written while no entry that reads it writes the
derived field.
The node's library declares the tie and the host checks it, so neither names the
other. The module knows no domain: orbit, disk and composition blocks subclass
:class:`AbstractBlock` in the libraries that own them. Needs the ``hwoutils[jax]``
extra.
"""

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike

from hwoutils.model_paths import ancestors, canonical_path, join, paths_overlap


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


class _CanonicalBlock(NamedTuple):
    key: str
    owns: str
    writes: tuple[tuple[str, str], ...]
    reads: tuple[str, ...]


def _canonical_blocks(blocks, model):
    normalized = [
        (
            key,
            _CanonicalBlock(
                canonical_path(model, key),
                canonical_path(model, block.owns),
                tuple((path, canonical_path(model, path)) for path in block.writes),
                tuple(canonical_path(model, path) for path in block.reads),
            ),
        )
        for key, block in blocks.items()
    ]
    return dict(normalized)


def check_writers(
    blocks: Mapping[str, AbstractBlock], leaf_paths: Iterable[str] = (), *, model=None
) -> None:
    """Refuse two writers of one path.

    ``blocks`` maps each block's key to the block and ``leaf_paths`` are the paths of
    free leaf laws.

    Args:
        blocks: Mapping from block key to block.
        leaf_paths: Model paths written by free leaf laws.
        model: Optional model used to resolve aliases before comparing paths.

    Raises:
        ValueError: Naming the path and both writers when a block's key is not its
            owned path, two blocks write overlapping paths, or a leaf law overlaps a
            path a block writes.
    """
    normalized = _canonical_blocks(blocks, model) if model is not None else {}
    leaves = tuple(
        (path, canonical_path(model, path) if model is not None else path)
        for path in leaf_paths
    )
    writers: list[tuple[str, str, str]] = []
    for key, block in blocks.items():
        entry = normalized.get(key)
        lookup, owns = (key, block.owns) if entry is None else (entry.key, entry.owns)
        if lookup != owns:
            raise ValueError(
                f"block entry {key!r} must be keyed by the path it owns, {block.owns!r}"
            )
        writes = (
            ((path, path) for path in block.writes) if entry is None else entry.writes
        )
        writers.extend(
            (path, canonical, f"block {key!r}") for path, canonical in writes
        )
    writers.extend(
        (path, canonical, f"free leaf {path!r}") for path, canonical in leaves
    )
    for i, (path, a, who_a) in enumerate(writers):
        for _, b, who_b in writers[i + 1 :]:
            if who_a != who_b and paths_overlap(a, b):
                raise ValueError(f"path {path!r} has two writers: {who_a} and {who_b}")


def block_order(blocks: Mapping[str, AbstractBlock], *, model=None) -> tuple[str, ...]:
    """Keys ordered so each block runs after the blocks that write what it reads.

    The order is stable (ties keep the mapping's order).

    Args:
        blocks: Mapping from block key to block.
        model: Optional model used to resolve aliases before comparing paths.

    Returns:
        The keys of ``blocks`` in an order where every writer of a path comes before
        the blocks that read it.

    Raises:
        ValueError: If the blocks read each other's writes in a cycle.
    """
    keys = list(blocks)
    if model is None:
        writes = {key: block.writes for key, block in blocks.items()}
        reads = {key: block.reads for key, block in blocks.items()}
    else:
        normalized = _canonical_blocks(blocks, model)
        writes = {
            key: tuple(path for _, path in entry.writes)
            for key, entry in normalized.items()
        }
        reads = {key: entry.reads for key, entry in normalized.items()}
    after = {
        k: {
            j
            for j in keys
            if j != k and any(paths_overlap(w, r) for w in writes[j] for r in reads[k])
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


def check_ties(
    model,
    writes: Mapping[str, Iterable[str]],
    reads: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Refuse free entries that break a tie a model node declares.

    A node whose class declares ``tied_fields`` (a mapping from a derived field to the
    fields it is derived from, paths relative to the node) is found at every proper
    prefix of each written path, the model root included. The check refuses two
    things: (a) a written path strictly inside a derived field, whose value would no
    longer agree with its sources; and (b) a written source of a derived field unless
    some entry writes that field (or a node containing it) and reads or writes that
    source, so that the derived field follows the new source values.

    Args:
        model: The model the entries write.
        writes: Mapping from each entry's key to the model paths it writes (a leaf
            law writes its key, a block its ``writes``).
        reads: Mapping from each entry's key to the model paths it reads (a block's
            ``reads``); an entry left out reads nothing.

    Raises:
        ValueError: Naming the path, the derived field and the way out.
    """

    def normalize(path):
        return canonical_path(model, path) if model is not None else path

    written = [
        (key, tuple((path, normalize(path)) for path in paths))
        for key, paths in writes.items()
    ]
    reading = [
        (key, tuple(normalize(path) for path in paths))
        for key, paths in (reads or {}).items()
    ]
    canonical_reads = dict(reading)
    write_paths = {key: tuple(path for _, path in paths) for key, paths in written}
    touched = {
        key: (*paths, *canonical_reads.get(key, ()))
        for key, paths in write_paths.items()
    }
    for path, canonical in (pair for _, paths in written for pair in paths):
        for prefix, node in ancestors(model, path):
            for derived, sources in getattr(node, "tied_fields", {}).items():
                target = join(prefix, derived)
                target_path = normalize(target)
                if len(canonical) > len(target_path) and paths_overlap(
                    canonical, target_path
                ):
                    raise ValueError(
                        f"free path {path!r} lies inside {target!r}, whose fields "
                        f"are tied together: free it through a block at {target!r}"
                    )
                for source in (join(prefix, s) for s in sources):
                    source_path = normalize(source)
                    if not paths_overlap(canonical, source_path):
                        continue
                    covered = any(
                        any(
                            paths_overlap(w, target_path) and len(w) <= len(target_path)
                            for w in paths
                        )
                        and any(paths_overlap(t, source_path) for t in touched[key])
                        for key, paths in write_paths.items()
                    )
                    if not covered:
                        raise ValueError(
                            f"free path {path!r} is a source of {target!r}, which no "
                            f"entry that reads it writes: add a block that writes "
                            f"{target!r} from {source!r}"
                        )
