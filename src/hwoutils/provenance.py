"""Host-side build facts and owner-relative reads for model products."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isnan
from typing import NamedTuple

import jax
import numpy as np

from hwoutils import model_paths
from hwoutils.leaves import HostGrid

Stamps = tuple[tuple[str, str], ...]
UNKNOWN = "unknown"
_LABEL = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
_OWNER = re.compile(r"[A-Za-z_]\w*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Digest:
    """A native SHA-256 digest, ready for use as a build fact.

    Attributes:
        hex: Full SHA-256 digest in lowercase hexadecimal.
    """

    hex: str

    def __post_init__(self):
        """Refuse incomplete or noncanonical digests."""
        if not isinstance(self.hex, str) or _SHA256.fullmatch(self.hex) is None:
            raise ValueError("digest hex must contain 64 lowercase hexadecimal digits")


def stamp_value(value: object, *, label: str = "") -> str:
    """Encode one concrete build fact as typed text.

    Args:
        value: Scalar, digest, host grid, concrete array or tuple of facts.
            Use None for an unknown fact.
        label: Fact label used in errors.

    Returns:
        Typed fact string with exact float encodings and full array digests.

    Raises:
        TypeError: If the value is traced or has an unsupported type.
        ValueError: If a floating scalar is NaN.
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "bool:true" if value else "bool:false"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.float64, np.float32)):
        number = float(value)
        if isnan(number):
            raise ValueError(f"NaN is not a build fact for {label!r}")
        width = "f32" if isinstance(value, np.float32) else "f64"
        return f"{width}:{(number or 0.0).hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, Digest):
        return f"sha256:{value.hex}"
    if isinstance(value, HostGrid):
        return f"grid:{len(value.values)}:sha256:{value._sha.hex()}"
    if isinstance(value, (np.ndarray, jax.Array)) and not isinstance(
        value, jax.core.Tracer
    ):
        array = np.asarray(value)
        digest = sha256(array.tobytes()).hexdigest()
        return f"array:{array.dtype.str}:{array.shape}:sha256:{digest}"
    if isinstance(value, tuple):
        items = ",".join(stamp_value(item, label=label) for item in value)
        return f"tuple:({items})"
    raise TypeError(f"unsupported build fact for {label!r}: {type(value).__name__}")


def _check_label(label: str):
    if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
        raise ValueError(f"invalid stamp label {label!r}")


def build_stamps(**facts: object) -> Stamps:
    """Encode build facts in sorted label order.

    Args:
        **facts: Label-to-value build facts with lowercase dotted labels.

    Returns:
        Sorted, hashable pairs of labels and typed fact strings.
    """
    for label in facts:
        _check_label(label)
    return tuple(
        (label, stamp_value(value, label=label))
        for label, value in sorted(facts.items())
    )


def prefix_stamps(prefix: str, stamps: Stamps) -> Stamps:
    """Give a child's build stamps a common dotted prefix.

    Args:
        prefix: Lowercase dotted label for the child.
        stamps: The child's sorted build stamps.

    Returns:
        The sorted build stamps with each label prefixed.
    """
    _check_label(prefix)
    return tuple((f"{prefix}.{label}", value) for label, value in stamps)


def merge_stamps(*stamps: Stamps) -> Stamps:
    """Combine build stamps that agree on shared labels.

    Args:
        *stamps: Zero or more sorted stamp collections.

    Returns:
        Sorted union of the stamps.

    Raises:
        ValueError: If a label has two different values.
    """
    merged: dict[str, str] = {}
    for collection in stamps:
        for label, value in collection:
            if label in merged and merged[label] != value:
                raise ValueError(f"conflicting build stamps for {label!r}")
            merged[label] = value
    return tuple(sorted(merged.items()))


def display_key(stamps: Stamps) -> str:
    """Produce a short identifier for messages and representations.

    Args:
        stamps: Sorted build stamps.

    Returns:
        The first 16 hexadecimal digits of the SHA-256 of the stamp text.
    """
    text = "\n".join(f"{label}={value}" for label, value in stamps)
    return sha256(text.encode()).hexdigest()[:16]


class StampComparison(NamedTuple):
    """Three-way comparison of build facts, in label order.

    Attributes:
        matched: Labels with the same known value on both sides.
        unverified: Labels missing or unknown on either side.
        mismatched: Triples of label, left value and right value for conflicts.
    """

    matched: tuple[str, ...]
    unverified: tuple[str, ...]
    mismatched: tuple[tuple[str, str, str], ...]


class StampMismatchError(ValueError):
    """Two builds recorded incompatible values for a known fact."""


def compare_stamps(
    left: Stamps, right: Stamps, *, labels: Iterable[str] | None = None
) -> StampComparison:
    """Compare build facts by their complete typed values.

    Args:
        left: First collection of build stamps.
        right: Second collection of build stamps.
        labels: Optional selection, defaulting to the union of labels.

    Returns:
        Matched, unverified and mismatched facts, each in sorted label order.
    """
    a, b = dict(left), dict(right)
    selected = a.keys() | b.keys() if labels is None else set(labels)
    matched, unverified, mismatched = [], [], []
    for label in sorted(selected):
        left_value, right_value = a.get(label, UNKNOWN), b.get(label, UNKNOWN)
        if UNKNOWN in (left_value, right_value):
            unverified.append(label)
        elif left_value == right_value:
            matched.append(label)
        else:
            mismatched.append((label, left_value, right_value))
    return StampComparison(tuple(matched), tuple(unverified), tuple(mismatched))


def require_stamps(
    left: Stamps, right: Stamps, *, what: str, labels: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Require agreement on known facts and return those still unverified.

    Args:
        left: First collection of build stamps.
        right: Second collection of build stamps.
        what: Product or operation being checked, for error messages.
        labels: Optional selection, defaulting to the union of labels.

    Returns:
        Sorted labels missing or unknown on either side.

    Raises:
        StampMismatchError: If both sides know a fact and its values differ.
    """
    result = compare_stamps(left, right, labels=labels)
    if result.mismatched:
        details = ", ".join(
            f"{label} ({a!r} vs {b!r})" for label, a, b in result.mismatched
        )
        raise StampMismatchError(f"{what} has incompatible build stamps: {details}")
    return result.unverified


def qualify(owner: str, reads: Iterable[str]) -> tuple[str, ...]:
    """Prefix relative read paths with their owner.

    Args:
        owner: Bare identifier for the owner.
        reads: Model paths relative to the owner, including "" for its root.

    Returns:
        Owner-qualified model paths in input order.
    """
    if not owner.isidentifier():
        raise ValueError(f"invalid read owner {owner!r}")
    return tuple(model_paths.join(owner, read) for read in reads)


def bind_reads(
    reads: Iterable[str], roots: Mapping[str, str | None]
) -> tuple[str, ...]:
    """Rewrite owner-qualified reads as paths on a single model.

    Args:
        reads: Owner-qualified model paths.
        roots: Model path of each owner, "" for the model root, or None for
            an owner held outside the model.

    Returns:
        Bound paths in input order, omitting reads of held owners.

    Raises:
        ValueError: If a read names an owner absent from roots.
    """
    bound = []
    for read in reads:
        match = _OWNER.match(read)
        owner = match.group() if match else read
        relative = read[len(owner) :].removeprefix(".")
        if owner not in roots:
            raise ValueError(
                f"unknown read owner {owner!r}; known owners: {sorted(roots)}"
            )
        root = roots[owner]
        if root is not None:
            bound.append(model_paths.join(root, relative))
    return tuple(bound)


def stale_reads(
    reads: Iterable[str], writes: Iterable[str]
) -> tuple[tuple[str, str], ...]:
    """Find writes that overlap a product's reads.

    Args:
        reads: Paths the product read on a model.
        writes: Paths written on the same model. Canonicalize both inputs
            first when they may use more than one spelling for the same node.

    Returns:
        Every overlapping (write, read) pair in write order, then read order.
    """
    reads = tuple(reads)
    return tuple(
        (write, read)
        for write in writes
        for read in reads
        if model_paths.paths_overlap(write, read)
    )
