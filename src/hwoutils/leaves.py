"""Leaf converters, leaf checks and the static host grid for Equinox models.

Every leaf of a model is a ``jax.Array``. Floating leaves have the active
float dtype (:func:`hwoutils.jax_config.float_dtype`) and a strong type;
integer and boolean leaves keep their kind. NumPy arrays, NumPy scalars and
Python numbers are never leaves: a Python number lives on a ``static=True``
field or is converted. Conversion is declared on the field::

    class Disk(eqx.Module):
        radius_au: jax.Array = eqx.field(converter=as_float_leaf)
        mask: jax.Array = eqx.field(converter=as_leaf)

Equinox runs converters after ``__init__``, so a custom ``__init__`` that
derives from a field converts its local first. ``eqx.tree_at`` bypasses
converters: pass replacement values through :func:`as_float_leaf`, and call
:func:`require_tree` where a model enters a fit or an exposure.

Three predicates decide what a leaf is, and nothing else tests array types:
:func:`is_data` (slice, stack, copy or serialize it), :func:`is_param` (free,
read, perturb or differentiate it) and :func:`is_traced` (host-only guards).
Both array predicates include NumPy, so a NumPy leaf that slips through is
still found and refused.

Data that a host build or a static layout depends on, such as a wavelength
grid an operator is built on, is a :class:`HostGrid`: immutable float64 on
the host, hashed and compared by a digest, stored on a static field.
"""

import contextlib
import dataclasses
import hashlib
import reprlib
from collections.abc import Callable, Iterable, Iterator
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hwoutils.jax_config import float_dtype

__all__ = [
    "FOREIGN_PACKAGES",
    "OWNED_PACKAGES",
    "HostGrid",
    "LeafViolation",
    "as_bulk_leaf",
    "as_complex_leaf",
    "as_float_leaf",
    "as_leaf",
    "canonical",
    "check_tree",
    "is_data",
    "is_owned",
    "is_param",
    "is_traced",
    "require_tree",
]

OWNED_PACKAGES = frozenset(
    {
        "coronachrome",
        "coronagraphoto",
        "hwoutils",
        "jaxedith",
        "optixstuff",
        "orbix",
        "photomancy",
        "physicaloptix",
        "skyscapes",
        "tiptilt",
        "yippy",
    }
)
"""Top-level packages whose Modules follow the leaf rule."""

FOREIGN_PACKAGES = frozenset({"diffrax", "equinox", "interpax", "lineax", "optimistix"})
"""Third-party packages whose Modules :func:`require_tree` does not enter.

An entry may be a dotted module path. Equinox is exempt as a whole (its
wrappers such as ``eqx.filter_jit`` and ``eqx.filter_grad``, and ``eqx.nn``)
except ``eqx.Partial``, which holds user leaves and is entered."""

_PYTHON_NUMBERS = (bool, int, float, complex)


def is_data(x: Any) -> bool:
    """Return True for a leaf that is data: a JAX or NumPy array or scalar.

    Use it to slice, stack, copy or serialize leaves.

    Args:
        x: Any pytree leaf.

    Returns:
        ``eqx.is_array(x)``.
    """
    return eqx.is_array(x)


def is_param(x: Any) -> bool:
    """Return True for a floating or complex array leaf, which a fit may free.

    Use it to free, read, hold, perturb or differentiate leaves.

    Args:
        x: Any pytree leaf.

    Returns:
        ``eqx.is_inexact_array(x)``.
    """
    return eqx.is_inexact_array(x)


def is_traced(x: Any) -> bool:
    """Return True for a value under a JAX trace; for host-only guards.

    Args:
        x: Any value.

    Returns:
        ``isinstance(x, jax.core.Tracer)``.
    """
    return isinstance(x, jax.core.Tracer)


def _refuse_key(x, helper: str) -> None:
    dtype = getattr(x, "dtype", None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        raise TypeError(
            "a typed PRNG key is not a number; declare the field without a "
            f"converter (a key is already a valid leaf) instead of {helper}"
        )


def _complex_dtype() -> np.dtype:
    return np.dtype(jax.dtypes.canonicalize_dtype(np.complex128))


def _source_dtype(x) -> np.dtype:
    """Dtype of an array, scalar or nested sequence, without a device copy."""
    dtype = getattr(x, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    return np.asarray(x).dtype


def as_float_leaf(x: Any) -> jax.Array:
    """Convert a physical quantity to a strong float leaf of the active dtype.

    Accepts arrays, scalars and nested sequences, NumPy or JAX, integer or
    floating. Tracers are converted the same way. For a field declared
    ``eqx.field(converter=as_float_leaf)``.

    Args:
        x: The value to store.

    Returns:
        A strongly typed ``jax.Array`` of ``float_dtype()``.

    Raises:
        TypeError: If ``x`` is complex (use :func:`as_complex_leaf`) or a typed
            PRNG key (declare the field without a converter).
        ValueError: If a concrete finite ``x`` becomes infinite in the active
            dtype (a float64 value or a huge Python integer beyond the float32
            range under x32), or a nonzero finite ``x`` underflows to zero.
            Tracers are not checked.
    """
    _refuse_key(x, "as_float_leaf")
    source = _source_dtype(x)
    if np.issubdtype(source, np.complexfloating):
        raise TypeError(
            "a complex value cannot be stored on a real field; declare the "
            "field with as_complex_leaf"
        )
    wide = source == np.dtype(object) and not is_traced(x)
    if wide:
        try:
            host = np.asarray(x, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as err:
            raise ValueError(
                f"{reprlib.repr(x)} does not fit a float leaf: {err}"
            ) from err
        x = host
    with np.errstate(over="ignore"):
        out = jnp.asarray(x, dtype=float_dtype())
    if (
        (
            wide
            or (
                np.issubdtype(source, np.floating)
                and source.itemsize > out.dtype.itemsize
            )
        )
        and not is_traced(x)
        and not is_traced(out)
    ):
        host, got = np.asarray(x), np.asarray(out)
        if (np.isfinite(host) & ~np.isfinite(got)).any():
            raise ValueError(
                f"finite values do not fit {out.dtype.name}, which holds up to "
                f"{float(np.finfo(out.dtype).max):.3e}; enable x64, or rescale "
                "the quantity"
            )
        if ((host != 0) & np.isfinite(host) & (got == 0)).any():
            tiny = float(np.finfo(out.dtype).smallest_subnormal)
            raise ValueError(
                f"nonzero values underflow to zero in {out.dtype.name}, whose "
                f"smallest positive value is {tiny:.3e}; "
                "enable x64, or rescale the quantity"
            )
    return out


def as_complex_leaf(x: Any) -> jax.Array:
    """Convert a value to a strong complex leaf of the active precision.

    Args:
        x: The value to store, real or complex.

    Returns:
        A strongly typed complex ``jax.Array`` (complex128 under x64, else
        complex64).
    """
    return jnp.asarray(x, dtype=_complex_dtype())


def _check_integer_range(x, target: np.dtype) -> None:
    """Refuse a concrete integer input that does not fit the target dtype."""
    if is_traced(x):
        return
    values = np.asarray(x)
    if values.size == 0:
        return
    info = np.iinfo(target)
    low, high = values.min(), values.max()
    if low < info.min or high > info.max:
        raise ValueError(
            f"integer values from {reprlib.repr(int(low))} to "
            f"{reprlib.repr(int(high))} do not fit {target.name}, which "
            f"holds {info.min} to {info.max}; enable x64, or store an identifier "
            "that is not an index as static metadata"
        )


def as_leaf(x: Any) -> jax.Array:
    """Convert a value to a strong leaf, keeping its kind.

    Booleans stay boolean and integers stay integer, in JAX's canonical width;
    a concrete integer that does not fit that width is refused rather than
    wrapped. Floats go to the active float dtype and complex values to the
    active complex dtype. For indices, masks and generic data.

    Args:
        x: The value to store.

    Returns:
        A strongly typed ``jax.Array`` of the same kind as ``x``.

    Raises:
        ValueError: If a concrete integer (a Python integer included) does not
            fit the canonical integer dtype (int32 or uint32 without x64).
        TypeError: If ``x`` is a typed PRNG key; declare the field without a
            converter.
    """
    _refuse_key(x, "as_leaf")
    source = _source_dtype(x)
    if source == np.dtype(object) and not is_traced(x):
        values = np.asarray(x, dtype=object)
        if values.size and all(isinstance(v, int) for v in values.ravel()):
            target = np.dtype(jax.dtypes.canonicalize_dtype(np.int64))
            _check_integer_range(x, target)
            source = np.dtype(np.int64)
    if source == np.bool_:
        target = np.dtype(np.bool_)
    elif np.issubdtype(source, np.integer):
        target = np.dtype(jax.dtypes.canonicalize_dtype(source))
        if target.itemsize < source.itemsize or (
            np.issubdtype(source, np.unsignedinteger)
            != np.issubdtype(target, np.unsignedinteger)
        ):
            _check_integer_range(x, target)
    elif np.issubdtype(source, np.complexfloating):
        target = _complex_dtype()
    else:
        target = float_dtype()
    return jnp.asarray(x, dtype=target)


def as_bulk_leaf(x: Any) -> jax.Array:
    """Convert a bulk data cube, keeping a floating dtype narrower than active.

    For declared bulk cubes only (a contrast or disk cube delivered as
    float32): under x64 a float32 input stays float32 and its reader promotes
    at use. Any other input becomes a float leaf of the active dtype, so under
    x32 this is :func:`as_float_leaf`.

    Args:
        x: The cube to store.

    Returns:
        A strongly typed floating ``jax.Array``.

    Raises:
        TypeError: If ``x`` is complex.
    """
    source = _source_dtype(x)
    if np.issubdtype(source, np.complexfloating):
        raise TypeError("a bulk cube must be real")
    if np.issubdtype(source, np.floating) and source.itemsize < float_dtype().itemsize:
        return jnp.asarray(x, dtype=source)
    return jnp.asarray(x, dtype=float_dtype())


class LeafViolation(NamedTuple):
    """A leaf that breaks the leaf rule.

    Attributes:
        path: Key path of the leaf, as ``jax.tree_util.keystr`` writes it.
        type: Type name of the leaf.
        dtype: Dtype name of the leaf, or ``""`` for a Python number.
        reason: What is wrong with it.
    """

    path: str
    type: str
    dtype: str
    reason: str


def _package(cls: type) -> str:
    return cls.__module__.partition(".")[0]


def is_owned(module: Any, packages: Iterable[str] = OWNED_PACKAGES) -> bool:
    """Return True when any class in the Module's MRO is defined in ``packages``.

    A user's subclass of an owned abstract class is owned, since it inherits
    the contract.

    Args:
        module: An ``eqx.Module`` instance.
        packages: Top-level package names.

    Returns:
        Whether the Module is owned by one of the packages.
    """
    packages = frozenset(packages)
    return any(_package(cls) in packages for cls in type(module).__mro__)


def _children(node):
    """Direct children of a pytree node with their keys, or None for a leaf."""
    flat, _ = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda x: x is not node
    )
    if len(flat) == 1 and flat[0][0] == () and flat[0][1] is node:
        return None
    return flat


def _bulk_fields(module) -> frozenset[str]:
    return frozenset(
        field.name
        for field in dataclasses.fields(module)
        if field.metadata.get("converter") is as_bulk_leaf
    )


def _walk(
    tree, enter: Callable[[Any], bool], bulk: frozenset[str]
) -> Iterator[tuple[tuple, Any, bool]]:
    """Yield ``(key path, leaf, bulk allowed)`` for the leaves to check.

    A Module for which ``enter`` is False is skipped with its whole subtree.
    A leaf under a field declared with ``as_bulk_leaf``, or under a path in
    ``bulk``, may keep a narrower float dtype.
    """
    stack = [((), tree, False)]
    while stack:
        path, node, allowed = stack.pop()
        allowed = allowed or jax.tree_util.keystr(path) in bulk
        fields = frozenset()
        if isinstance(node, eqx.Module):
            if path and not enter(node):
                continue
            fields = _bulk_fields(node)
        children = _children(node)
        if children is None:
            yield path, node, allowed
            continue
        for key_path, child in reversed(children):
            if child is None:
                continue
            name = getattr(key_path[0], "name", None) if key_path else None
            stack.append((path + key_path, child, allowed or name in fields))


def _violation(path, leaf, allowed: bool) -> LeafViolation | None:
    """Return the violation of one leaf against the active precision, or None."""
    name = jax.tree_util.keystr(path)
    kind = type(leaf).__name__
    if isinstance(leaf, np.ndarray | np.generic):
        return LeafViolation(name, kind, leaf.dtype.name, "NumPy leaf")
    if isinstance(leaf, _PYTHON_NUMBERS):
        return LeafViolation(name, kind, "", "Python number leaf")
    if not isinstance(leaf, jax.Array | jax.core.Tracer):
        return None
    if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        return None
    dtype = np.dtype(leaf.dtype)
    if getattr(leaf, "weak_type", False):
        return LeafViolation(name, kind, dtype.name, "weakly typed leaf")
    if np.issubdtype(dtype, np.complexfloating) and dtype != _complex_dtype():
        return LeafViolation(name, kind, dtype.name, "complex dtype is not active")
    if jnp.issubdtype(dtype, jnp.floating) and dtype != float_dtype():
        if not (allowed and dtype.itemsize < float_dtype().itemsize):
            return LeafViolation(name, kind, dtype.name, "float dtype is not active")
    return None


def _check(tree, enter, bulk) -> tuple[LeafViolation, ...]:
    bulk = frozenset(bulk)
    found = (_violation(path, leaf, ok) for path, leaf, ok in _walk(tree, enter, bulk))
    return tuple(v for v in found if v is not None)


def check_tree(
    tree: Any, *, bulk: Iterable[str] = (), owned: Iterable[str] = ()
) -> tuple[LeafViolation, ...]:
    """Return the leaves of a tree that break the leaf rule.

    A violation is a NumPy array or scalar, a Python number, a weakly typed
    array, or a floating (float16 and bfloat16 included) or complex array
    whose dtype is not the active one. A typed PRNG key is a valid leaf.
    A leaf under a field declared with ``as_bulk_leaf``, or at a path named in
    ``bulk``, may keep a narrower float dtype. The walk enters a nested Module
    only when it is owned (:func:`is_owned` over :data:`OWNED_PACKAGES` and
    ``owned``); the root is always checked.

    Args:
        tree: Any pytree, usually a model.
        bulk: Key paths (``jax.tree_util.keystr`` form) of waived bulk cubes.
        owned: Extra package names whose Modules are entered.

    Returns:
        The violations, in tree order; empty when the tree follows the rule.
    """
    packages = OWNED_PACKAGES | frozenset(owned)
    return _check(tree, lambda module: is_owned(module, packages), bulk)


def _is_foreign(module, foreign: frozenset[str]) -> bool:
    if isinstance(module, eqx.Partial):
        return False
    name = type(module).__module__
    return any(name == f or name.startswith(f + ".") for f in foreign)


def require_tree(
    tree: Any, *, bulk: Iterable[str] = (), foreign: Iterable[str] = ()
) -> None:
    """Refuse a tree with any leaf that breaks the leaf rule.

    The entry check of a fit or an exposure build. It enters every Module
    except those defined in :data:`FOREIGN_PACKAGES` and ``foreign``, so a
    user's own Module is checked too.

    Args:
        tree: The model or scene to check.
        bulk: Key paths (``jax.tree_util.keystr`` form) of waived bulk cubes.
        foreign: Extra third-party package names whose Modules are skipped.

    Raises:
        TypeError: If any leaf breaks the rule, naming each path.
    """
    skip = FOREIGN_PACKAGES | frozenset(foreign)
    found = _check(tree, lambda module: not _is_foreign(module, skip), bulk)
    if found:
        lines = "\n".join(
            f"  {v.path or '<root>'}: {v.type}"
            + (f" {v.dtype}" if v.dtype else "")
            + f" ({v.reason})"
            for v in found
        )
        raise TypeError(
            "the model holds leaves that are not JAX arrays of the active "
            "precision; build it through its constructors, or pass replacement "
            f"values through as_float_leaf or as_leaf:\n{lines}"
        )


def _is_number(x) -> bool:
    if isinstance(x, jax.Array) and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        return False
    return is_data(x) or isinstance(x, _PYTHON_NUMBERS)


def canonical(tree: Any, *, device: Any = None, foreign: Iterable[str] = ()) -> Any:
    """Map every numeric leaf of a tree through :func:`as_leaf`.

    For the ``eqx.tree_at`` and table I/O boundaries. Floats go to the active
    dtype, integers and booleans keep their kind, Python numbers become
    arrays. A leaf under a field declared with ``as_bulk_leaf`` goes through
    that converter. Modules of third-party packages (:data:`FOREIGN_PACKAGES`
    and ``foreign``) are returned unchanged.

    Args:
        tree: Any pytree.
        device: Device or sharding to place the leaves on; the default device
            when None. Given, each leaf is converted on the host (on that
            device when it is a single device, else on the first CPU device)
            and placed once, so nothing is staged on the default device.
        foreign: Extra third-party package names whose Modules are kept as is.

    Returns:
        The tree with canonical leaves.
    """
    skip = FOREIGN_PACKAGES | frozenset(foreign)

    def stop(x):
        return isinstance(x, eqx.Module) and _is_foreign(x, skip)

    bulk = {
        path
        for path, _, allowed in _walk(tree, lambda m: not stop(m), frozenset())
        if allowed
    }

    if device is None:
        staging = contextlib.nullcontext()
    else:
        if isinstance(device, jax.Device):
            stage = device
        else:
            stage = min(device.device_set, key=lambda d: d.id)
        staging = jax.default_device(stage)

    def convert(path, leaf):
        if stop(leaf) or not _is_number(leaf):
            return leaf
        with staging:
            out = as_bulk_leaf(leaf) if path in bulk else as_leaf(leaf)
        return out if device is None else jax.device_put(out, device)

    return jax.tree_util.tree_map_with_path(convert, tree, is_leaf=stop)


class HostGrid:
    """An immutable float64 host array, hashed and compared by its digest.

    For data that a host-built operator or a static layout depends on, such as
    a native wavenumber grid: store it on a ``static=True`` field. It is never
    a leaf, so no fit can free it and no digest of it can go stale. Equality
    tests identity first and then the full SHA-256 of the float64 bytes, never
    the array, so ``eqx.filter_jit`` pays a constant cost per dispatch for it.
    ``-0.0`` is stored as ``0.0``.

    Construct it on the host from any concrete array-like; a tracer is
    refused. ``values`` is the read-only float64 array and ``digest`` the
    first 16 hex digits of its SHA-256.
    """

    __slots__ = ("_sha", "_values")

    def __init__(self, values: Any):
        """Copy concrete values into a read-only float64 array.

        Args:
            values: A concrete, finite 1D array-like.

        Raises:
            TypeError: If ``values`` is traced.
            ValueError: If ``values`` is not a finite 1D array.
        """
        if isinstance(values, HostGrid):
            array = values._values
        else:
            if is_traced(values):
                raise TypeError(
                    f"a {type(self).__name__} must be concrete; build it on the "
                    "host, outside any traced function"
                )
            array = np.array(values, dtype=np.float64) + 0.0
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"a {type(self).__name__} must be a finite 1D array")
        array.setflags(write=False)
        object.__setattr__(self, "_values", array)
        object.__setattr__(self, "_sha", hashlib.sha256(array.tobytes()).digest())

    def __setattr__(self, name, value) -> None:
        """Refuse assignment: the grid is immutable.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name) -> None:
        """Refuse deletion: the grid is immutable.

        Args:
            name: Attribute name.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self):
        """Pickle by value, rebuilding through the constructor.

        Returns:
            The class and its constructor arguments.
        """
        return type(self), (self._values,)

    @property
    def values(self) -> np.ndarray:
        """The read-only float64 array."""
        return self._values

    @property
    def digest(self) -> str:
        """The first 16 hex digits of the SHA-256 of the float64 values."""
        return self._sha.hex()[:16]

    def array(self, dtype: Any = None) -> jax.Array:
        """Return the values as a JAX constant, by default in the active dtype.

        Not memoized, so a value made under a trace is never kept.

        Args:
            dtype: Dtype of the result; ``float_dtype()`` when None.

        Returns:
            The values as a ``jax.Array``.
        """
        return jnp.asarray(
            self._values, dtype=float_dtype() if dtype is None else dtype
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the values for ``np.asarray``, read-only unless copied.

        Args:
            dtype: Requested dtype; the stored float64 when None.
            copy: NumPy's copy request: True always copies, False never does.

        Returns:
            The values as a NumPy array.

        Raises:
            ValueError: If ``copy`` is False and ``dtype`` needs a cast.
        """
        if dtype is None or np.dtype(dtype) == self._values.dtype:
            return self._values.copy() if copy else self._values
        if copy is False:
            raise ValueError("changing the dtype of a host grid needs a copy")
        return self._values.astype(dtype)

    def __len__(self) -> int:
        """Return the number of values.

        Returns:
            The size of the array.
        """
        return self._values.size

    def __eq__(self, other) -> bool:
        """Compare by identity, then by the full SHA-256; never by array.

        Args:
            other: The object compared.

        Returns:
            Whether both are grids of one class with equal digests, or
            ``NotImplemented`` for another type.
        """
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._sha == other._sha

    def __hash__(self) -> int:
        """Hash the full SHA-256.

        Returns:
            The hash.
        """
        return hash(self._sha)

    def __repr__(self) -> str:
        """Show the size, the ends and the digest, not every value.

        Returns:
            The representation.
        """
        v = self._values
        span = f"{float(v[0])!r} to {float(v[-1])!r}" if v.size else "empty"
        return f"{type(self).__name__}({v.size} values, {span}, {self.digest})"
