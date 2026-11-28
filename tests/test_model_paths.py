"""Dotted model paths: ``[k]`` indexes a tuple or list, never an array axis."""

import re
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hwoutils.leaves import as_float_leaf
from hwoutils.model_paths import (
    ancestors,
    apply_at,
    assign,
    assign_many,
    canonical_path,
    display_path,
    resolve,
    scale_at,
)
from hwoutils.testing import ulps


class _Planet(eqx.Module):
    radius: jnp.ndarray


class _Model(eqx.Module):
    planets: tuple
    names: list


def _model():
    return _Model(
        planets=(_Planet(jnp.array([1.0, 2.0])), _Planet(jnp.array([3.0, 4.0]))),
        names=["a", "b"],
    )


def test_resolve_indexes_tuple_and_list():
    """Fails if a tuple or list element cannot be read through ``[k]``."""
    model = _model()
    assert resolve(model, "planets[1].radius") is model.planets[1].radius
    assert resolve(model, "names[1]") == "b"


def test_assign_replaces_tuple_element():
    """Fails if assign does not replace the addressed node only."""
    new = assign(_model(), "planets[0].radius", jnp.array([9.0, 9.0]))
    assert float(new.planets[0].radius[0]) == 9.0
    assert float(new.planets[1].radius[0]) == 3.0


@pytest.mark.parametrize("path", ["planets[0].radius[0]", "planets[0].radius[1]"])
def test_array_axis_index_is_refused_with_the_path(path):
    """Fails if an array-axis index is accepted by resolve or assign."""
    with pytest.raises(ValueError, match="tuple or list, not an array axis") as err:
        resolve(_model(), path)
    assert path in str(err.value)
    with pytest.raises(ValueError, match="tuple or list, not an array axis"):
        assign(_model(), path, jnp.asarray(0.0))


class _Shift(eqx.Module):
    shift: jax.Array = eqx.field(converter=as_float_leaf)


class _Stage(eqx.Module):
    name: str | int = eqx.field(static=True)
    op: _Shift


class _Group(NamedTuple):
    entrance: _Stage
    stop: _Stage
    exit: _Stage

    @property
    def name(self):
        return "inner"


class _NamedModel(eqx.Module):
    stages: tuple
    groups: tuple
    invalid_names: list
    unnamed: tuple
    count: jax.Array
    optional: object = None
    spare: object = None


@pytest.fixture(scope="module")
def named_model():
    return _NamedModel(
        stages=(
            _Stage("clear", _Shift(1.0)),
            _Stage("mask", _Shift(2.0)),
            _Stage("mask_extra", _Shift(3.0)),
        ),
        groups=(
            (_Shift(4.0),),
            _Group(
                _Stage("a", _Shift(5.0)),
                _Stage("b", _Shift(6.0)),
                _Stage("c", _Shift(7.0)),
            ),
        ),
        invalid_names=[
            _Stage(name, _Shift(8.0)) for name in ("bad name", "3", "dup", "dup", "", 7)
        ],
        unnamed=(_Shift(9.0),),
        count=jnp.asarray(3, dtype="int32"),
    )


def test_named_resolution_preserves_element_identity(named_model):
    """Fails if a name resolves by prefix or unwraps an element.

    Truth: exact object identity to the fixture's own nodes and prefixes.
    """
    model = named_model
    assert resolve(model, "stages[mask]") is model.stages[1]
    assert resolve(model, "stages[1]") is model.stages[1]
    assert resolve(model, "stages[mask].op.shift") is model.stages[1].op.shift
    assert resolve(model, "groups[inner][2].op") is model.groups[1][2].op
    path = "groups[inner][2].op.shift"
    assert tuple(ancestors(model, path))[-1][0] == "groups[inner][2].op"
    assert tuple(ancestors(model, path))[-1][1] is model.groups[1][2].op


def test_name_key_refusals_and_numeric_precedence(named_model):
    """Fails if invalid names resolve or digit-only keys address names.

    Truth: exact grammar, case, ambiguity and lookup error contract.
    """
    model = named_model
    missing = "stages[missing]"
    with pytest.raises(ValueError, match=re.escape(missing)) as error:
        resolve(model, missing)
    assert all(name in str(error.value) for name in ("clear", "mask", "mask_extra"))
    duplicate = "invalid_names[dup]"
    with pytest.raises(ValueError, match=re.escape(duplicate) + ".*ambiguous"):
        resolve(model, duplicate)
    for path in (
        "stages[mask].op.shift[mask]",
        "unnamed[clear]",
        "stages[]",
        "invalid_names[bad name]",
        "stages[MASK]",
    ):
        with pytest.raises(ValueError, match=re.escape(path)):
            resolve(model, path)
    assert resolve(model, "invalid_names[0]") is model.invalid_names[0]
    assert model.invalid_names[1].name == "3"
    assert resolve(model, "invalid_names[3]") is model.invalid_names[3]


def test_canonical_path_normalizes_aliases_with_traced_leaves(named_model):
    """Fails if aliases remain in the canonical spelling or tracing reads a value.

    Truth: exact paths written by hand and the corresponding fixture leaf.
    """
    path = "groups[inner][02].op.shift"
    expected = "groups[1][2].op.shift"
    assert canonical_path(named_model, path) == expected
    assert canonical_path(named_model, expected) == expected
    assert canonical_path(named_model, "stages[01].op.shift") == "stages[1].op.shift"

    @jax.jit
    def resolve_canonical(model):
        canonical = canonical_path(model, path)
        assert canonical == expected
        return resolve(model, canonical)

    assert resolve_canonical(named_model) == named_model.groups[1][2].op.shift


def test_display_path_round_trips_all_fixture_leaves(named_model):
    """Fails if display paths use an invalid or ambiguous name.

    Truth: exact canonical-display round trip for each leaf.
    """
    for keys, _ in jax.tree_util.tree_leaves_with_path(named_model):
        path = jax.tree_util.keystr(keys).removeprefix(".")
        displayed = display_path(named_model, path)
        assert canonical_path(named_model, displayed) == canonical_path(
            named_model, path
        )
    assert display_path(named_model, "stages[01].op.shift") == "stages[mask].op.shift"
    for index in range(len(named_model.invalid_names)):
        path = f"invalid_names[{index}].op.shift"
        assert display_path(named_model, path) == path


def test_assign_preserves_raw_values_and_applies_an_opt_in_field_converter(
    named_model, precision
):
    """Fails if raw writes cast values or conversion runs for a sequence member.

    Truth: exact Python type and the dtype of the fixture's field converter.
    """
    path = "stages[mask].op.shift"
    raw = resolve(assign(named_model, path, 5.0), path)
    assert type(raw) is float and raw == 5.0
    converted = resolve(assign(named_model, path, 5.0, convert=True), path)
    assert isinstance(converted, jax.Array)
    assert converted.dtype == precision and not converted.weak_type and converted == 5.0
    replaced = assign(named_model, "stages[mask]", 5.0, convert=True)
    assert type(replaced.stages[1]) is float and replaced.stages[1] == 5.0
    applied = apply_at(named_model, path, lambda value: float(value) * 2)
    assert type(resolve(applied, path)) is float and resolve(applied, path) == 4.0


def test_assign_populates_and_clears_a_structural_none(named_model):
    """Fails if None cannot be selected or a second slot is also updated.

    Truth: exact equality of the untouched tree and the restored tree.
    """
    value = named_model.stages[1].op.shift
    populated = assign(named_model, "optional", value)
    assert populated.optional is value and populated.spare is None
    assert populated.stages[1].op.shift is value
    assert eqx.tree_equal(assign(populated, "optional", None), named_model)


def test_assign_many_validates_and_updates_against_the_original_tree(named_model):
    """Fails if aliases overlap or one write changes another write's selector.

    Truth: exact sequential parity for disjoint writes, explicit original
    indices for interacting names, and exact overlap and atomicity contracts.
    """
    model = named_model
    first, second = "stages[clear].op.shift", "stages[mask].op.shift"
    assert eqx.tree_equal(
        assign_many(model, {first: 11.0, second: 12.0}),
        assign(assign(model, first, 11.0), second, 12.0),
    )
    assert assign_many(model, {}) is model
    assert assign_many(model, {"": 13.0}) == 13.0
    converted = assign_many(model, {first: 11.0, "optional": 12.0}, convert=True)
    assert resolve(converted, first).dtype == resolve(model, first).dtype
    assert converted.optional == 12.0 and converted.spare is None
    for a, b in (
        ("stages[1]", "stages[mask]"),
        ("stages[01]", "stages[1]"),
        ("stages[mask]", second),
        ("", first),
    ):
        with pytest.raises(ValueError) as error:
            assign_many(model, {a: 1.0, b: 2.0})
        assert repr(a) in str(error.value) and repr(b) in str(error.value)

    original = resolve(model, first)
    with pytest.raises(ValueError, match=re.escape("stages[]")):
        assign_many(model, {first: 11.0, "stages[]": 12.0})
    assert resolve(model, first) is original

    # This replacement introduces a second "mask"; the original selects index 1.
    replacement = _Stage("mask", model.stages[0].op)
    result = assign_many(model, {"stages[0]": replacement, second: 12.0})
    assert eqx.tree_equal(result.stages[0], replacement)
    assert result.stages[1].op.shift == 12.0


def test_scale_at_preserves_dtype_and_has_the_closed_form_gradient(
    named_model, precision
):
    """Fails if scaling widens a leaf or loses its sensitivity to the factor.

    Truth: product by two and derivative equal to the selected leaf, one ULP.
    """
    path = "stages[mask].op.shift"
    expected = resolve(named_model, path)

    def scaled(factor):
        return resolve(scale_at(named_model, path, factor), path)

    factor = jnp.asarray(2.0, dtype=precision)
    product = scaled(factor)
    assert (
        product.dtype == precision and not product.weak_type and product == 2 * expected
    )
    grad = jax.grad(scaled)(factor)
    assert grad.dtype == precision
    assert abs(grad - expected) <= ulps(1, expected)

    with jax.enable_x64(False):
        weak_float32 = jnp.asarray(2.0)
    narrow = assign(named_model, path, weak_float32)
    product = resolve(scale_at(narrow, path, factor), path)
    assert product.dtype == jnp.float32 and not product.weak_type and product == 4.0


def test_scale_at_refuses_non_inexact_leaves(named_model):
    """Fails if integer leaves, empty slots or modules can be scaled.

    Truth: exact invalid-target type contract.
    """
    for path in ("count", "optional", "stages[mask].op"):
        with pytest.raises(TypeError, match=re.escape(path)):
            scale_at(named_model, path, 2.0)
    with pytest.raises(TypeError, match="real scale factor"):
        scale_at(named_model, "stages[mask].op.shift", 1j)
