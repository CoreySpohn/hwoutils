"""Dotted model paths: ``[k]`` indexes a tuple or list, never an array axis."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from hwoutils.model_paths import assign, resolve


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
