"""Model paths and the block base: reading, writing, refusal and order."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from hwoutils.blocks import AbstractBlock, block_order, check_writers
from hwoutils.model_paths import assign, join, paths_overlap, resolve


class _Star(eqx.Module):
    mass_Msun: jnp.ndarray


class _Model(eqx.Module):
    star: _Star
    planets: tuple


def _model():
    return _Model(_Star(jnp.asarray(1.0)), (jnp.asarray(2.0), jnp.asarray(3.0)))


def test_resolve_reads_attributes_and_tuple_indices():
    """Fails if a dotted path or a tuple index is read wrongly."""
    model = _model()
    assert resolve(model, "star.mass_Msun") == 1.0
    assert resolve(model, "planets[1]") == 3.0
    assert resolve(model, "star") is model.star


def test_resolve_refuses_malformed_paths():
    """Fails if a path with an empty step or a bare index is accepted."""
    for path in ("star..mass_Msun", "[0]", "planets[x]", ".star"):
        with pytest.raises(ValueError, match="malformed"):
            resolve(_model(), path)


def test_assign_replaces_one_node_and_leaves_the_rest():
    """Fails if assign mutates the model or moves another leaf."""
    model = _model()
    new = assign(assign(model, "star.mass_Msun", jnp.asarray(5.0)), "planets[0]", 7.0)
    assert new.star.mass_Msun == 5.0 and new.planets[0] == 7.0
    assert new.planets[1] == 3.0 and model.star.mass_Msun == 1.0


def test_paths_overlap_matches_whole_nodes_only():
    """Fails if a path prefix is confused with a longer name."""
    assert paths_overlap("dynamics.orbits", "dynamics.orbits")
    assert paths_overlap("dynamics.orbits", "dynamics.orbits[0].ex")
    assert paths_overlap("dynamics.frame.dist_pc", "dynamics.frame")
    assert not paths_overlap("dynamics.orbits", "dynamics.orbits_extra")
    assert not paths_overlap("planets[0].Rp_Rearth", "planets[1].Rp_Rearth")


class _Block(AbstractBlock):
    owns: str = eqx.field(static=True)
    reads: tuple[str, ...] = eqx.field(static=True, default=())
    writes: tuple[str, ...] = eqx.field(static=True, default=())
    coordinate_names: tuple[str, ...] = eqx.field(static=True, default=("a", "b"))

    def forward(self, x, model):
        return {}

    def inverse(self, model):
        return {}

    def log_prob(self, x, model):
        return jnp.zeros(())

    def sample(self, key, model):
        return {}

    def same_map(self, other):
        return True


def _orbit():
    return _Block("dynamics.orbits", ("dynamics.star.mass_Msun",), ("dynamics.orbits",))


def _mass():
    return _Block("dynamics.star.mass_Msun", (), ("dynamics.star.mass_Msun",))


def test_a_block_must_define_its_contract():
    """Fails if a block missing a method or a declaration can be built."""

    class Incomplete(AbstractBlock):
        owns: str = "a"

    with pytest.raises(TypeError):
        Incomplete()


def test_two_writers_of_one_path_are_refused():
    """Fails if two entries may write one path, or a key is not the owned node."""
    orbit, mass = _orbit(), _mass()
    check_writers({"dynamics.orbits": orbit})
    check_writers({"dynamics.orbits": orbit}, leaf_paths=["dynamics.star.mass_Msun"])
    with pytest.raises(ValueError, match=r"dynamics\.orbits.*two writers"):
        check_writers({"dynamics.orbits": orbit}, leaf_paths=["dynamics.orbits.ex"])
    two = _Block("dynamics.orbits", writes=("dynamics.orbits", mass.owns))
    with pytest.raises(ValueError, match=r"dynamics\.star\.mass_Msun.*two writers"):
        check_writers({"dynamics.orbits": two, mass.owns: mass})
    with pytest.raises(ValueError, match="must be keyed by the path it owns"):
        check_writers({"orbits": orbit})


def test_blocks_run_after_the_blocks_that_write_what_they_read():
    """Fails if a block would read a path before another block has written it."""
    orbit, mass = _orbit(), _mass()
    expected = (mass.owns, "dynamics.orbits")
    assert block_order({"dynamics.orbits": orbit, mass.owns: mass}) == expected
    assert block_order({mass.owns: mass, "dynamics.orbits": orbit}) == expected


def test_a_read_cycle_is_refused():
    """Fails if two blocks reading each other's writes are ordered."""
    a = _Block("a", ("b",), ("a",))
    b = _Block("b", ("a",), ("b",))
    with pytest.raises(ValueError, match="cycle"):
        block_order({"a": a, "b": b})


def test_a_block_defaults_to_non_periodic_with_no_unit_map():
    """Fails if a plain block claims a period or a unit-cube map it does not have."""
    block = _orbit()
    assert block.periods == (None, None)
    assert block.unit_size is None
    with pytest.raises(NotImplementedError, match="no unit-cube map"):
        block.from_unit(jnp.zeros(1), _model())


def test_periods_are_static_and_aligned_with_the_coordinate_names():
    """Fails if a subclass cannot declare periods aligned with its names."""

    class Wrapped(_Block):
        @property
        def periods(self):
            return (None, 6.283185307179586)

    block = Wrapped("a", coordinate_names=("x", "phase"))
    assert dict(zip(block.coordinate_names, block.periods, strict=True)) == {
        "x": None,
        "phase": 6.283185307179586,
    }


def test_the_empty_path_is_the_model_and_overlaps_everything():
    """Fails if the empty path does not read, replace or overlap as the model root."""
    model = _model()
    assert resolve(model, "") is model
    assert assign(model, "", 5) == 5
    assert paths_overlap("", "star.mass_Msun") and paths_overlap("planets[0]", "")
    assert paths_overlap("", "")


def test_join_handles_empty_and_indexed_segments():
    """Fails if a join leaves a stray dot or drops a segment."""
    assert join("", "star.mass_Msun") == "star.mass_Msun"
    assert join("scene", "") == "scene"
    assert join("", "") == ""
    assert join("planets", "[1].mass") == "planets[1].mass"
    assert join("scene", "star.mass") == "scene.star.mass"
    model = _model()
    assert resolve(model, join("", join("planets", "[1]"))) == 3.0
