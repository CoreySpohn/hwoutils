"""Contract tests for the leaf converters, checks, host grid and pytest plugin.

Truth sources are dtype and type identities and the documented integer ranges,
so every comparison is exact. Classes that stand in for an owned library or a
third-party package set ``__module__`` to a name in that package; the checks
decide ownership from module names, so the stand-ins exercise the same path.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils.jax_config import dtype_tag, float_dtype
from hwoutils.leaves import (
    HostGrid,
    as_bulk_leaf,
    as_complex_leaf,
    as_float_leaf,
    as_leaf,
    canonical,
    check_tree,
    is_owned,
    require_tree,
)

pytest_plugins = ["pytester"]


@pytest.fixture(params=[False, True], ids=["x32", "x64"])
def x64(request):
    """Run the test under each precision, scoped like a host build."""
    with jax.enable_x64(request.param):
        yield request.param


class Held(eqx.Module):
    """An owned Module whose single field stores what it is given."""

    __module__ = "skyscapes._standin"
    value: object


class Cube(eqx.Module):
    """An owned Module with a declared bulk field."""

    __module__ = "skyscapes._standin"
    cube: jax.Array = eqx.field(converter=as_bulk_leaf)


class ForeignInterpolator(eqx.Module):
    """Stands in for a third-party interpolator holding a Python bool leaf."""

    __module__ = "interpax._standin"
    x: jax.Array
    extrap: bool


class OwnedBase(eqx.Module):
    """Stands in for an owned abstract class."""

    __module__ = "skyscapes._standin"
    value: eqx.AbstractVar[object]


class UserSubclass(OwnedBase):
    """A user's subclass of the owned abstract class, defined in user code."""

    value: object


class UserModule(eqx.Module):
    """A user's Module with no owned base."""

    value: object


def _strong(x, dtype):
    return isinstance(x, jax.Array) and x.dtype == dtype and not x.weak_type


def test_float_dtype_follows_scope(x64):
    """Exact dtype basis. Fails if the dtype is read once instead of per call."""
    assert float_dtype() == (np.float64 if x64 else np.float32)
    assert dtype_tag() == ("f64" if x64 else "f32")


def test_as_float_leaf_is_strong_and_active(x64):
    """Exact type and dtype basis (test a).

    Fails if a NumPy, list, NumPy scalar, Python int or weak input keeps its
    own dtype or weak type, or if complex input is accepted on a real field.
    """
    inputs = [
        np.array([1.0, 2.0]),
        [1.0, 2.0],
        np.float64(1.5),
        3,
        jnp.asarray(1.0),
        np.array([1.0, 2.0], dtype=np.float32),
    ]
    for value in inputs:
        assert _strong(as_float_leaf(value), float_dtype()), value
    with pytest.raises(TypeError, match="complex"):
        as_float_leaf(np.array([1 + 1j]))
    assert _strong(
        as_complex_leaf([1 + 1j]), np.dtype(np.complex128 if x64 else np.complex64)
    )


def test_as_leaf_keeps_kind_and_refuses_wrap(x64):
    """Exact kind basis and the int32 range (test b).

    Fails if integers or booleans become floats, or if 3e9 is wrapped to
    -1294967296 under x32 instead of refused.
    """
    assert _strong(as_leaf(np.array([1, 2])), np.dtype(np.int64 if x64 else np.int32))
    assert _strong(as_leaf(np.array([True])), np.dtype(np.bool_))
    assert _strong(as_leaf(True), np.dtype(np.bool_))
    assert _strong(as_leaf(jnp.asarray(2)), np.dtype(np.int64 if x64 else np.int32))
    assert _strong(as_leaf(np.array([1.5])), float_dtype())
    big = np.array([3_000_000_000], dtype=np.int64)
    if x64:
        assert int(as_leaf(big)[0]) == 3_000_000_000
    else:
        with pytest.raises(ValueError, match="do not fit int32"):
            as_leaf(big)
        with pytest.raises(ValueError, match="do not fit int32"):
            as_leaf(3_000_000_000)


def test_check_tree_finds_each_violation_once():
    """Exact count basis (test c).

    One violation each for a NumPy array, a NumPy scalar, a Python float, a
    weak scalar and a float32 leaf under x64; none after ``canonical``.
    Fails if any kind is missed or counted twice, or if ``canonical`` leaves
    one behind.
    """
    with jax.enable_x64(True):
        bad = [
            np.ones(2),
            np.float64(1.0),
            1.0,
            jnp.asarray(1.0),
            jnp.ones(2, dtype=jnp.float32),
        ]
        reasons = [
            "NumPy leaf",
            "NumPy leaf",
            "Python number leaf",
            "weakly typed leaf",
            "float dtype is not active",
        ]
        for value, reason in zip(bad, reasons, strict=True):
            found = check_tree(Held(value))
            assert [v.reason for v in found] == [reason], value
            assert found[0].path == ".value"
            assert check_tree(canonical(Held(value))) == ()


def test_bulk_field_keeps_float32_only_under_x64(x64):
    """Exact dtype basis. Fails if a declared bulk cube is widened or flagged."""
    cube = Cube(np.ones((2, 2), dtype=np.float32))
    assert cube.cube.dtype == np.float32
    assert check_tree(cube) == ()
    assert canonical(cube).cube.dtype == np.float32
    held = Held(jnp.ones(2, dtype=jnp.float32))
    assert len(check_tree(held)) == (1 if x64 else 0)
    assert check_tree(held, bulk=(".value",)) == ()


def test_foreign_modules_are_not_entered():
    """Exact count basis (test c, third-party boundary).

    Fails if the check walks into a third-party Module (its Python bool leaf
    would be reported), or if ``canonical`` converts that leaf.
    """
    inner = ForeignInterpolator(jnp.ones(2), True)
    model = Held(inner)
    assert check_tree(model) == ()
    require_tree(model)
    assert canonical(model).value.extrap is True


def test_ownership_is_decided_by_the_mro():
    """Exact count basis (test c, subclass boundary).

    A user's subclass of an owned abstract class is owned, so the check
    enters it inside an owned model and reports its NumPy leaf; a user Module
    with no owned base is not entered by ``check_tree`` but is refused by
    ``require_tree``. Fails if ownership is read from the class's own package.
    """
    sub = UserSubclass(np.ones(2))
    assert is_owned(sub)
    assert [v.path for v in check_tree(Held(sub))] == [".value.value"]
    user = UserModule(np.ones(2))
    assert not is_owned(user)
    assert check_tree(Held(user)) == ()
    with pytest.raises(TypeError, match=r"\.value\.value: ndarray float64"):
        require_tree(Held(user))


def test_host_grid_identity():
    """Exact digest basis (test e).

    Equal arrays compare and hash equal; one changed bit does not; the array
    is read-only and a tracer is refused. Fails if equality compares arrays
    loosely, if the stored array can be written, or if traced knots are kept.
    """
    values = np.array([400.0, 450.1, 520.3, 700.0])
    a, b = HostGrid(values), HostGrid(values.tolist())
    assert a == b and hash(a) == hash(b) and a.digest == b.digest
    moved = values.copy()
    moved[1] = np.nextafter(moved[1], np.inf)
    assert HostGrid(moved) != a
    assert HostGrid([-0.0, 1.0]) == HostGrid([0.0, 1.0])
    with pytest.raises(ValueError, match="read-only"):
        a.values[0] = 1.0
    values[0] = 1.0
    assert a.values[0] == 400.0
    with pytest.raises(TypeError, match="concrete"):
        jax.jit(lambda x: HostGrid(x).digest)(jnp.ones(2))


_PLUGIN_TESTS = """
import numpy as np
import pytest
import equinox as eqx


class Raw(eqx.Module):
    value: object


def library_call():
    try:
        return Raw(np.ones(2))
    except ValueError:
        return None


def test_inside_raises():
    with pytest.raises(ValueError, match="expected"):
        Raw(np.ones(2))
        raise ValueError("expected")


def test_inside_library_try():
    assert library_call() is not None
"""


def test_plugin_records_and_never_raises(pytester, monkeypatch):
    """Exact count basis (test d).

    A raw NumPy field built inside ``pytest.raises(ValueError)`` and inside a
    library ``try/except ValueError`` is recorded for each test, and neither
    test fails. Fails if the check raises during construction.
    """
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest('pytest_plugins = ["hwoutils.leaves_contract"]')
    pytester.makeini("[pytest]\nleaf_contract_packages = test_plugin_cases\n")
    pytester.makepyfile(test_plugin_cases=_PLUGIN_TESTS)
    report = pytester.path / "leaves.json"
    result = pytester.runpytest_inprocess(f"--leaf-report={report}")
    result.assert_outcomes(passed=2)
    data = json.loads(report.read_text())
    assert [(v["path"], v["reason"]) for v in data["violations"]] == [
        (".value", "NumPy leaf")
    ]
    assert data["violations"][0]["constructions"] == 2
    assert sorted(data["tests"]) == [
        "test_plugin_cases.py::test_inside_library_try",
        "test_plugin_cases.py::test_inside_raises",
    ]
    result.stdout.fnmatch_lines(["*test_plugin_cases.Raw.value: ndarray float64*x2"])
