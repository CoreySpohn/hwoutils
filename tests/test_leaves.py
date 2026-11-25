"""Contract tests for the leaf converters, checks, host grid and pytest plugin.

Truth sources are dtype and type identities and the documented integer ranges,
so every comparison is exact. Classes that stand in for an owned library or a
third-party package set ``__module__`` to a name in that package; the checks
decide ownership from module names, so the stand-ins exercise the same path.
"""

import json
import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import PrecisionError, like, precision_of
from hwoutils.jax_config import dtype_tag, float_dtype
from hwoutils.leaves import (
    HostGrid,
    as_bulk_leaf,
    as_complex_leaf,
    as_float_leaf,
    as_leaf,
    canonical,
    check_tree,
    is_foreign,
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
        jnp.float16(1),
        jnp.bfloat16(1),
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
    weak scalar; none after ``canonical``.
    Fails if any kind is missed or counted twice, or if ``canonical`` leaves
    one behind.
    """
    with jax.enable_x64(True):
        bad = [
            np.ones(2),
            np.float64(1.0),
            1.0,
            jnp.asarray(1.0),
        ]
        reasons = [
            "NumPy leaf",
            "NumPy leaf",
            "Python number leaf",
            "weakly typed leaf",
        ]
        for value, reason in zip(bad, reasons, strict=True):
            found = check_tree(Held(value))
            assert [v.reason for v in found] == [reason], value
            assert found[0].path == ".value"
            assert check_tree(canonical(Held(value))) == ()


def test_numpy_leaf_in_float32_model(x64):
    """The NumPy violation reaches the entry point with its full field path."""
    model = Held((jnp.float32(1), np.ones(2, dtype=np.float64)))
    assert [(v.path, v.reason) for v in check_tree(model)] == [
        (".value[1]", "NumPy leaf")
    ]
    with pytest.raises(TypeError, match=r"\.value\[1\].*\(NumPy leaf\)"):
        require_tree(model)


def test_unavailable_strong_precision():
    """A wide array built in an earlier scope needs x64 on reconstruction."""
    with jax.enable_x64(True):
        real, field = jnp.float64(1), jnp.complex128(1j)
    with jax.enable_x64(False):
        for convert, value in [
            (as_float_leaf, real),
            (as_complex_leaf, real),
            (as_complex_leaf, field),
        ]:
            with pytest.raises(PrecisionError, match="x64"):
                convert(value)


def test_converter_preserves_strong_arrays_and_tracers(x64):
    """Strong fields retain their dtype when an Equinox model is rebuilt."""
    for fn, dtype in [(as_float_leaf, jnp.float32), (as_complex_leaf, jnp.complex64)]:
        value = jnp.ones(2, dtype=dtype)
        assert _strong(fn(value), dtype)
        assert _strong(jax.jit(fn)(value), dtype)
    assert _strong(as_complex_leaf(jnp.float32(3)), np.complex64)
    with pytest.raises(TypeError, match="complex"):
        jax.jit(lambda x: as_float_leaf([x, x]))(jnp.complex64(1j))
    for dtype in ["int32", "unrecognized"]:
        with pytest.raises(PrecisionError, match="Expected float32 or float64"):
            canonical(1.0, precision=dtype)
    assert _strong(
        as_complex_leaf(np.complex64(3j)), np.complex128 if x64 else np.complex64
    )
    assert (
        jax.jit(lambda x: as_float_leaf([x, x]))(jnp.float32(1)).dtype == float_dtype()
    )


def test_bulk_reader_and_owner_precision(x64):
    """A narrow complex table keeps both components and follows its reader."""
    data = np.array([1.25 - 0.5j], np.complex64)
    cube = Cube(data)
    tree = (cube, as_float_leaf(2.0), HostGrid([1.0]))
    assert _strong(cube.cube, np.complex64)
    assert precision_of(tree) == float_dtype()
    assert check_tree(tree) == ()
    require_tree(tree)
    read = like(cube.cube, tree)
    assert _strong(read, np.complex128 if x64 else np.complex64)
    np.testing.assert_array_equal(read, data)
    mask = eqx.tree_at(lambda module: module.cube, cube, np.array([True]))
    assert _strong(canonical(mask).cube, np.bool_)
    for table in (data, data.real):
        packed = Cube(table)
        mixed = (packed, packed.cube)
        for requested in (None, float_dtype()):
            stored, working = canonical(mixed, precision=requested)
            assert _strong(stored.cube, table.dtype)
            assert working.dtype == (
                read.dtype if np.iscomplexobj(table) else float_dtype()
            )
            np.testing.assert_array_equal(stored.cube, table)
        if x64:
            wide = Cube(
                table.astype(np.complex128 if np.iscomplexobj(table) else np.float64)
            )
            with jax.enable_x64(False):
                assert canonical(wide).cube.dtype == wide.cube.dtype
            narrowed = canonical(wide, precision="float32").cube
            assert _strong(narrowed, table.dtype)
            np.testing.assert_array_equal(narrowed, table)


def test_mixed_tree_and_explicit_canonical_round_trip(x64):
    """Exactly representable inputs survive a narrow and wide round trip."""
    host = (np.array([1.25, -0.5]), np.array([-2.25 + 0.5j]), 7, True)
    narrow = canonical(host, precision="float32")
    assert [x.dtype.kind for x in narrow] == ["f", "c", "i", "b"]
    assert all(not x.weak_type for x in narrow)
    assert precision_of(narrow) == np.float32
    require_tree(narrow)
    if not x64:
        with pytest.raises(PrecisionError, match="x64"):
            canonical(narrow, precision="float64")
        return
    wide = canonical(narrow, precision="float64")
    assert _strong(wide[0], np.float64) and _strong(wide[1], np.complex128)
    for result, expected in zip(wide, host, strict=True):
        np.testing.assert_array_equal(result, expected)
    # Include the bottom of the float32 range, where a device cast can flush.
    tiny = float(np.finfo(np.float32).smallest_subnormal)
    values = np.array([complex(tiny, -tiny), complex(-0.0, 3.5)], np.complex128)
    for cast in [canonical, jax.jit(canonical, static_argnames="precision")]:
        packed = cast(jnp.asarray(values), precision="float32")
        restored = cast(packed, precision="float64")
        np.testing.assert_array_equal(
            np.asarray(restored).view(np.uint64), values.view(np.uint64)
        )
    assert jax.grad(lambda x: canonical(x, precision="float64"))(jnp.float32(0)) == 1
    mixed = (narrow[0], wide[1])
    for check in [precision_of, check_tree, require_tree]:
        with pytest.raises(PrecisionError, match=r"canonical\(tree, precision="):
            check(mixed)
    require_tree(mixed, bulk=("[0]",))
    assert check_tree(mixed, bulk=("[0]",)) == ()


def test_range_refusal_checks_each_component(x64):
    """A representable component cannot mask overflow or loss of the other."""
    for value, message in [(1e100, "do not fit"), (1e-100, "underflow")]:
        for data in [np.array([value]), np.array([complex(3, value)])]:
            inputs = [data, jnp.asarray(data)] if x64 else [data]
            if x64:
                with pytest.raises(ValueError, match=message):
                    canonical(Cube(data), precision="float32")
            for value_to_cast in inputs:
                with pytest.raises(ValueError, match=message):
                    canonical(value_to_cast, precision="float32")
            if not x64:
                for cast in [as_bulk_leaf, as_complex_leaf]:
                    with pytest.raises(ValueError, match=message):
                        cast(data)


def test_foreign_modules_are_not_entered():
    """Exact count basis (test c, third-party boundary).

    Fails if the check walks into a third-party Module (its Python bool leaf
    would be reported), or if ``canonical`` converts that leaf.
    """
    inner = ForeignInterpolator(jnp.ones(2), True)
    model = Held(inner)
    assert check_tree(model) == ()
    require_tree(model)
    assert canonical(model).value is inner
    assert is_foreign(inner)
    assert not is_foreign(eqx.Partial(print, 1.0))
    assert not is_foreign(jnp.asarray(1.0))
    custom = UserModule(np.ones(2))
    assert not is_foreign(custom)
    assert is_foreign(custom, [__name__])
    require_tree(custom, foreign=[__name__])
    assert canonical(custom, foreign=[__name__]) is custom


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
    assert np.asarray(a) is a.values
    assert np.asarray(a, dtype=np.float32).dtype == np.float32
    with pytest.raises(TypeError, match="concrete"):
        jax.jit(lambda x: HostGrid(x).digest)(jnp.ones(2))


_PLUGIN_TESTS = """
import jax
import jax.numpy as jnp
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


def test_mixed_fields():
    with jax.enable_x64(True):
        Raw((jnp.float32(1), jnp.float64(2)))
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
    result.assert_outcomes(passed=3)
    data = json.loads(report.read_text())
    assert not data["errors"]
    mixed, numpy_leaf = data["violations"]
    assert mixed["path"] == "" and "canonical(tree, precision=" in mixed["reason"]
    assert (numpy_leaf["path"], numpy_leaf["reason"], numpy_leaf["constructions"]) == (
        ".value",
        "NumPy leaf",
        2,
    )
    assert sorted(data["tests"]) == [
        "test_plugin_cases.py::test_inside_library_try",
        "test_plugin_cases.py::test_inside_raises",
        "test_plugin_cases.py::test_mixed_fields",
    ]
    result.stdout.fnmatch_lines(["*test_plugin_cases.Raw.value: ndarray float64*x2"])


class Keyed(eqx.Module):
    key: jax.Array
    value: jax.Array


def test_typed_key_is_a_valid_leaf(x64):
    """Exact count basis. Fails if a typed PRNG key crashes or is flagged."""
    model = Keyed(jax.random.key(0), jnp.ones(2))
    assert check_tree(model) == ()
    require_tree(model)
    assert canonical(model).key.dtype == model.key.dtype


def test_equinox_modules_are_foreign():
    """Exact count basis. Fails if an ``eqx.nn`` module is entered: it holds a
    Python float and bool that the leaf rule would report."""
    held = Held(eqx.nn.Dropout(0.5))
    assert check_tree(held) == ()
    require_tree(held)
    assert canonical(held).value.p == 0.5


def test_as_float_leaf_refuses_overflow():
    """Exact basis: float32 holds up to 3.4028235e38. Fails if a finite
    float64 above it silently becomes inf under x32, or if x64, NaN, inf or a
    tracer is refused."""
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="do not fit float32"):
            as_float_leaf(np.array([1.0, 1e300]))
        assert np.isinf(as_float_leaf(np.array([np.inf]))[0])
        assert np.isnan(as_float_leaf(np.array([np.nan]))[0])
        assert as_float_leaf(1e30) == jnp.float32(1e30)
        assert jax.jit(as_float_leaf)(jnp.ones(2)).dtype == jnp.float32
    with jax.enable_x64(True):
        assert as_float_leaf(np.array([1e300]))[0] == 1e300


def test_equinox_partial_with_numpy_is_flagged():
    """Exact count basis. Fails if all of equinox is exempt: an ``eqx.Partial``
    holding a NumPy array must be refused by ``require_tree`` (``check_tree``
    enters owned modules only), while ``eqx.nn.Dropout`` and the wrappers
    ``eqx.filter_grad`` and ``eqx.filter_jit`` stay accepted."""
    held = Held(eqx.Partial(print, np.ones(2)))
    with pytest.raises(TypeError, match="ndarray"):
        require_tree(held)
    require_tree(Held(eqx.filter_grad(lambda x: x)))
    require_tree(Held(eqx.filter_jit(lambda x: x)))
    assert check_tree(Held(eqx.nn.Dropout(0.5))) == ()


def test_converters_refuse_values_that_do_not_fit():
    """Exact basis: int32 and float32 ranges, float32 smallest subnormal 1.4e-45.
    Fails if a huge Python integer wraps, overflows to inf or underflows
    silently under x32, or if x64 refuses what float64 holds."""
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="do not fit int32"):
            as_leaf(2**70)
        with pytest.raises(ValueError, match="do not fit"):
            as_float_leaf(2**200)
        with pytest.raises(ValueError, match="underflow"):
            as_float_leaf(1e-50)
        assert as_float_leaf(0.0) == 0.0
    with jax.enable_x64(True):
        with pytest.raises(ValueError) as huge:
            as_float_leaf(2**2000)
        assert len(str(huge.value)) < 300
        with pytest.raises(ValueError, match="do not fit int64"):
            as_leaf(2**70)
        assert as_float_leaf(2**200) == float(2**200)
        assert as_float_leaf(1e-50) == 1e-50


def test_converters_name_the_key_refusal(x64):
    """Exact basis: a typed key has no numeric dtype. Fails if a key crashes
    with an opaque dtype error instead of a clear TypeError."""
    for convert in (as_leaf, as_float_leaf):
        with pytest.raises(TypeError, match="PRNG key"):
            convert(jax.random.key(0))


def test_bfloat16_is_flagged_like_float16(x64):
    """Exact count basis. Fails if a bfloat16 leaf passes as a float."""
    for dtype in (jnp.float16, jnp.bfloat16):
        found = check_tree(Held(jnp.ones(2, dtype=dtype)))
        assert [v.reason for v in found] == ["unsupported floating dtype"], dtype


_TWO_CPUS = """
import jax, numpy as np
from hwoutils.leaves import canonical
cpu0, cpu1 = jax.devices("cpu")[:2]
tree = {"x": np.arange(5.0), "n": np.arange(3), "k": [1.5, 2]}
staged = []
put = jax.device_put
jax.device_put = lambda x, *a, **k: (staged.append(x.devices()), put(x, *a, **k))[1]
with jax.default_device(cpu1):
    before = len([a for a in jax.live_arrays() if a.devices() == {cpu1}])
    out = canonical(tree, device=cpu0)
    after = len([a for a in jax.live_arrays() if a.devices() == {cpu1}])
leaves = jax.tree.leaves(out)
assert leaves and all(x.devices() == {cpu0} for x in leaves)
assert after == before
assert staged == [{cpu0}] * len(leaves)
# A sharding target stages on its own device and needs no other backend.
jax.devices = lambda *a: (_ for _ in ()).throw(RuntimeError("Unknown backend"))
staged.clear()
with jax.default_device(cpu1):
    out = canonical(tree, device=jax.sharding.SingleDeviceSharding(cpu0))
assert all(x.devices() == {cpu0} for x in jax.tree.leaves(out))
assert staged == [{cpu0}] * len(leaves)
"""


def test_canonical_places_once_without_staging_on_the_default_device():
    """Fails if a leaf is first built on the default device, then moved.

    Truth: with two CPU devices and the second as the default, every array
    handed to ``device_put`` must already be on the target device (a leaf built
    on the default device would show there), every output leaf sits on the
    target, and no buffer remains on the default device. A sharding target
    stages on its own device without querying another backend (a host started
    with only an accelerator platform has no CPU backend). Exact, no tolerance.
    A subprocess sets the device count before JAX starts.
    """
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2")
    subprocess.run([sys.executable, "-c", _TWO_CPUS], check=True, env=env)
