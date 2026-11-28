"""Contract tests for build stamps and owner-relative reads."""

import hashlib
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hwoutils import provenance as prov
from hwoutils.leaves import HostGrid


def test_stamp_value_table():
    """Fails if a fact loses its type, exact value, or unknown status.

    Truth: the specified strings, with exact equality. The second binary64
    value next to one is one ULP away and must have a different encoding.
    """
    adjacent = np.nextafter(1.0, 2.0)
    assert prov.stamp_value(1.0) != prov.stamp_value(adjacent)
    digest = "0123456789abcdef" * 4
    grid = HostGrid([1.0, 2.0])
    grid_sha = hashlib.sha256(grid.values.tobytes()).hexdigest()
    table = (
        (None, "unknown"),
        (True, "bool:true"),
        (False, "bool:false"),
        (-3, "int:-3"),
        (np.uint16(7), "int:7"),
        (0.0, "f64:0x0.0p+0"),
        (-0.0, "f64:0x0.0p+0"),
        (1.0, "f64:0x1.0000000000000p+0"),
        (adjacent, "f64:0x1.0000000000001p+0"),
        (0.1, "f64:0x1.999999999999ap-4"),
        (np.float32(0.1), "f32:0x1.99999a0000000p-4"),
        (np.float32(-0.0), "f32:0x0.0p+0"),
        ("unknown", "str:unknown"),
        (prov.Digest(digest), f"sha256:{digest}"),
        (grid, f"grid:2:sha256:{grid_sha}"),
        ((1, (None, "star")), "tuple:(int:1,tuple:(unknown,str:star))"),
    )
    for value, expected in table:
        assert prov.stamp_value(value) == expected
    assert prov.UNKNOWN == "unknown"
    for value in (float("nan"), np.float32("nan")):
        with pytest.raises(ValueError, match="flux"):
            prov.stamp_value(value, label="flux")
    for value in (eqx.nn.Identity(), np.float16(1), object()):
        with pytest.raises(TypeError, match="flux"):
            prov.stamp_value(value, label="flux")
    for value in (digest.upper(), digest[:-1], "g" + digest[1:]):
        with pytest.raises(ValueError, match="hex"):
            prov.Digest(value)


def test_array_stamps_cover_bytes_dtype_and_shape():
    """Fails if array stamps lose their bytes, dtype, shape, or host boundary.

    Truth: hashlib of the C-order bytes and the hand-written shape formats.
    All encodings and refusals are checked exactly.
    """
    host = np.array([17, 53], dtype=np.int32)
    device = jnp.asarray(host)
    sha = hashlib.sha256(host.tobytes()).hexdigest()
    dtype = host.dtype.str
    other_dtype = host.view(np.uint32)
    assert prov.stamp_value(host) == f"array:{dtype}:(2,):sha256:{sha}"
    assert prov.stamp_value(device) == f"array:{dtype}:(2,):sha256:{sha}"
    assert (
        prov.stamp_value(other_dtype)
        == f"array:{other_dtype.dtype.str}:(2,):sha256:{sha}"
    )
    assert prov.stamp_value(host.reshape(1, 2)) == f"array:{dtype}:(1, 2):sha256:{sha}"

    stamp = partial(prov.stamp_value, label="pixels")
    with pytest.raises(TypeError, match="pixels"):
        jax.jit(stamp).lower(device)


def test_build_stamps_sorts_and_checks_labels():
    """Fails if keyword order leaks into stamps or invalid labels enter them.

    Truth: the label grammar and exact sorted tuple.
    """
    expected = (
        ("aperture_1", "int:2"),
        ("stellar.radius", "unknown"),
        ("z", "bool:true"),
    )
    assert (
        prov.build_stamps(z=True, **{"stellar.radius": None}, aperture_1=2) == expected
    )
    assert (
        prov.build_stamps(aperture_1=2, **{"stellar.radius": None}, z=True) == expected
    )
    for label in ("", "Flux", "star..flux", "1star", "star-flux", "_star"):
        with pytest.raises(ValueError, match="label"):
            prov.build_stamps(**{label: 0})


def test_compare_stamps_distinguishes_unverified_from_mismatch():
    """Fails if absence or unknown becomes a match or mismatch.

    Truth: a hand-written union of facts and exact three-way comparison.
    """
    left = (
        ("absent_right", "int:1"),
        ("different", "int:2"),
        ("equal", "str:star"),
        ("unknown_both", "unknown"),
        ("unknown_left", "unknown"),
    )
    right = (
        ("absent_left", "int:1"),
        ("different", "int:3"),
        ("equal", "str:star"),
        ("unknown_both", "unknown"),
        ("unknown_left", "f64:0x0.0p+0"),
    )
    assert prov.compare_stamps(left, right) == prov.StampComparison(
        ("equal",),
        ("absent_left", "absent_right", "unknown_both", "unknown_left"),
        (("different", "int:2", "int:3"),),
    )
    assert prov.compare_stamps(left, right, labels=("equal", "absent_both")) == (
        ("equal",),
        ("absent_both",),
        (),
    )
    assert prov.compare_stamps(prov.build_stamps(flux=None), (("flux", "int:0"),)) == (
        (),
        ("flux",),
        (),
    )


def test_require_stamps_reports_conflicts_and_returns_unverified():
    """Fails if a mismatch is silent or an unknown fact raises.

    Truth: the exact labels and values of a mismatch and an unverified fact.
    """
    left = (("band", "int:2"), ("source", "unknown"))
    right = (("band", "int:3"), ("source", "str:calibration"))
    with pytest.raises(prov.StampMismatchError, match=r"exposure.*band.*int:2.*int:3"):
        prov.require_stamps(left, right, what="exposure")
    assert prov.require_stamps(left, right, what="exposure", labels=("source",)) == (
        "source",
    )


def test_prefix_and_merge_stamps_keep_children_without_overwriting_conflicts():
    """Fails if child labels collide or duplicate facts hide a conflict.

    Truth: exact expected child labels and the conflict contract.
    """
    left = prov.build_stamps(radius=1)
    right = prov.build_stamps(radius=2)
    with pytest.raises(ValueError, match="radius"):
        prov.merge_stamps(left, right)
    with pytest.raises(ValueError, match="label"):
        prov.prefix_stamps("Bad", left)
    stellar = prov.prefix_stamps("stellar", left)
    planet = prov.prefix_stamps("planet", right)
    assert prov.merge_stamps(stellar, planet, stellar) == (
        ("planet.radius", "int:2"),
        ("stellar.radius", "int:1"),
    )


def test_bind_reads_rewrites_root_nested_and_held_owners():
    """Fails if reads lose their owner or bind to an absent or held owner.

    Truth: the owner-to-model mapping and the exact paths written by hand.
    """
    scene = prov.qualify("scene", ("star", "dynamics.frame", ""))
    instrument = prov.qualify("instrument", ("band", ""))
    held = prov.qualify("sampling", ("grid",))
    assert scene == ("scene.star", "scene.dynamics.frame", "scene")
    assert instrument == ("instrument.band", "instrument")
    roots = {"scene": "", "instrument": "instruments[0]", "sampling": None}
    assert prov.bind_reads((*scene, *instrument, *held), roots) == (
        "star",
        "dynamics.frame",
        "",
        "instruments[0].band",
        "instruments[0]",
    )
    with pytest.raises(ValueError, match=r"exposure.*instrument.*sampling.*scene"):
        prov.bind_reads(("exposure",), roots)


def test_stale_reads_matches_only_whole_nodes():
    """Fails if string prefixes alone count as path overlaps.

    Truth: exact whole-node ancestry in both directions.
    """
    reads = ("star", "star.spectrum.flux", "star_b")
    write = "star.spectrum"
    assert prov.stale_reads(iter(reads), (write, "star")) == (
        (write, "star"),
        (write, "star.spectrum.flux"),
        ("star", "star"),
        ("star", "star.spectrum.flux"),
    )


def test_display_key_is_only_for_display(monkeypatch):
    """Fails if the key is too short, ignores facts, or decides equality.

    Truth: SHA-256 of the exact canonical text and a simulated key collision.
    """
    left = prov.build_stamps(a=1, b=3)
    changed_a = prov.build_stamps(a=2, b=3)
    changed_b = prov.build_stamps(a=1, b=4)
    assert (
        prov.display_key(left) == hashlib.sha256(b"a=int:1\nb=int:3").hexdigest()[:16]
    )
    assert (
        len({prov.display_key(stamps) for stamps in (left, changed_a, changed_b)}) == 3
    )
    monkeypatch.setattr(prov, "display_key", lambda _: "0" * 16)
    assert prov.display_key(left) == prov.display_key(changed_a)
    assert prov.compare_stamps(left, changed_a) == (
        ("b",),
        (),
        (("a", "int:1", "int:2"),),
    )
