"""Exercise the precision plugin in fresh pytest processes."""

import os
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("ini", "cli", "lane", "parent_lane", "xdist"),
    [
        ("default_precision = float64", ["--precision=float32"], "float32", "1", False),
        ("default_precision = float32", ["--precision=float64"], "float64", "0", False),
        ("default_precision = float64", ["--precision=float32"], "float32", "1", True),
        ("default_precision = float32", ["--precision=float64"], "float64", "0", True),
        ("", [], "float32", "1", False),
        ("default_precision = float64", [], "float64", "0", False),
    ],
)
def test_precision_lane(pytester, monkeypatch, ini, cli, lane, parent_lane, xdist):
    """Fails if lane selection, import-time dtype, reporting or deselection breaks."""
    options = []
    if xdist:
        # Source-tree tests do not install the new distribution metadata.
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        metadata = tomllib.loads(project.read_text())["project"]
        entry_points = metadata.get("entry-points", {}).get("pytest11", {})
        dist_info = pytester.path / "lane_probe-0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Name: lane-probe\nVersion: 0\n")
        entries = "".join(
            f"{name} = {module}\n" for name, module in entry_points.items()
        )
        (dist_info / "entry_points.txt").write_text(f"[pytest11]\n{entries}")
        search_path = [str(pytester.path)]
        if pythonpath := os.environ.get("PYTHONPATH"):
            search_path.append(pythonpath)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(search_path))
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        options = ["-n", "2"]
    else:
        monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        options = ["-p", "hwoutils.testing.precision"]
    monkeypatch.setenv("JAX_ENABLE_X64", parent_lane)
    pytester.makeini(f"[pytest]\n{ini}\n")
    pytester.makeconftest("from sample_package import ON_IMPORT as CONFTIMPORT")
    pytester.makepyfile(
        sample_package="""
        import jax.numpy as jnp

        ON_IMPORT = jnp.asarray(1.0)
        """,
        test_lanes=f"""
        import numpy as np
        import pytest
        from conftest import CONFTIMPORT

        def test_common(precision):
            assert precision == CONFTIMPORT.dtype == np.dtype("{lane}")

        @pytest.mark.float64_only(reason="Needs float64")
        def test_double(precision):
            assert precision == np.dtype("float64")

        @pytest.mark.float32_only(reason="Tests float32")
        def test_single(precision):
            assert precision == np.dtype("float32")
        """,
    )
    result = pytester.runpytest_subprocess(
        "-v", "--junitxml=result.xml", *options, *cli
    )
    result.assert_outcomes(passed=2, deselected=int(not xdist), skipped=0)
    result.stdout.fnmatch_lines([f"*precision: {lane}*"])
    suite = ET.parse(pytester.path / "result.xml")
    assert (
        suite.find(".//testsuite/properties/property[@name='precision']").get("value")
        == lane
    )


def test_precision_plugin_without_jax(pytester, monkeypatch):
    """Loading the plugin preserves pytest startup for NumPy-only installs."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makepyfile("def test_smoke(precision): assert precision.name == 'float32'")
    result = pytester.run(
        sys.executable,
        "-c",
        "import sys; sys.modules['jax'] = None; import pytest; "
        "sys.exit(pytest.main(['-p', 'hwoutils.testing.precision']))",
    )
    result.assert_outcomes(passed=1)
