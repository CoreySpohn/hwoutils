"""Exercise the precision plugin in fresh pytest processes."""

import xml.etree.ElementTree as ET

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("ini", "cli", "lane", "parent_lane", "xdist"),
    [
        ("", [], "float32", "1", False),
        ("default_precision = float64", [], "float64", "0", False),
        ("default_precision = float32", ["--precision=float64"], "float64", "0", True),
        ("default_precision = float64", ["--precision=float32"], "float32", "1", False),
    ],
)
def test_precision_lane(pytester, monkeypatch, ini, cli, lane, parent_lane, xdist):
    """Fails if lane selection, import-time dtype, reporting or deselection breaks."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("JAX_ENABLE_X64", parent_lane)
    pytester.makeini(f"[pytest]\n{ini}\n")
    pytester.makeconftest('pytest_plugins = ["hwoutils.testing.precision"]')
    pytester.makepyfile(
        sample_package="""
        from hwoutils.precision import float_dtype

        ON_IMPORT = float_dtype()
        """,
        test_lanes=f"""
        import numpy as np
        import pytest
        from sample_package import ON_IMPORT

        def test_common(precision):
            assert precision == ON_IMPORT == np.dtype("{lane}")

        @pytest.mark.float64_only(reason="Needs float64")
        def test_double(precision):
            assert precision == np.dtype("float64")

        @pytest.mark.float32_only(reason="Tests float32")
        def test_single(precision):
            assert precision == np.dtype("float32")
        """,
    )
    options = ["-p", "xdist", "-n", "1"] if xdist else []
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
