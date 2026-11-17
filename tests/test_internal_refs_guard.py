"""Contract checks for the shared text scanner and pytest plugin."""

import subprocess
import sys
from pathlib import Path

from hwoutils import internal_refs

pytest_plugins = ["pytester"]


def test_clean_file_passes(tmp_path):
    """Exact text basis. Fails if clean text or the scanner's source is rejected."""
    path = tmp_path / "clean.py"
    path.write_text('"""Pupils and focal planes."""\nx = 1\n')
    assert internal_refs.scan_file(path) == []
    assert internal_refs.scan([path, Path(internal_refs.__file__)]) == []
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.argv = ['hwoutils.internal_refs', sys.argv[1]]\n"
            "try:\n runpy.run_module('hwoutils.internal_refs', run_name='__main__')\n"
            "finally:\n assert 'jax' not in sys.modules\n"
            " assert 'pytest' not in sys.modules\n",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_reference_fails(tmp_path, capsys):
    """Mutation basis. Fails if any prohibited family escapes or loses its report."""
    examples = [
        "mission-control",  # internal-ref-ok
        "research-brain",  # internal-ref-ok
        "physicaloptix-analysis",  # internal-ref-ok
        "brain/notes",  # internal-ref-ok
        "deep-thoughts",  # internal-ref-ok
        "SOME_DESIGN_DOC.md",  # internal-ref-ok
        "eac1_dlux",  # internal-ref-ok
        "Tier-G",  # internal-ref-ok
        "Tier R",  # internal-ref-ok
        "GREENFIELD",  # internal-ref-ok
        "Synthesis Family",  # internal-ref-ok
        "SYNTHESIS FAMILIES",  # internal-ref-ok
        "hwosim-paper",  # internal-ref-ok
        "hwosim-build",  # internal-ref-ok
        "spohnbook-build",  # internal-ref-ok
        "zodi-library",  # internal-ref-ok
    ]
    path = tmp_path / "bad.py"
    for example in examples:
        path.write_text(f"# {example}\n")
        hits = internal_refs.scan([path])
        assert len(hits) == 1
        assert hits[0][0:2] == (path, 1)
        assert hits[0][3] == f"# {example}"
        assert internal_refs.main([str(path)]) == 1
        message = capsys.readouterr().out
        assert f"{path}:1: matches {hits[0][2]!r}: # {example}" in message
        assert "Internal research references do not belong in library code." in message
        assert internal_refs.WAIVER in message


def test_waiver_comment_allows(tmp_path):
    """Mutation basis. Fails if the exact per-line waiver stops suppressing hits."""
    path = tmp_path / "waived.py"
    content = "# see brain/notes\n"  # internal-ref-ok
    path.write_text(content.rstrip() + f"  # {internal_refs.WAIVER}\n")
    assert internal_refs.scan_file(path) == []
    assert internal_refs.main([str(path)]) == 0
    path.write_text(content)
    assert len(internal_refs.scan_file(path)) == 1


def test_tracked_but_deleted_file_is_ignored(tmp_path, monkeypatch):
    """Mutation basis. Fails if an unstaged deletion or text filtering breaks scans."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    names = [
        "deleted.py",
        "CHANGELOG.md",
        "image.png",
        "notes.txt",
        "notebook.ipynb",
        "space and\nnewline.md",
        "untracked.md",
    ]
    for name in names:
        (tmp_path / name).write_text("# Public description\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", *names[:-1]], check=True)
    (tmp_path / names[0]).unlink()
    assert set(internal_refs.tracked_text_files(tmp_path)) == {
        tmp_path / name for name in names[3:-1]
    }
    monkeypatch.chdir(tmp_path)
    assert internal_refs.main([]) == 0


def test_plugin_contributes_one_item(pytester, monkeypatch):
    """Exact collection basis. Fails if the guard duplicates, cannot fail or disable."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest('pytest_plugins = ["hwoutils.internal_refs"]')
    pytester.makepyfile(
        test_one="# Public description", test_two="# Public description"
    )
    subprocess.run(["git", "init", "-q", str(pytester.path)], check=True)
    subprocess.run(["git", "add", "."], cwd=pytester.path, check=True)
    result = pytester.runpytest_inprocess("-v")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*internal-refs PASSED*"])
    path = pytester.path / "test_two.py"
    path.write_text("# see brain/notes\n")  # internal-ref-ok
    result = pytester.runpytest_inprocess("-q")
    result.assert_outcomes(failed=1)
    assert f"{path}:1: matches" in result.stdout.str()
    result = pytester.runpytest_inprocess("--deselect=internal-refs")
    result.assert_outcomes(deselected=1)
    result = pytester.runpytest_inprocess("-p", "no:hwoutils.internal_refs")
    assert result.ret == 5
