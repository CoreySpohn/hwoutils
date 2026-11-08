"""Tests for the pytest test budget guard (hwoutils.testbudget)."""

import json

import pytest

from hwoutils.testbudget import trend
from hwoutils.testbudget.cli import main as cli_main
from hwoutils.testbudget.config import load_config

pytest_plugins = ["pytester"]


@pytest.fixture
def project(pytester, tmp_path):
    """Return a helper that writes a pyproject with a budget table and runs pytest."""
    trend_file = tmp_path / "trend.jsonl"

    def make(table="", body="", args=()):
        pytester.makepyprojecttoml(
            '[project]\nname = "demo"\n[tool.testbudget]\n'
            f'trend_file = "{trend_file}"\nrss_poll_s = 0\n{table}\n'
        )
        pytester.makepyfile(test_demo=body)
        return pytester.runpytest_inprocess("-p", "hwoutils.testbudget.plugin", *args)

    make.trend_file = trend_file
    return make


SLEEPY = """
import time
def test_sleepy():
    time.sleep(0.3)
"""


def test_over_budget_fails_in_enforce_mode(project):
    """Fails if an over-budget unit test passes silently under enforce."""
    result = project('mode = "enforce"\nunit_s = 0.1', SLEEPY)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*over its budget of 0.1 s*"])


def test_over_budget_only_reported_in_report_mode(project):
    """Fails if report mode changes the test outcome."""
    result = project("unit_s = 0.1", SLEEPY)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*violations=1*", "*over budget:*took*"])


def test_tier_marker_selects_budget(project):
    """Fails if a medium marker does not get the medium budget."""
    body = SLEEPY.replace("def test", "import pytest\n@pytest.mark.medium\ndef test")
    result = project('mode = "enforce"\nunit_s = 0.1\nmedium_s = 5', body)
    result.assert_outcomes(passed=1)


def test_cli_mode_overrides_pyproject(project):
    """Fails if the command line switch cannot adopt a library in report mode."""
    result = project(
        'mode = "enforce"\nunit_s = 0.1', SLEEPY, args=("--testbudget-mode", "report")
    )
    result.assert_outcomes(passed=1)


def test_compile_budget_marker_counts_distinct_programs(project):
    """Fails if value-different static jit keys are not counted as compiles."""
    body = """
import jax, jax.numpy as jnp, functools, pytest

@pytest.mark.compile_budget(1)
def test_static_keys():
    @functools.partial(jax.jit, static_argnums=0)
    def scale(k, x):
        return x * k
    x = jnp.ones(3)
    for k in (2, 3, 4):
        scale(k, x)
"""
    result = project('mode = "enforce"', body)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*compiled [345] programs*"])


def test_module_compile_budget_fails_module_teardown(project):
    """Fails if a per-module compile budget does not turn into an error."""
    body = """
import jax, jax.numpy as jnp

def test_a():
    x = jnp.ones(2)
    jax.jit(lambda x: x + 1)(x)
    jax.jit(lambda x: x + 2)(x)
"""
    result = project('mode = "enforce"\nmodule_compile_budget = 1', body)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*test_demo.py compiled * programs*"])


FAKE_EVENTS = """
import jax.monitoring

COMPILE = "/jax/core/compile/backend_compile_duration"
HIT = "/jax/compilation_cache/cache_hits"

def test_a():
    jax.monitoring.record_event_duration_secs(COMPILE, 2.0)
    jax.monitoring.record_event_duration_secs(COMPILE, 0.5)
    jax.monitoring.record_event(HIT)
    jax.monitoring.record_event_duration_secs(COMPILE, 9.0)
    jax.monitoring.record_event_duration_secs(COMPILE, 1.0)
"""


def test_compile_seconds_exclude_cache_hits(project):
    """Fails if a cache load is counted as a compile or its time is added."""
    result = project("", FAKE_EVENTS, args=("--budget-report",))
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["*compiles=3 violations=0 compile_s=3.5", "*3*3.5 s  test_demo.py"]
    )


def test_summary_line_keeps_field_order_and_appends_compile_s(project):
    """Fails if compile_s is not appended after the existing summary fields."""
    result = project("", "def test_a():\n    pass\n")
    result.stdout.fnmatch_lines(
        ["mode=report tests=1 wall=*s peak_rss=*GB compiles=* violations=0 compile_s=*"]
    )


def test_module_compile_seconds_fails_in_enforce_mode(project):
    """Fails if a module over its compile-seconds budget is not an error."""
    result = project('mode = "enforce"\nmodule_compile_seconds = 1', FAKE_EVENTS)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*test_demo.py spent 3.5 s compiling 3 programs*"])


def test_module_compile_seconds_only_reported_in_report_mode(project):
    """Fails if report mode changes the outcome of a compile-seconds violation."""
    result = project("module_compile_seconds = 1", FAKE_EVENTS)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*violations=1*", "*over budget:*spent 3.5 s*"])


def test_module_compile_seconds_per_file_override(project):
    """Fails if the per-file value does not override the global budget."""
    table = (
        'mode = "enforce"\nmodule_compile_seconds = 1\n'
        'module_compile_seconds_by_file = {"test_demo.py" = 10}'
    )
    result = project(table, FAKE_EVENTS)
    result.assert_outcomes(passed=1)


def test_reentrant_hits_pair_with_their_durations():
    """Fails if nested cache loads leave a hit unpaired and count a false compile."""
    from hwoutils.testbudget import compiles

    compiles.reset_pending()
    before = compiles.compile_count()
    compiles._on_event(compiles._CACHE_HIT)
    compiles._on_event(compiles._CACHE_HIT)
    compiles._on_duration(compiles._BACKEND_COMPILE, 1.0)
    compiles._on_duration(compiles._BACKEND_COMPILE, 1.0)
    assert compiles.compile_count() == before
    compiles._on_duration(compiles._BACKEND_COMPILE, 1.0)
    assert compiles.compile_count() == before + 1


def test_stranded_hit_does_not_leak_across_tests(project):
    """Fails if a hit whose duration never arrived hides the next test's compile."""
    body = """
import jax.monitoring

def test_a():
    jax.monitoring.record_event("/jax/compilation_cache/cache_hits")

def test_b():
    jax.monitoring.record_event_duration_secs(
        "/jax/core/compile/backend_compile_duration", 1.0
    )
"""
    result = project("", body)
    result.stdout.fnmatch_lines(["*compiles=1 violations=0 compile_s=1.0"])


SPLIT = """
import jax.monitoring

def _compile():
    jax.monitoring.record_event_duration_secs(
        "/jax/core/compile/backend_compile_duration", 2.0
    )

def test_a():
    _compile()

def test_b():
    _compile()
"""


def test_xdist_module_budget_uses_totals_across_workers(project):
    """Fails if a module split over workers is judged on one worker's share."""
    pytest.importorskip("xdist")
    table = 'mode = "enforce"\nmodule_compile_seconds = 3\nmodule_compile_budget = 1'
    serial = project(table, SPLIT)
    serial.assert_outcomes(passed=2, errors=1)
    split = project(table, SPLIT, args=("-n", "2", "--dist", "load"))
    assert split.ret != 0
    split.stdout.fnmatch_lines(
        [
            "*over budget:*test_demo.py compiled 2 programs*",
            "*over budget:*test_demo.py spent 4.0 s compiling 2 programs*",
        ]
    )
    report = project("module_compile_seconds = 3", SPLIT, args=("-n", "2"))
    assert report.ret == 0
    report.stdout.fnmatch_lines(["*violations=1*"])


def test_background_cache_load_spanning_a_test_boundary_is_excluded(project):
    """Fails if a background cache load spanning a boundary counts as a compile."""
    body = """
import threading
import jax.monitoring

release = threading.Event()
started = threading.Event()

def _load():
    jax.monitoring.record_event("/jax/compilation_cache/cache_hits")
    started.set()
    release.wait(10)
    jax.monitoring.record_event_duration_secs(
        "/jax/core/compile/backend_compile_duration", 1.0
    )

thread = threading.Thread(target=_load)

def test_a():
    thread.start()
    assert started.wait(10)

def test_b():
    release.set()
    thread.join(10)
"""
    result = project("", body)
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*compiles=0 violations=0 compile_s=0.0"])


def test_same_basename_in_two_directories_is_two_modules(pytester, tmp_path):
    """Fails if modules sharing a file name merge into one budget under xdist."""
    pytest.importorskip("xdist")
    pytester.makepyprojecttoml(
        '[project]\nname = "demo"\n[tool.testbudget]\n'
        f'trend_file = "{tmp_path / "t.jsonl"}"\nrss_poll_s = 0\n'
        'mode = "enforce"\nmodule_compile_seconds = 3\n'
        'module_compile_seconds_by_file = {"test_same.py" = 3}\n'
    )
    for sub in ("a", "b"):
        pytester.mkpydir(sub)
        (pytester.path / sub / "test_same.py").write_text(SPLIT.split("def test_b")[0])
    result = pytester.runpytest_inprocess(
        "-p", "hwoutils.testbudget.plugin", "-n", "2", "--dist", "load"
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0


def _two_modules(pytester, tmp_path, table):
    pytester.makepyprojecttoml(
        '[project]\nname = "demo"\n[tool.testbudget]\n'
        f'trend_file = "{tmp_path / "t.jsonl"}"\nrss_poll_s = 0\n{table}\n'
    )
    (pytester.path / "sub").mkdir(exist_ok=True)
    (pytester.path / "sub" / "__init__.py").touch()
    (pytester.path / "sub" / "test_m.py").write_text(SPLIT)
    return pytester.runpytest_inprocess("-p", "hwoutils.testbudget.plugin")


def test_by_file_key_matches_basename_of_nested_module(pytester, tmp_path):
    """Fails if a bare file name stops matching a module in a subdirectory."""
    table = 'mode = "enforce"\nmodule_compile_seconds_by_file = {"test_m.py" = 1}'
    result = _two_modules(pytester, tmp_path, table)
    result.assert_outcomes(passed=2, errors=1)


def test_by_file_relative_path_wins_over_basename(pytester, tmp_path):
    """Fails if the bare file name overrides the relative path key."""
    loose = 'module_compile_seconds_by_file = {"sub/test_m.py" = 10, "test_m.py" = 1}'
    result = _two_modules(pytester, tmp_path, 'mode = "enforce"\n' + loose)
    result.assert_outcomes(passed=2)
    tight = 'module_compile_seconds_by_file = {"sub/test_m.py" = 1, "test_m.py" = 10}'
    result = _two_modules(pytester, tmp_path, 'mode = "enforce"\n' + tight)
    result.assert_outcomes(passed=2, errors=1)


def test_revisit_violation_is_listed_once(pytester, tmp_path):
    """Fails if a module checked at two teardowns lists its violation twice."""
    pytester.makepyprojecttoml(
        '[project]\nname = "demo"\n[tool.testbudget]\n'
        f'trend_file = "{tmp_path / "t.jsonl"}"\nrss_poll_s = 0\n'
        "module_compile_seconds = 1\n"
    )
    pytester.makeconftest(
        "def pytest_collection_modifyitems(items):\n"
        "    order = ['test_a', 'test_b1', 'test_b']\n"
        "    items.sort(key=lambda i: order.index(i.name))\n"
    )
    pytester.makepyfile(test_a=SPLIT, test_other="def test_b1():\n    pass\n")
    result = pytester.runpytest_inprocess("-p", "hwoutils.testbudget.plugin")
    result.stdout.fnmatch_lines(["*violations=1 *"])


def test_module_revisits_are_judged_cumulatively(pytester, tmp_path):
    """Fails if a module run in two fixture lifetimes passes on each part alone."""
    pytester.makepyprojecttoml(
        '[project]\nname = "demo"\n[tool.testbudget]\n'
        f'trend_file = "{tmp_path / "t.jsonl"}"\nrss_poll_s = 0\n'
        'mode = "enforce"\nmodule_compile_seconds = 3\n'
    )
    pytester.makeconftest(
        "def pytest_collection_modifyitems(items):\n"
        "    order = ['test_a1', 'test_b1', 'test_a2']\n"
        "    items.sort(key=lambda i: order.index(i.name))\n"
    )
    pytester.makepyfile(
        test_a=SPLIT.replace("test_a", "test_a1").replace("test_b", "test_a2"),
        test_b="def test_b1():\n    pass\n",
    )
    result = pytester.runpytest_inprocess("-p", "hwoutils.testbudget.plugin")
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(["*test_a.py spent 4.0 s*"])


def test_clear_caches_is_opt_in(project, monkeypatch):
    """Fails if the module fixture clears caches without the config switch."""
    import hwoutils.testbudget.plugin as plugin

    calls = []
    monkeypatch.setattr(plugin, "clear_jax_caches", lambda: calls.append(1))
    project("", "def test_a():\n    pass\n")
    assert calls == []
    project("clear_caches = true", "def test_a():\n    pass\n")
    assert calls == [1]


def test_rss_ceiling_stops_session(project):
    """Fails if a session over the memory ceiling keeps running."""
    body = "def test_a():\n    pass\ndef test_b():\n    pass\n"
    result = project("rss_ceiling_gb = 0.001", body)
    assert result.ret == 97
    result.stdout.fnmatch_lines(["*passed the 0.0 GB ceiling*"])


def test_rss_growth_reported(project):
    """Fails if a test that allocates memory is missing from the growth report."""
    body = (
        "keep = []\n"
        "def test_big():\n"
        "    buf = bytearray(200 * 1024**2)\n"
        "    buf[::4096] = b'x' * len(buf[::4096])\n"
        "    keep.append(buf)\n"
    )
    result = project("rss_growth_report_mb = 50", body, args=("--budget-report",))
    result.stdout.fnmatch_lines(["*MB  *test_big*"])


def test_trend_line_is_written(project):
    """Fails if a session does not append one parseable summary line."""
    project("", "def test_a():\n    pass\n")
    lines = project.trend_file.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["library"] == "demo"
    assert rec["tests"] == 1
    assert rec["slowest"][0][0].endswith("test_a")


def test_compile_cache_dir_gets_gitignore(pytester, project):
    """Fails if the persistent cache directory is not self-ignoring."""
    project("compile_cache = true", "def test_a():\n    pass\n")
    assert (pytester.path / ".jax_cache" / ".gitignore").read_text() == "*\n"


def test_unknown_config_key_is_an_error(tmp_path):
    """Fails if a misspelled key silently leaves a budget at its default."""
    (tmp_path / "pyproject.toml").write_text("[tool.testbudget]\nunit_secs = 1\n")
    with pytest.raises(ValueError, match="unit_secs"):
        load_config(tmp_path)


def test_xdist_workers_are_aggregated(project):
    """Fails if worker results are lost from the controller summary."""
    pytest.importorskip("xdist")
    body = "def test_a():\n    pass\ndef test_b():\n    pass\n"
    result = project("", body, args=("-n", "2"))
    result.assert_outcomes(passed=2)
    assert json.loads(project.trend_file.read_text())["tests"] == 2


def _rec(wall, args="x"):
    return {
        "wall_s": wall,
        "peak_rss_gb": 1.0,
        "compiles": 10,
        "tests": 5,
        "args": args,
    }


def test_growth_flag_needs_20_percent_over_prior_mean():
    """Fails if growth is flagged below 20 percent or missed above it."""
    history = [_rec(10.0)] * 5
    assert trend.growth_flags([*history, _rec(11.5)]) == {}
    assert set(trend.growth_flags([*history, _rec(13.0)])) == {"wall_s"}


def test_growth_compares_only_matching_invocations():
    """Fails if a one-file run is compared against full-suite runs."""
    history = [_rec(100.0, "all")] * 5
    assert trend.growth_flags([*history, _rec(1.0, "one"), _rec(2.0, "one")]) == {}


def test_cli_exit_code_follows_flags(tmp_path, capsys):
    """Fails if the CLI does not print the table and exit 1 on a flag."""
    path = tmp_path / "lib.jsonl"
    for rec in [*[_rec(10.0)] * 4, _rec(20.0)]:
        trend.append_record(path, {**rec, "time": "2026-01-01T00:00:00", "sha": "abc"})
    assert cli_main(["trend", str(path)]) == 1
    assert "FLAG wall_s" in capsys.readouterr().out


def test_watchdog_kills_a_runaway_test(pytester):
    """Fails if a single test that never returns can exceed the ceiling unnoticed."""
    pytester.makepyprojecttoml(
        '[project]\nname = "demo"\n[tool.testbudget]\ntrend = false\n'
        "rss_ceiling_gb = 0.001\nrss_poll_s = 0.05\n"
    )
    pytester.makepyfile(test_demo="import time\ndef test_hang():\n    time.sleep(20)\n")
    result = pytester.runpytest_subprocess("-p", "hwoutils.testbudget.plugin")
    assert result.ret == 97
    result.stderr.fnmatch_lines(["*ceiling during*test_hang*"])
