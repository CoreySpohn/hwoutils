"""Pytest plugin enforcing per-test time, memory and compile budgets.

Load with ``pytest_plugins = ["hwoutils.testbudget.plugin"]`` in the top-level
``conftest.py``. Budgets come from ``[tool.testbudget]`` in ``pyproject.toml``;
see ``hwoutils.testbudget.config.BudgetConfig`` for the keys.
"""

import gc
import os
import sys
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import jax
import pytest

from hwoutils.testbudget import compiles, trend
from hwoutils.testbudget.config import MODES, load_config
from hwoutils.testbudget.memory import GB, current_rss_bytes, peak_rss_bytes

ABORT_EXIT_CODE = 97
TOP_N = 5


def pytest_addoption(parser):
    """Add the command line switches."""
    group = parser.getgroup("testbudget")
    group.addoption(
        "--testbudget-mode",
        choices=MODES,
        default=None,
        help="override [tool.testbudget] mode",
    )
    group.addoption(
        "--testbudget-off", action="store_true", help="disable the guard for this run"
    )
    group.addoption(
        "--budget-report", action="store_true", help="print the full per-test report"
    )


def pytest_configure(config):
    """Create the guard, register markers and start the watchdog."""
    config.addinivalue_line("markers", "medium: integration test, 30 s default budget")
    config.addinivalue_line(
        "markers", "slow: production-size test, 300 s default budget"
    )
    config.addinivalue_line(
        "markers", "time_budget(seconds): override the tier time budget for one test"
    )
    config.addinivalue_line(
        "markers", "compile_budget(n): fail if the test compiles more than n programs"
    )
    if config.getoption("testbudget_off"):
        return
    guard = BudgetGuard(config)
    config.pluginmanager.register(guard, "testbudget-guard")
    guard.start()


def pytest_unconfigure(config):
    """Stop the watchdog."""
    guard = config.pluginmanager.get_plugin("testbudget-guard")
    if guard is not None:
        guard.stop()


@pytest.fixture(autouse=True, scope="module")
def _testbudget_module(request):
    """Check the module compile budget, then clear JAX caches if configured."""
    guard = request.config.pluginmanager.get_plugin("testbudget-guard")
    if guard is None:
        yield
        return
    before = compiles.compile_count()
    before_s = compiles.compile_seconds()
    yield
    used = compiles.compile_count() - before
    used_s = compiles.compile_seconds() - before_s
    name = Path(str(request.node.path)).name
    guard.module_compiles[name] = guard.module_compiles.get(name, 0) + used
    guard.module_compile_s[name] = guard.module_compile_s.get(name, 0.0) + used_s
    if guard.cfg.clear_caches:
        clear_jax_caches()
    if hasattr(request.config, "workerinput"):
        return
    messages = module_budget_messages(guard.cfg, name, used, used_s)
    guard.module_violations.extend(messages)
    if messages and guard.cfg.mode == "enforce":
        raise AssertionError("\n".join(messages))


def module_budget_messages(cfg, name: str, used: int, used_s: float) -> list[str]:
    """Return the module compile budget violations for one test module.

    Under xdist the controller calls this with totals summed over workers, so a
    module split across workers is judged on its whole cost.
    """
    messages = []
    limit = cfg.module_compile_budget_by_file.get(name, cfg.module_compile_budget)
    if limit and used > limit:
        messages.append(
            f"testbudget: {name} compiled {used} programs, over its budget of {limit}. "
            "A numeric value is probably a static jit key; make it an array, or "
            "build the jitted function once in a fixture."
        )
    limit_s = cfg.module_compile_seconds_by_file.get(name, cfg.module_compile_seconds)
    if limit_s and used_s > limit_s:
        messages.append(
            f"testbudget: {name} spent {used_s:.1f} s compiling {used} programs, "
            f"over its budget of {limit_s:.1f} s. Find the expensive programs "
            "with --budget-report; shrink them, share them through a fixture, "
            "or make a static jit key an array."
        )
    return messages


def clear_jax_caches() -> None:
    """Drop JAX compilation caches and collect garbage."""
    jax.clear_caches()
    gc.collect()


def _tier(item) -> str:
    if item.get_closest_marker("slow"):
        return "slow"
    if item.get_closest_marker("medium"):
        return "medium"
    return "unit"


class BudgetGuard:
    """Per-session state: measurements, violations, watchdog and ledger."""

    def __init__(self, config):
        """Read the project configuration and set up empty session state."""
        self.config = config
        self.root = Path(config.rootpath)
        self.cfg = load_config(self.root)
        mode = config.getoption("testbudget_mode")
        if mode:
            self.cfg = replace(self.cfg, mode=mode)
        self.records: list[dict] = []
        self.violations: list[str] = []
        self.module_violations: list[str] = []
        self.module_compiles: dict[str, int] = {}
        self.module_compile_s: dict[str, float] = {}
        self.worker_summaries: list[dict] = []
        self.current_tier = "unit"
        self.current_test = ""
        self.start_time = time.perf_counter()
        self._stop = threading.Event()
        self._thread = None
        self._base_compiles = 0
        self._base_compile_s = 0.0
        self._call_compiles: dict[str, tuple[int, float]] = {}

    def start(self) -> None:
        """Install the compile counter, the compilation cache and the watchdog."""
        compiles.install()
        self._base_compiles = compiles.compile_count()
        self._base_compile_s = compiles.compile_seconds()
        if self.cfg.compile_cache:
            self._enable_compile_cache()
        if self.cfg.rss_ceiling_gb > 0 and self.cfg.rss_poll_s > 0:
            self._thread = threading.Thread(
                target=self._watch, name="testbudget-watchdog", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _enable_compile_cache(self) -> None:
        path = self.cfg.resolved_cache_dir(self.root)
        path.mkdir(parents=True, exist_ok=True)
        ignore = path / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n")
        jax.config.update("jax_compilation_cache_dir", str(path))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

    def _watch(self) -> None:
        while not self._stop.wait(self.cfg.rss_poll_s):
            ceiling = self.cfg.ceiling_gb(self.current_tier)
            rss = current_rss_bytes()
            if ceiling > 0 and rss > ceiling * GB:
                self._announce(
                    f"\ntestbudget: resident memory {rss / GB:.1f} GB passed the "
                    f"{ceiling:.1f} GB ceiling during {self.current_test}; "
                    "aborting this process.\n"
                )
                os._exit(ABORT_EXIT_CODE)

    def _announce(self, text: str) -> None:
        """Write to the real stderr, bypassing pytest output capture."""
        capture = self.config.pluginmanager.get_plugin("capturemanager")
        if capture is not None:
            capture.suspend_global_capture(in_=True)
        sys.stderr.write(text)
        sys.stderr.flush()

    def check_ceiling(self, item) -> None:
        """Stop the session between tests if resident memory is over the ceiling."""
        ceiling = self.cfg.ceiling_gb(_tier(item))
        rss = current_rss_bytes()
        if ceiling > 0 and rss > ceiling * GB:
            pytest.exit(
                f"testbudget: resident memory {rss / GB:.1f} GB passed the "
                f"{ceiling:.1f} GB ceiling "
                f"after {item.nodeid}; stopping the session.",
                returncode=ABORT_EXIT_CODE,
            )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Measure resident-memory growth across setup, call and teardown."""
        self.current_tier = _tier(item)
        self.current_test = item.nodeid
        compiles.reset_pending()
        rss_before = current_rss_bytes()
        result = yield
        rss_after = current_rss_bytes()
        record = getattr(item, "_testbudget_record", None)
        if record is not None:
            record["rss_growth_mb"] = (rss_after - rss_before) / 1024**2
        self.check_ceiling(item)
        return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        """Count compiles that happen during the call phase."""
        before = compiles.compile_count()
        before_s = compiles.compile_seconds()
        try:
            return (yield)
        finally:
            self._call_compiles[item.nodeid] = (
                compiles.compile_count() - before,
                compiles.compile_seconds() - before_s,
            )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Apply the time and compile budgets to the call report."""
        report = yield
        if call.when != "call":
            return report
        tier = _tier(item)
        n_compiles, compile_s = self._call_compiles.pop(item.nodeid, (0, 0.0))
        marker = item.get_closest_marker("time_budget")
        limit_s = (
            float(marker.args[0])
            if marker and marker.args
            else self.cfg.tier_budget_s(tier)
        )
        item._testbudget_record = {
            "nodeid": item.nodeid,
            "tier": tier,
            "call_s": report.duration,
            "compiles": n_compiles,
            "compile_s": compile_s,
        }
        self.records.append(item._testbudget_record)
        if report.outcome != "passed":
            return report
        problems = []
        if report.duration > limit_s:
            problems.append(
                f"testbudget: this {tier} test took {report.duration:.2f} s, "
                f"over its budget of "
                f"{limit_s:.1f} s. Shrink the grid, reuse a compiled fixture, "
                f"or move it to a "
                "slower tier with a reason."
            )
        cmarker = item.get_closest_marker("compile_budget")
        if cmarker and cmarker.args and n_compiles > int(cmarker.args[0]):
            problems.append(
                f"testbudget: this test compiled {n_compiles} programs, "
                f"over its budget of "
                f"{int(cmarker.args[0])}. A value is probably a static jit key."
            )
        if problems:
            self.violations.extend(f"{item.nodeid}: {p}" for p in problems)
            if self.cfg.mode == "enforce":
                report.outcome = "failed"
                report.longrepr = "\n".join(problems)
        return report

    def summary(self) -> dict:
        """Return the merged session summary, including worker results under xdist."""
        records = list(self.records)
        peak = peak_rss_bytes()
        n_compiles = compiles.compile_count() - self._base_compiles
        compile_s = compiles.compile_seconds() - self._base_compile_s
        violations = list(self.violations) + list(self.module_violations)
        module_compiles = dict(self.module_compiles)
        module_compile_s = dict(self.module_compile_s)
        for worker in self.worker_summaries:
            records += worker["records"]
            peak = max(peak, worker["peak_rss_bytes"])
            n_compiles += worker["compiles"]
            compile_s += worker.get("compile_s", 0.0)
            violations += worker["violations"]
            for name, count in worker["module_compiles"].items():
                module_compiles[name] = module_compiles.get(name, 0) + count
            for name, seconds in worker.get("module_compile_s", {}).items():
                module_compile_s[name] = module_compile_s.get(name, 0.0) + seconds
        return {
            "records": records,
            "peak_rss_bytes": peak,
            "compiles": n_compiles,
            "violations": violations,
            "module_compiles": module_compiles,
            "compile_s": compile_s,
            "module_compile_s": module_compile_s,
            "wall_s": time.perf_counter() - self.start_time,
        }

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        """Collect a finished xdist worker's summary on the controller."""
        out = getattr(node, "workeroutput", {}).get("testbudget")
        if out:
            self.worker_summaries.append(out)

    def pytest_sessionfinish(self, session):
        """Hand worker data to the controller, or write the trend line."""
        if hasattr(self.config, "workerinput"):
            self.config.workeroutput["testbudget"] = {
                "records": self.records,
                "peak_rss_bytes": peak_rss_bytes(),
                "compiles": compiles.compile_count() - self._base_compiles,
                "violations": self.violations,
                "module_compiles": self.module_compiles,
                "compile_s": compiles.compile_seconds() - self._base_compile_s,
                "module_compile_s": self.module_compile_s,
            }
            return
        self.final = self.summary()
        if self.worker_summaries:
            self._check_aggregated_modules(session)
        if self.cfg.trend and self.final["records"]:
            self._write_trend(self.final)

    def _check_aggregated_modules(self, session) -> None:
        """Judge module budgets on totals summed over xdist workers."""
        final = self.final
        for name, count in sorted(final["module_compiles"].items()):
            seconds = final["module_compile_s"].get(name, 0.0)
            messages = module_budget_messages(self.cfg, name, count, seconds)
            final["violations"].extend(messages)
            if messages and self.cfg.mode == "enforce" and not session.exitstatus:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _write_trend(self, final: dict) -> None:
        slowest = sorted(final["records"], key=lambda r: r["call_s"], reverse=True)[
            :TOP_N
        ]
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "library": self.cfg.library,
            "sha": trend.git_sha(self.root),
            "args": " ".join(self.config.invocation_params.args),
            "mode": self.cfg.mode,
            "tests": len(final["records"]),
            "wall_s": round(final["wall_s"], 2),
            "peak_rss_gb": round(final["peak_rss_bytes"] / GB, 3),
            "compiles": final["compiles"],
            "compile_s": round(final["compile_s"], 2),
            "violations": len(final["violations"]),
            "slowest": [[r["nodeid"], round(r["call_s"], 2)] for r in slowest],
        }
        try:
            trend.append_record(self.cfg.resolved_trend_file(), record)
        except OSError as err:
            sys.stderr.write(f"testbudget: could not write trend file: {err}\n")

    def pytest_terminal_summary(self, terminalreporter):
        """Print the budget summary after the test results."""
        if hasattr(self.config, "workerinput") or not hasattr(self, "final"):
            return
        final = self.final
        records = final["records"]
        write = terminalreporter.write_line
        terminalreporter.section("testbudget")
        write(
            f"mode={self.cfg.mode} tests={len(records)} wall={final['wall_s']:.1f}s "
            f"peak_rss={final['peak_rss_bytes'] / GB:.2f}GB "
            f"compiles={final['compiles']} "
            f"violations={len(final['violations'])} "
            f"compile_s={final['compile_s']:.1f}"
        )
        for line in final["violations"]:
            write(f"  over budget: {line}")
        if self.config.getoption("budget_report"):
            self._write_detail(write, records, final)

    def _write_detail(self, write, records, final) -> None:
        write("slowest tests:")
        for rec in sorted(records, key=lambda r: r["call_s"], reverse=True)[:TOP_N]:
            write(f"  {rec['call_s']:7.2f} s  [{rec['tier']}] {rec['nodeid']}")
        grown = [
            r
            for r in records
            if r.get("rss_growth_mb", 0) >= self.cfg.rss_growth_report_mb
        ]
        write(
            f"tests growing resident memory by >= "
            f"{self.cfg.rss_growth_report_mb:.0f} MB:"
        )
        for rec in sorted(grown, key=lambda r: r["rss_growth_mb"], reverse=True)[
            : TOP_N * 2
        ]:
            write(f"  {rec['rss_growth_mb']:8.0f} MB  {rec['nodeid']}")
        write("compile count and seconds per module (top by seconds):")
        by_seconds = sorted(
            final["module_compile_s"].items(), key=lambda kv: kv[1], reverse=True
        )
        for name, seconds in by_seconds[: TOP_N * 2]:
            count = final["module_compiles"].get(name, 0)
            write(f"  {count:5d}  {seconds:8.1f} s  {name}")
        biggest = sorted(records, key=lambda r: r["compiles"], reverse=True)[:TOP_N]
        write("most compiles in one test:")
        for rec in biggest:
            write(f"  {rec['compiles']:5d}  {rec['nodeid']}")
