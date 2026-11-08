"""Pytest plugin enforcing per-test time, memory and compile budgets.

Load with ``pytest_plugins = ["hwoutils.testbudget.plugin"]`` in the top-level
``conftest.py``. Budgets come from ``[tool.testbudget]`` in ``pyproject.toml``;
see ``hwoutils.testbudget.config.BudgetConfig`` for the keys.
"""

import gc
import json
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
    group.addoption(
        "--budget-json",
        default=None,
        metavar="PATH",
        help="write the session summary, per-test records and per-module "
        "compiles as JSON",
    )


def pytest_configure(config):
    """Create the guard, register markers and start the watchdog."""
    config.addinivalue_line(
        "markers", "medium(reason): integration test, 30 s default budget"
    )
    config.addinivalue_line(
        "markers", "slow(reason): production-size test, 300 s default budget"
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
    before_loads = compiles.cache_hit_count()
    yield
    used = compiles.compile_count() - before
    used_s = compiles.compile_seconds() - before_s
    loads = compiles.cache_hit_count() - before_loads
    name = request.node.nodeid
    guard.module_compiles[name] = guard.module_compiles.get(name, 0) + used
    guard.module_compile_s[name] = guard.module_compile_s.get(name, 0.0) + used_s
    guard.module_cache_loads[name] = guard.module_cache_loads.get(name, 0) + loads
    if guard.cfg.clear_caches:
        clear_jax_caches()
    if hasattr(request.config, "workerinput"):
        return
    messages = module_budget_messages(
        guard.cfg,
        name,
        guard.module_compiles[name],
        guard.module_compile_s[name],
        guard.module_cache_loads[name],
    )
    guard.module_violations[name] = messages
    if messages and guard.cfg.module_enforced():
        raise AssertionError("\n".join(messages))


def module_budget_messages(
    cfg, name: str, used: int, used_s: float, loads: int = 0
) -> list[str]:
    """Return the module compile budget violations for one test module.

    The count budget is on programs compiled or loaded from the persistent
    compilation cache, so it does not depend on whether the cache is warm. The
    seconds budget is on real compiles only.

    Under xdist the controller calls this with totals summed over workers, so a
    module split across workers is judged on its whole cost.
    """
    messages = []
    base = Path(name).name
    limit = cfg.module_compile_budget_by_file.get(
        name, cfg.module_compile_budget_by_file.get(base, cfg.module_compile_budget)
    )
    if limit and used + loads > limit:
        detail = f" ({loads} from the cache)" if loads else ""
        messages.append(
            f"testbudget: {name} compiled {used + loads} programs{detail}, over its "
            f"budget of {limit}. "
            "A numeric value is probably a static jit key; make it an array, or "
            "build the jitted function once in a fixture."
        )
    limit_s = cfg.module_compile_seconds_by_file.get(
        name,
        cfg.module_compile_seconds_by_file.get(base, cfg.module_compile_seconds),
    )
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


def _has_reason(marker) -> bool:
    """Return whether a tier mark states why the test is in that tier."""
    reason = marker.kwargs.get("reason")
    if reason is None and marker.args:
        reason = marker.args[0]
    return isinstance(reason, str) and bool(reason.strip())


def _resolved_fixturedefs(item):
    """Return the fixture definitions an item resolved, override parents included.

    The item's top request records every fixture resolved for it: requested as
    an argument, by another fixture, or by name with ``getfixturevalue`` in a
    fixture or in the test body. An overriding fixture that requests its own
    name uses the definition it overrides, which the request does not record,
    so the chain is walked up. Items without a request (doctests, other
    non-Python items) fall back to their static fixture names.
    """
    request = getattr(item, "_request", None) or getattr(item, "fixture_request", None)
    defs = list(getattr(request, "_fixture_defs", {}).values())
    if request is None:
        info = getattr(item, "_fixtureinfo", None)
        if info is not None:
            for name in getattr(item, "fixturenames", ()):
                found = info.name2fixturedefs.get(name)
                if found:
                    defs.append(found[-1])
    manager = item.session._fixturemanager
    seen = []
    for fd in defs:
        while fd is not None and fd not in seen:
            seen.append(fd)
            if fd.argname not in fd.argnames:
                break
            chain = list(manager.getfixturedefs(fd.argname, item) or ())
            position = chain.index(fd) if fd in chain else 0
            fd = chain[position - 1] if position > 0 else None
    return seen


def fixture_charges(records: list[dict], instances: dict) -> None:
    """Add each test's share of fixture setup and its total cost to its record.

    A fixture instance's setup time is split equally among the distinct tests
    that used it, so a rerun of one test does not dilute its share; a
    function-scoped instance has exactly one user. A test that used an instance
    also used every instance set up inside its setup. An instance set up inside a
    call phase carries no charge, because its time is already in that call.
    The cost of a test is its call time plus its shares.
    """
    for rec in records:
        keys, todo = set(), list(rec.get("fixtures", ()))
        while todo:
            key = todo.pop()
            if key not in keys:
                keys.add(key)
                todo.extend(instances.get(key, {}).get("inside", ()))
        rec["fixtures"] = sorted(keys)
    users: dict[int, set] = {}
    for rec in records:
        for key in rec["fixtures"]:
            users.setdefault(key, set()).add(rec["nodeid"])
    for rec in records:
        share = 0.0
        heaviest = ("", 0.0)
        for key in rec.get("fixtures", ()):
            inst = instances.get(key)
            if inst is None:
                continue
            part = inst["setup_s"] / len(users[key])
            share += part
            if part > heaviest[1]:
                heaviest = (inst["name"], part)
        rec["setup_share_s"] = share
        rec["cost_s"] = rec["call_s"] + share
        rec["heaviest_fixture"] = heaviest[0]


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
        self.module_violations: dict[str, list[str]] = {}
        self.module_compiles: dict[str, int] = {}
        self.module_compile_s: dict[str, float] = {}
        self.module_cache_loads: dict[str, int] = {}
        self.worker_summaries: list[dict] = []
        self.current_tier = "unit"
        self.current_test = ""
        self.start_time = time.perf_counter()
        self._stop = threading.Event()
        self._thread = None
        self._base_compiles = 0
        self._base_compile_s = 0.0
        self._call_compiles: dict[str, tuple[int, float]] = {}
        # Fixture instances by serial number and the live serial of each
        # fixture definition.
        self._fixture_instances: dict[int, dict] = {}
        self._live_fixture: dict = {}
        self._fixture_serial = 0
        self._fixture_stack: list[list[float]] = []
        self._in_call = False
        self.fixture_violations: list[str] = []

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
        item._testbudget_record = None
        item._testbudget_fixtures = []
        rss_before = current_rss_bytes()
        result = yield
        rss_after = current_rss_bytes()
        record = getattr(item, "_testbudget_record", None)
        if record is not None:
            record["rss_growth_mb"] = (rss_after - rss_before) / 1024**2
        self.check_ceiling(item)
        return result

    @pytest.hookimpl(wrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        """Time each fixture instance's own setup.

        Fixtures it requests as arguments are set up before this hook runs; a
        fixture it pulls in with ``getfixturevalue`` is set up inside it. That
        nested time is subtracted here and charged to the nested instance. Every
        fixture looked up during this setup, set up now or already cached, is
        linked to this instance, and every test that uses this instance also
        uses the linked ones, because a later test reuses the cached value
        without resolving them again.
        """
        if not self.cfg.charge_fixture_setup:
            return (yield)
        resolved = getattr(request, "_fixture_defs", None)
        known = set(resolved) if isinstance(resolved, dict) else set()
        self._fixture_stack.append([0.0, []])
        start = time.perf_counter()
        try:
            return (yield)
        finally:
            elapsed = time.perf_counter() - start
            nested, inside = self._fixture_stack.pop()
            if isinstance(resolved, dict):
                # Fixtures looked up by name during this setup, whether set up
                # now or already cached, are part of this instance.
                for name in set(resolved) - known:
                    serial = self._live_fixture.get(resolved[name])
                    if serial is not None and serial not in inside:
                        inside.append(serial)
            self._fixture_serial += 1
            serial = self._fixture_serial
            if self._fixture_stack:
                self._fixture_stack[-1][0] += elapsed
                self._fixture_stack[-1][1].append(serial)
            self._fixture_instances[serial] = {
                "name": fixturedef.argname,
                "scope": fixturedef.scope,
                "setup_s": 0.0 if self._in_call else max(elapsed - nested, 0.0),
                "inside": inside,
            }
            self._live_fixture[fixturedef] = serial

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        """Count compiles that happen during the call phase."""
        before = compiles.compile_count()
        before_s = compiles.compile_seconds()
        self._in_call = True
        try:
            return (yield)
        finally:
            self._in_call = False
            if self.cfg.charge_fixture_setup:
                # Taken as the call ends, before any teardown hook can finish
                # or replace a fixture.
                item._testbudget_fixtures = [
                    self._live_fixture[fd]
                    for fd in _resolved_fixturedefs(item)
                    if fd.cached_result is not None and fd in self._live_fixture
                ]
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
            "limit_s": limit_s,
            "passed": report.outcome == "passed",
            "fixtures": getattr(item, "_testbudget_fixtures", []),
        }
        self.records.append(item._testbudget_record)
        if report.outcome != "passed":
            return report
        problems = []
        tmarker = item.get_closest_marker(tier) if tier != "unit" else None
        if self.cfg.require_tier_reason and tmarker and not _has_reason(tmarker):
            problems.append(
                f"testbudget: this test's {tier} mark gives no reason; write "
                f'@pytest.mark.{tier}(reason="...") saying why it cannot be a '
                "cheaper tier."
            )
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

    def judge_fixture_charges(self) -> None:
        """Charge fixture setup to this process's tests and record violations."""
        if not self.cfg.charge_fixture_setup:
            return
        fixture_charges(self.records, self._fixture_instances)
        for rec in self.records:
            limit = rec["limit_s"]
            if not rec["passed"] or rec["call_s"] > limit or rec["cost_s"] <= limit:
                continue
            self.fixture_violations.append(
                f"{rec['nodeid']}: testbudget: this {rec['tier']} test costs "
                f"{rec['cost_s']:.2f} s with its share of fixture setup (call "
                f"{rec['call_s']:.2f} s, fixtures {rec['setup_share_s']:.2f} s, "
                f"most from {rec['heaviest_fixture']!r}), over its budget of "
                f"{limit:.1f} s. Share the fixture across tests or move the test "
                "to a slower tier with a reason."
            )

    def summary(self) -> dict:
        """Return the merged session summary, including worker results under xdist."""
        records = list(self.records)
        peak = peak_rss_bytes()
        n_compiles = compiles.compile_count() - self._base_compiles
        compile_s = compiles.compile_seconds() - self._base_compile_s
        violations = (
            list(self.violations)
            + list(self.fixture_violations)
            + [m for msgs in self.module_violations.values() for m in msgs]
        )
        fixture_violations = list(self.fixture_violations)
        module_compiles = dict(self.module_compiles)
        module_compile_s = dict(self.module_compile_s)
        module_cache_loads = dict(self.module_cache_loads)
        for worker in self.worker_summaries:
            records += worker["records"]
            peak = max(peak, worker["peak_rss_bytes"])
            n_compiles += worker["compiles"]
            compile_s += worker.get("compile_s", 0.0)
            violations += worker["violations"]
            violations += worker.get("fixture_violations", [])
            fixture_violations += worker.get("fixture_violations", [])
            for name, count in worker["module_compiles"].items():
                module_compiles[name] = module_compiles.get(name, 0) + count
            for name, seconds in worker.get("module_compile_s", {}).items():
                module_compile_s[name] = module_compile_s.get(name, 0.0) + seconds
            for name, count in worker.get("module_cache_loads", {}).items():
                module_cache_loads[name] = module_cache_loads.get(name, 0) + count
        tiers = {tier: 0 for tier in ("unit", "medium", "slow")}
        for tier in {rec["nodeid"]: rec["tier"] for rec in records}.values():
            tiers[tier] = tiers.get(tier, 0) + 1
        return {
            "records": records,
            "tiers": tiers,
            "fixture_violations": fixture_violations,
            "peak_rss_bytes": peak,
            "compiles": n_compiles,
            "violations": violations,
            "module_compiles": module_compiles,
            "compile_s": compile_s,
            "module_compile_s": module_compile_s,
            "module_cache_loads": module_cache_loads,
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
        self.judge_fixture_charges()
        if hasattr(self.config, "workerinput"):
            self.config.workeroutput["testbudget"] = {
                "records": self.records,
                "fixture_violations": self.fixture_violations,
                "peak_rss_bytes": peak_rss_bytes(),
                "compiles": compiles.compile_count() - self._base_compiles,
                "violations": self.violations,
                "module_compiles": self.module_compiles,
                "compile_s": compiles.compile_seconds() - self._base_compile_s,
                "module_compile_s": self.module_compile_s,
                "module_cache_loads": self.module_cache_loads,
            }
            return
        self.final = self.summary()
        if self.worker_summaries:
            self._check_aggregated_modules(session)
        if (
            self.final["fixture_violations"]
            and self.cfg.mode == "enforce"
            and not session.exitstatus
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        if self.cfg.trend and self.final["records"]:
            self._write_trend(self.final)
        path = self.config.getoption("budget_json")
        if path:
            self._write_json(Path(path), self.final)

    def _check_aggregated_modules(self, session) -> None:
        """Judge module budgets on totals summed over xdist workers."""
        final = self.final
        for name, count in sorted(final["module_compiles"].items()):
            seconds = final["module_compile_s"].get(name, 0.0)
            loads = final["module_cache_loads"].get(name, 0)
            messages = module_budget_messages(self.cfg, name, count, seconds, loads)
            final["violations"].extend(messages)
            if messages and self.cfg.module_enforced() and not session.exitstatus:
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
            "tiers": final["tiers"],
            "slowest": [[r["nodeid"], round(r["call_s"], 2)] for r in slowest],
        }
        try:
            trend.append_record(self.cfg.resolved_trend_file(), record)
        except OSError as err:
            sys.stderr.write(f"testbudget: could not write trend file: {err}\n")

    def _write_json(self, path: Path, final: dict) -> None:
        """Write the merged session summary for tools that set budgets from it."""
        payload = {
            "library": self.cfg.library,
            "sha": trend.git_sha(self.root),
            "mode": self.cfg.mode,
            "module_mode": self.cfg.module_mode or self.cfg.mode,
            "wall_s": final["wall_s"],
            "peak_rss_gb": final["peak_rss_bytes"] / GB,
            "compiles": final["compiles"],
            "compile_s": final["compile_s"],
            "tiers": final["tiers"],
            "violations": final["violations"],
            "module_compiles": final["module_compiles"],
            "module_compile_s": final["module_compile_s"],
            "module_cache_loads": final["module_cache_loads"],
            "records": [
                {k: v for k, v in rec.items() if k != "fixtures"}
                for rec in final["records"]
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        except OSError as err:
            sys.stderr.write(f"testbudget: could not write {path}: {err}\n")

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
        tiers = final["tiers"]
        write(
            f"tiers: unit={tiers.get('unit', 0)} medium={tiers.get('medium', 0)} "
            f"slow={tiers.get('slow', 0)}"
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
        charged = [r for r in records if r.get("setup_share_s", 0.0) > 0.0]
        if charged:
            write("most fixture setup charged to one test:")
            for rec in sorted(charged, key=lambda r: r["setup_share_s"], reverse=True)[
                :TOP_N
            ]:
                write(
                    f"  {rec['setup_share_s']:7.2f} s  ({rec['heaviest_fixture']}) "
                    f"{rec['nodeid']}"
                )
        biggest = sorted(records, key=lambda r: r["compiles"], reverse=True)[:TOP_N]
        write("most compiles in one test:")
        for rec in biggest:
            write(f"  {rec['compiles']:5d}  {rec['nodeid']}")
