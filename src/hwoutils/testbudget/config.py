"""Configuration for the test budget guard, read from ``[tool.testbudget]``."""

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

MODES = ("report", "enforce")


@dataclass(frozen=True)
class BudgetConfig:
    """Budgets and switches; every field maps to a ``[tool.testbudget]`` key.

    Attributes:
        library: Name used in the trend file name. Defaults to the project name.
        mode: ``report`` records and prints violations; ``enforce`` also fails
            the offending test (or module teardown).
        unit_s: Call-time budget for an unmarked test, seconds. Includes the
            first compile of the jitted functions the test exercises.
        medium_s: Call-time budget for a ``medium`` test, seconds.
        slow_s: Call-time budget for a ``slow`` test, seconds.
        rss_ceiling_gb: Hard abort when one process exceeds this resident size.
            Applies in every mode. Zero disables the ceiling.
        slow_rss_ceiling_gb: Ceiling while a ``slow`` test runs. Zero means
            use ``rss_ceiling_gb``.
        rss_poll_s: Watchdog polling period, seconds.
        rss_growth_report_mb: Tests that grow resident memory by at least this
            much are listed in the report.
        module_compile_budget: Maximum compiles in one test module. Zero
            disables the check.
        module_compile_budget_by_file: Per-file overrides keyed by file name.
        clear_caches: Run ``jax.clear_caches()`` and ``gc.collect()`` after
            each test module.
        compile_cache: Enable the persistent JAX compilation cache.
        compile_cache_dir: Cache directory, relative to the project root. The
            ``TESTBUDGET_CACHE_DIR`` environment variable overrides it.
        trend: Append a session summary line to the trend file.
        trend_file: Trend file path. ``TESTBUDGET_TREND_FILE`` overrides it.
            Empty means ``~/hwo-data/testbudget/<library>.jsonl``.
    """

    library: str = ""
    mode: str = "report"
    unit_s: float = 5.0
    medium_s: float = 30.0
    slow_s: float = 300.0
    rss_ceiling_gb: float = 12.0
    slow_rss_ceiling_gb: float = 0.0
    rss_poll_s: float = 0.5
    rss_growth_report_mb: float = 500.0
    module_compile_budget: int = 0
    module_compile_budget_by_file: dict = field(default_factory=dict)
    clear_caches: bool = False
    compile_cache: bool = False
    compile_cache_dir: str = ".jax_cache"
    trend: bool = True
    trend_file: str = ""

    def tier_budget_s(self, tier: str) -> float:
        """Return the call-time budget for a tier name."""
        return {"unit": self.unit_s, "medium": self.medium_s, "slow": self.slow_s}[tier]

    def ceiling_gb(self, tier: str) -> float:
        """Return the resident-size ceiling that applies while a tier runs."""
        if tier == "slow" and self.slow_rss_ceiling_gb > 0:
            return self.slow_rss_ceiling_gb
        return self.rss_ceiling_gb

    def resolved_trend_file(self) -> Path:
        """Return the trend file path after the environment override."""
        override = os.environ.get("TESTBUDGET_TREND_FILE") or self.trend_file
        if override:
            return Path(override).expanduser()
        return Path("~/hwo-data/testbudget").expanduser() / f"{self.library}.jsonl"

    def resolved_cache_dir(self, root: Path) -> Path:
        """Return the compilation cache directory after the environment override."""
        override = os.environ.get("TESTBUDGET_CACHE_DIR") or self.compile_cache_dir
        path = Path(override).expanduser()
        return path if path.is_absolute() else root / path


def load_config(root: Path) -> BudgetConfig:
    """Read ``[tool.testbudget]`` from ``root/pyproject.toml``.

    Unknown keys and a bad mode raise ``ValueError`` so a typo does not
    silently leave a budget at its default.
    """
    pyproject = Path(root) / "pyproject.toml"
    data = {}
    project_name = ""
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            parsed = tomllib.load(handle)
        data = dict(parsed.get("tool", {}).get("testbudget", {}))
        project_name = parsed.get("project", {}).get("name", "")
    known = {f.name for f in fields(BudgetConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown [tool.testbudget] keys: {', '.join(unknown)}")
    cfg = BudgetConfig(**data)
    if cfg.mode not in MODES:
        raise ValueError(
            f"[tool.testbudget] mode must be one of {MODES}, got {cfg.mode!r}"
        )
    if not cfg.library:
        cfg = BudgetConfig(**{**data, "library": project_name or Path(root).name})
    return cfg
