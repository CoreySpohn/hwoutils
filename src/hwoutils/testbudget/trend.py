"""Trend file: one JSON line per pytest session, plus the growth check."""

import json
import statistics
import subprocess
from pathlib import Path

METRICS = ("wall_s", "peak_rss_gb", "compiles", "tests")
GROWTH_LIMIT = 0.20
WINDOW = 10


def git_sha(root: Path) -> str:
    """Return the short commit hash of ``root``, or an empty string."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def append_record(path: Path, record: dict) -> None:
    """Append one record to the JSONL trend file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_records(path: Path) -> list[dict]:
    """Read every record, skipping blank or damaged lines."""
    records = []
    for line in Path(path).read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def growth_flags(records: list[dict], window: int = WINDOW) -> dict[str, float]:
    """Flag metrics where the newest run is over 20 percent above the prior mean.

    Only runs with the same ``args`` as the newest are compared, because a
    one-file run and a full run are not comparable. Needs at least three prior
    runs. Returns metric name -> fractional growth for the flagged metrics.
    """
    if not records:
        return {}
    latest = records[-1]
    same = [r for r in records[:-1] if r.get("args") == latest.get("args")][-window:]
    if len(same) < 3:
        return {}
    flags = {}
    for metric in METRICS:
        base = statistics.fmean(r.get(metric, 0) for r in same)
        if base > 0 and latest.get(metric, 0) / base - 1.0 > GROWTH_LIMIT:
            flags[metric] = latest[metric] / base - 1.0
    return flags


def format_trend(records: list[dict], last: int = 20) -> str:
    """Return a table of the most recent runs, then any growth flags."""
    lines = [
        f"{'time':20} {'sha':8} {'tests':>6} {'wall_s':>8} {'rss_gb':>7} "
        f"{'compiles':>8}  args"
    ]
    for rec in records[-last:]:
        lines.append(
            f"{rec.get('time', '')[:19]:20} {rec.get('sha', ''):8} "
            f"{rec.get('tests', 0):6d} "
            f"{rec.get('wall_s', 0):8.1f} {rec.get('peak_rss_gb', 0):7.2f} "
            f"{rec.get('compiles', 0):8d}  {rec.get('args', '')}"
        )
    flags = growth_flags(records)
    if flags:
        for metric, frac in flags.items():
            lines.append(
                f"FLAG {metric}: {frac:+.0%} over the mean of the last {WINDOW} "
                "comparable runs"
            )
    else:
        lines.append("no growth flags")
    return "\n".join(lines)
