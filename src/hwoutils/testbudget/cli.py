"""Command line entry: ``testbudget trend <library-or-path>``."""

import argparse
import sys
from pathlib import Path

from hwoutils.testbudget.config import BudgetConfig
from hwoutils.testbudget.trend import format_trend, growth_flags, read_records


def main(argv: list[str] | None = None) -> int:
    """Print the trend for a library; exit 1 if any metric grew over 20 percent."""
    parser = argparse.ArgumentParser(prog="testbudget", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trend = sub.add_parser("trend", help="print recent runs and flag growth")
    trend.add_argument("target", help="library name or path to a .jsonl file")
    trend.add_argument("--last", type=int, default=20, help="rows to print")
    args = parser.parse_args(argv)

    path = Path(args.target).expanduser()
    if path.suffix != ".jsonl":
        path = BudgetConfig(library=args.target).resolved_trend_file()
    if not path.is_file():
        print(f"no trend file at {path}", file=sys.stderr)
        return 2
    records = read_records(path)
    print(format_trend(records, last=args.last))
    return 1 if growth_flags(records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
