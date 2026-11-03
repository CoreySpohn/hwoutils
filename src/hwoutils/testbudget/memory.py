"""Resident memory readings with no hard dependency beyond the standard library."""

import os
import resource
import subprocess
import sys

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only without psutil
    psutil = None

GB = 1024**3


def peak_rss_bytes() -> int:
    """Return the peak resident size of this process in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def current_rss_bytes() -> int:
    """Return the current resident size of this process in bytes."""
    if psutil is not None:
        return psutil.Process().memory_info().rss
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return int(out) * 1024 if out else peak_rss_bytes()
