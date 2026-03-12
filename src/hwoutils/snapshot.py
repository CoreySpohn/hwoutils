"""Workspace snapshot utilities for reproducible runs.

Records the state of all git repos in the hwo-dev workspace so that
any pipeline or analysis run can be traced back to an exact set of
library versions.

Usage in scripts:

    # Development mode: just record what we used
    from hwoutils.snapshot import record

    record("data/manifest.json")

    # Production mode: refuse to run if anything is dirty
    from hwoutils.snapshot import require_clean, record

    require_clean()  # raises SystemExit if any repo has uncommitted changes
    record("data/manifest.json")
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Libraries that are part of the HWO simulation suite
WORKSPACE_REPOS = [
    "yippy",
    "orbix",
    "coronagraphoto",
    "coronalyze",
    "hwoutils",
    "yieldplotlib",
    "hwostyle",
    "exosims-plugins",
    "hwo-mission-control",
    "eacy",
    "pyEDITH",
    "EXOSIMS",
    "ExoVista",
]


def _find_workspace_root():
    """Walk up from hwoutils to find the workspace root."""
    current = Path(__file__).resolve().parent
    # hwoutils/src/hwoutils/snapshot.py -> hwoutils -> hwo-dev
    for _ in range(5):
        if (current / "pyproject.toml").exists() and (current / ".venv").exists():
            return current
        current = current.parent
    msg = "Could not find hwo-dev workspace root"
    raise RuntimeError(msg)


def _git_info(repo_path):
    """Get git state for a single repository."""
    if not (repo_path / ".git").exists():
        return None

    def _run(cmd):
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    if not commit:
        return None

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    full_hash = _run(["git", "rev-parse", "HEAD"])

    # Check for uncommitted changes
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status)

    # Try to get the package version
    version = None
    try:
        # Try importlib.metadata first
        import importlib.metadata

        pkg_name = repo_path.name.replace("-", "_").replace("_", "-")
        # Try a few name variations
        for name in [repo_path.name, pkg_name]:
            try:
                version = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    except Exception:
        pass

    info = {
        "commit": commit,
        "commit_full": full_hash,
        "branch": branch,
        "dirty": dirty,
    }
    if version:
        info["version"] = version
    return info


def scan_workspace(workspace_root=None):
    """Scan all repos in the workspace and return their git state.

    Returns a dict with timestamp, python version, and per-repo info.
    """
    if workspace_root is None:
        workspace_root = _find_workspace_root()
    workspace_root = Path(workspace_root)

    repos = {}
    for name in WORKSPACE_REPOS:
        repo_path = workspace_root / name
        if repo_path.exists():
            info = _git_info(repo_path)
            if info:
                repos[name] = info

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(workspace_root),
        "python": sys.version.split()[0],
        "repos": repos,
    }


def record(output_path, workspace_root=None):
    """Record a snapshot of the workspace state to a JSON file.

    Args:
        output_path: Path to write the manifest JSON.
        workspace_root: Optional workspace root override.

    Returns:
        The snapshot dict.
    """
    snapshot = scan_workspace(workspace_root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    dirty_repos = [name for name, info in snapshot["repos"].items() if info["dirty"]]
    if dirty_repos:
        print(
            f"Warning: {len(dirty_repos)} repo(s) have uncommitted "
            f"changes: {', '.join(dirty_repos)}",
            file=sys.stderr,
        )

    print(f"Snapshot recorded to {output_path}")
    return snapshot


def require_clean(workspace_root=None, repos=None):
    """Require all workspace repos to be clean (no uncommitted changes).

    Call this at the top of production/final pipeline scripts. If any
    repo has uncommitted changes, prints a clear error and exits.

    Args:
        workspace_root: Optional workspace root override.
        repos: Optional list of repo names to check. If None, checks all.

    Raises:
        SystemExit: If any repo has uncommitted changes.
    """
    snapshot = scan_workspace(workspace_root)

    dirty = []
    for name, info in snapshot["repos"].items():
        if repos and name not in repos:
            continue
        if info["dirty"]:
            dirty.append(name)

    if dirty:
        print(
            "\n"
            "=" * 60 + "\n"
            "  PRODUCTION RUN BLOCKED: uncommitted changes detected\n"
            "=" * 60 + "\n",
            file=sys.stderr,
        )
        for name in dirty:
            info = snapshot["repos"][name]
            print(
                f"  {name:30s} [{info['branch']}] {info['commit']} DIRTY",
                file=sys.stderr,
            )
        print(
            "\nCommit or stash all changes before running in production mode.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print clean confirmation
    n_repos = len(repos) if repos else len(snapshot["repos"])
    print(f"All {n_repos} repos clean. Production run approved.")
    return snapshot
