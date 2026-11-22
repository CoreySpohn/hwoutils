#!/usr/bin/env python3
"""Scan public repository text for private workspace references.

Run ``python -m hwoutils.internal_refs [FILE ...]`` or load this module through
``pytest_plugins`` to collect one repository-wide ``internal-refs`` item.
Append ``internal-ref-ok`` to a deliberately functional line to waive it.
"""

import re
import subprocess
import sys
from pathlib import Path

WAIVER = "internal-ref-ok"
PATTERNS = [
    r"mission-control",  # internal-ref-ok
    r"research-brain",  # internal-ref-ok
    r"physicaloptix-[a-z]+",  # internal-ref-ok
    r"\bbrain/",  # internal-ref-ok
    r"deep-thoughts",  # internal-ref-ok
    r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\.md\b",  # internal-ref-ok
    r"eac1_dlux",  # internal-ref-ok
    r"Tier[- ][GR]\b",  # internal-ref-ok
    r"(?i)\bgreenfield\b",  # internal-ref-ok
    r"(?i)\bsynthesis famil(?:y|ies)\b",  # internal-ref-ok
    r"hwosim-(?:paper|build)",  # internal-ref-ok
    r"spohnbook-build",  # internal-ref-ok
    r"zodi-library",  # internal-ref-ok
]
COMPILED = [(pattern, re.compile(pattern)) for pattern in PATTERNS]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".cff",
    ".ipynb",
    ".txt",
}


def scan_file(path):
    """Return (lineno, pattern, stripped line) hits for one UTF-8 text file.

    Args:
        path: Path of the file to scan.

    Returns:
        One ``(lineno, pattern, line)`` tuple per matching pattern and line, with a
        one-based line number. An unreadable file gives one ``(0, "<unreadable>",
        "")`` entry.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as err:
        print(f"{path}: unreadable ({err})")
        return [(0, "<unreadable>", "")]
    return [
        (lineno, pattern, line.strip())
        for lineno, line in enumerate(text.splitlines(), start=1)
        if WAIVER not in line
        for pattern, regex in COMPILED
        if regex.search(line)
    ]


def scan(paths):
    """Return (path, lineno, pattern, line) hits across the supplied files.

    Args:
        paths: Paths of the files to scan.

    Returns:
        One ``(path, lineno, pattern, line)`` tuple per hit.
    """
    return [
        (Path(path), lineno, pattern, line)
        for path in paths
        for lineno, pattern, line in scan_file(path)
    ]


def tracked_text_files(repo):
    """List existing tracked text files in a Git repository, except CHANGELOG.md.

    Args:
        repo: Root directory of the Git repository.

    Returns:
        Paths of the tracked files whose suffix is a scanned text suffix.
    """
    repo = Path(repo)
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return [
        repo / name
        for name in tracked
        if Path(name).suffix in TEXT_SUFFIXES
        and name != "CHANGELOG.md"
        and (repo / name).is_file()
    ]


def _format_hits(hits):
    lines = [
        f"{path}:{lineno}: matches {pattern!r}: {line}"
        for path, lineno, pattern, line in hits
    ]
    lines.append(
        "\nInternal research references do not belong in library code. "
        f"Rephrase self-contained, or append '{WAIVER}' to a "
        "deliberately functional line."
    )
    return "\n".join(lines)


def main(argv=None):
    """Report hits and return 1 on failure, or return 0 for clean files.

    Args:
        argv: File paths to scan. Defaults to ``sys.argv[1:]``; with no paths, the
            tracked text files of the current directory are scanned.

    Returns:
        Process exit status: 1 when any hit is found, 0 otherwise.
    """
    args = sys.argv[1:] if argv is None else argv
    hits = scan(args if args else tracked_text_files(Path.cwd()))
    if hits:
        print(_format_hits(hits))
        return 1
    return 0


def pytest_configure(config):
    """Register collection support only when pytest loads this module as a plugin.

    Args:
        config: The pytest configuration object.
    """
    # The CLI and scanner also work without pytest installed.
    import pytest

    class InternalRefsItem(pytest.Item):
        def runtest(self):
            hits = scan(tracked_text_files(self.config.rootpath))
            if hits:
                pytest.fail(_format_hits(hits), pytrace=False)

        def reportinfo(self):
            return self.config.rootpath, None, self.name

    class InternalRefsPlugin:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, items):
            # Add before keyword, marker and node-id deselection; xdist schedules
            # the single item just like any other collected test.
            items.append(
                InternalRefsItem.from_parent(
                    session, name="internal-refs", nodeid="internal-refs"
                )
            )

    config.pluginmanager.register(InternalRefsPlugin(), "internal-refs-collector")


if __name__ == "__main__":
    sys.exit(main())
