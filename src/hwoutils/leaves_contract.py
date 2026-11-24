"""Pytest plugin: record the leaf-rule violations of every Module a suite builds.

Load it from a library's ``tests/conftest.py``::

    pytest_plugins = ["hwoutils.leaves_contract"]

For the test session it installs a check on ``eqx.Module.__check_init__``,
which Equinox runs for every Module constructed, including subclasses with
their own ``__check_init__``. The check inspects the non-static fields of
each instance owned by the library under test (any class of its MRO defined
in one of the library's packages) with :func:`hwoutils.leaves.check_tree`'s
rule: a NumPy leaf, a Python number, a weakly typed array, or a float dtype
other than the active one. Nested Modules are checked when they are built,
not again inside their parent. The active dtype is read when the object is
built, so a test that scopes ``jax.enable_x64()`` is checked against the
precision in force there.

The check records and never raises: a raise inside construction would be
swallowed by a library ``try/except`` or matched by ``pytest.raises``. This
version reports only. The terminal summary lists each violating field once
with the number of constructions; ``--leaf-report PATH`` also writes every
record and the concrete classes no test built as JSON (one file per xdist
worker, suffixed with the worker id).

The library under test is the ``leaf_contract_packages`` ini option, or the
``[project] name`` of the root ``pyproject.toml`` when that is unset.
"""

import inspect
import json
import os
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import pytest

from hwoutils.leaves import _check, is_owned

_KEY = "leaf-contract"


def pytest_addoption(parser):
    """Register the package setting and the report option.

    Args:
        parser: The pytest option parser.
    """
    parser.addini(
        "leaf_contract_packages",
        type="args",
        default=[],
        help="top-level packages whose Modules the leaf contract records",
    )
    parser.getgroup("leaf-contract").addoption(
        "--leaf-report",
        default=None,
        metavar="PATH",
        help="write the leaf-contract records as JSON to PATH",
    )


def _project_packages(config) -> list[str]:
    path = Path(config.rootpath) / "pyproject.toml"
    if not path.is_file():
        return []
    name = tomllib.loads(path.read_text()).get("project", {}).get("name")
    return [name.replace("-", "_")] if name else []


def pytest_configure(config):
    """Install the construction check for the session.

    Args:
        config: The pytest configuration object.
    """
    packages = config.getini("leaf_contract_packages") or _project_packages(config)
    plugin = LeafContract(frozenset(packages), config.getoption("--leaf-report"))
    config.pluginmanager.register(plugin, _KEY)
    plugin.install()


def pytest_unconfigure(config):
    """Remove the construction check.

    Args:
        config: The pytest configuration object.
    """
    plugin = config.pluginmanager.get_plugin(_KEY)
    if plugin is not None:
        plugin.uninstall()
        config.pluginmanager.unregister(plugin)


def _is_abstract(cls) -> bool:
    return (
        inspect.isabstract(cls)
        or bool(getattr(cls, "__abstractvars__", ()))
        or bool(getattr(cls, "__abstractclassvars__", ()))
    )


def _subclasses(cls):
    seen, stack = set(), [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def _name(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class LeafContract:
    """Records leaf-rule violations by test and by class field."""

    def __init__(self, packages: frozenset[str], report_path: str | None):
        """Start an empty record.

        Args:
            packages: Top-level packages of the library under test.
            report_path: Where to write the JSON report, or None.
        """
        self.packages = packages
        self.report_path = report_path
        self.current = "<collection>"
        self.by_test = defaultdict(list)
        self.by_field = Counter()
        self.errors = Counter()
        self.constructed = set()
        self._previous = None
        self._installed = False

    def install(self) -> None:
        """Put the check on ``eqx.Module`` for the session."""
        if not self.packages or self._installed:
            return
        self._previous = eqx.Module.__dict__.get("__check_init__")
        plugin = self

        def __check_init__(module):
            plugin.record(module)

        eqx.Module.__check_init__ = __check_init__
        self._installed = True

    def uninstall(self) -> None:
        """Restore ``eqx.Module`` as it was before the session."""
        if not self._installed:
            return
        if self._previous is None:
            del eqx.Module.__check_init__
        else:
            eqx.Module.__check_init__ = self._previous
        self._installed = False

    def record(self, module) -> None:
        """Check one constructed Module and record what it breaks; never raise.

        Args:
            module: The Module just constructed.
        """
        try:
            if not is_owned(module, self.packages):
                return
            cls = type(module)
            self.constructed.add(cls)
            for v in _check(module, lambda _: False, ()):
                entry = (_name(cls), v.path, v.type, v.dtype, v.reason)
                self.by_field[entry] += 1
                self.by_test[self.current].append(entry)
        except Exception as error:
            self.errors[f"{type(error).__name__}: {error}"] += 1

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem) -> Iterator[None]:
        """Attribute constructions to the running test.

        Args:
            item: The test item.
            nextitem: The next test item; unused.

        Yields:
            Control to the test protocol.
        """
        self.current = item.nodeid
        yield
        self.current = "<between tests>"

    def pytest_runtest_makereport(self, item, call) -> None:
        """Attach the test's violations to its report, without changing it.

        Args:
            item: The test item.
            call: The call information.
        """
        found = self.by_test.get(item.nodeid)
        if call.when == "teardown" and found:
            item.add_report_section(
                "teardown",
                "leaf contract",
                "\n".join(" ".join(map(str, entry)) for entry in sorted(set(found))),
            )

    def unconstructed(self) -> list[str]:
        """Concrete Module classes of the library that nothing built.

        Returns:
            Their qualified names, sorted. Only imported classes are known.
        """
        return sorted(
            _name(cls)
            for cls in _subclasses(eqx.Module)
            if cls.__module__.partition(".")[0] in self.packages
            and not _is_abstract(cls)
            and cls not in self.constructed
        )

    def pytest_sessionfinish(self, session) -> None:
        """Write the JSON report when asked.

        Args:
            session: The pytest session.
        """
        if not self.report_path:
            return
        path = self.report_path
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if worker:
            path = f"{path}.{worker}"
        fields = ("class", "path", "type", "dtype", "reason")
        report = {
            "packages": sorted(self.packages),
            "violations": [
                dict(zip(fields, entry, strict=True), constructions=count)
                for entry, count in sorted(self.by_field.items())
            ],
            "tests": {k: sorted(map(list, set(v))) for k, v in self.by_test.items()},
            "constructed": sorted(map(_name, self.constructed)),
            "unconstructed": self.unconstructed(),
            "errors": dict(self.errors),
        }
        Path(path).write_text(json.dumps(report, indent=1))

    def pytest_terminal_summary(self, terminalreporter) -> None:
        """Print each violating field once, with its construction count.

        Args:
            terminalreporter: The terminal reporter.
        """
        if not self._installed and not self.by_field:
            return
        write = terminalreporter.write_line
        terminalreporter.section("leaf contract (report only)")
        classes = {entry[0] for entry in self.by_field}
        write(
            f"{len(self.by_field)} violating fields in {len(classes)} classes; "
            f"{len(self.constructed)} classes constructed; "
            f"{len(self.unconstructed())} concrete classes not constructed"
        )
        for (cls, path, kind, dtype, reason), count in sorted(self.by_field.items()):
            write(f"  {cls}{path}: {kind} {dtype} ({reason}) x{count}")
        for error, count in self.errors.items():
            write(f"  check failed: {error} x{count}")
