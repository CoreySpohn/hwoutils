#!/usr/bin/env python3
"""Report precision-contract violations in Python source and tests.

Run ``python -m hwoutils.precision_guard [--enforce] [FILE ...]`` or load
this module through ``pytest_plugins``. Add ``# precision-ok: <reason>``
on a finding's line to waive it. For pytest, use ``--precision-guard-enforce``
or set ``precision_guard_enforce = true`` to make findings fail.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

WAIVER = "precision-ok:"
_WAIVER_COMMENT = re.compile(r"#\s*precision-ok:\s*\S")
_TOL_COMMENT = re.compile(r"#\s*tol:\s*(physical|statistical|algorithm),\s*\S")
_TOLERANCES = {"atol", "rtol", "tol"}
_DTYPES = {"float", "complex", "builtins.float", "builtins.complex"}
_DTYPES |= {"jax.numpy.float64", "jax.numpy.complex128"}
_DTYPES |= {"numpy.float64", "numpy.complex128"}
_BLOCKS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ClassDef,
    *(getattr(ast, name) for name in ("TryStar", "Match") if hasattr(ast, name)),
)
_FLAG_ATTRS = {"jax.config.jax_enable_x64", "jax.config.x64_enabled"}
_FLAG_SETTERS = {
    "jax.enable_x64",
    "jax.experimental.enable_x64",
    "jax.experimental.disable_x64",
    "jax.experimental.x64_context.enable_x64",
    "jax.experimental.x64_context.disable_x64",
    "hwoutils.enable_x64",
    "hwoutils.precision.enable_x64",
    "hwoutils.jax_config.enable_x64",
}
_FLAG_KEYWORDS = {
    "jax.config.update": ("name", "jax_enable_x64", True),
    "jax.config.read": ("name", "jax_enable_x64", False),
    "monkeypatch.setenv": ("name", "JAX_ENABLE_X64", True),
    "monkeypatch.delenv": ("name", "JAX_ENABLE_X64", True),
    "os.getenv": ("key", "JAX_ENABLE_X64", False),
    "os.putenv": ("name", "JAX_ENABLE_X64", True),
    "os.unsetenv": ("name", "JAX_ENABLE_X64", True),
    "os.environ.get": ("key", "JAX_ENABLE_X64", False),
    "os.environ.setdefault": ("key", "JAX_ENABLE_X64", True),
    "os.environ.pop": ("key", "JAX_ENABLE_X64", True),
}
_CLOSE = {
    "jax.numpy.allclose": ("rtol", "atol"),
    "jax.numpy.isclose": ("rtol", "atol"),
    "numpy.allclose": ("rtol", "atol"),
    "numpy.isclose": ("rtol", "atol"),
    "math.isclose": ("rel_tol", "abs_tol"),
    "allclose": ("rtol", "atol"),
    "isclose": ("rtol", "atol"),
}
_MESSAGES = {
    "x64-flag": "Set or read the x64 flag through hwoutils.precision.",
    "jnp-dtype": "Derive the dtype from the input precision.",
    "small-tolerance": "Derive the tolerance or use a named constant with a tol tag.",
    "default-close": "Specify both tolerances explicitly.",
    "test-x64": "Select the precision lane through the pytest plugin.",
}


def _scope(path):
    parts = path.absolute().parts
    for index, part in enumerate(parts[:-1]):
        if part in {"src", "tests"}:
            return part, parts[index + 1 :]
    return None, ()


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    return ""


def _imports(tree):
    # Recognize both direct imports and the conventional module spellings.
    names = {"jnp": "jax.numpy", "np": "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                names[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _literal(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def _numeric_literal(node):
    return isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float, complex)
    )


def _small_float(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and 0 < abs(node.value) <= 1e-5  # precision-ok: threshold for this source rule
    )


class _Guard(ast.NodeVisitor):
    def __init__(self, tree, comments, scope, parts):
        self.names = _imports(tree)
        self.comments = comments
        self.is_test = scope == "tests"
        self.conftest = self.is_test and parts[-1] == "conftest.py"
        self.flag_owner = scope == "src" and parts == ("hwoutils", "precision.py")
        self.deferred = 0
        self.hits = set()
        self.numpy_names = set()
        self.small_constants = set()
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            else:
                continue
            untagged_small = _small_float(stmt.value) and not _TOL_COMMENT.match(
                comments.get(stmt.end_lineno, "")
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    if untagged_small:
                        self.small_constants.add(target.id)
                    else:
                        self.small_constants.discard(target.id)

    def _qualified(self, node):
        name = _name(node)
        root, dot, suffix = name.partition(".")
        return self.names.get(root, root) + dot + suffix

    def _report(self, node, rule, call=None):
        lines = [node.lineno]
        if call is not None:
            lines.extend((call.lineno, call.end_lineno))
        if not any(
            _WAIVER_COMMENT.match(self.comments.get(line, "")) for line in lines
        ):
            self.hits.add((node.lineno, node.col_offset, rule))

    def _flag(self, node, *, setting):
        if self.is_test:
            if setting and (self.conftest or not self.deferred):
                self._report(node, "test-x64", node)
        elif not self.flag_owner:
            self._report(node, "x64-flag", node)

    def _flag_call(self, node, func):
        if func in _FLAG_SETTERS:
            self._flag(node, setting=True)
            return
        if func in _FLAG_KEYWORDS:
            keyword, flag_key, setting = _FLAG_KEYWORDS[func]
            key = (
                node.args[0]
                if node.args
                else next(
                    (arg.value for arg in node.keywords if arg.arg == keyword), None
                )
            )
            if _literal(key, flag_key):
                self._flag(node, setting=setting)
        if func == "os.environ.update":
            if any(key.arg == "JAX_ENABLE_X64" for key in node.keywords) or any(
                isinstance(arg, ast.Dict)
                and any(_literal(key, "JAX_ENABLE_X64") for key in arg.keys)
                for arg in node.args
            ):
                self._flag(node, setting=True)

    def _tolerance(self, value, call=None):
        if _small_float(value) or (
            isinstance(value, ast.Name) and value.id in self.small_constants
        ):
            self._report(value, "small-tolerance", call)
        else:
            for part in ast.iter_child_nodes(value):
                self._tolerance(part, call)

    def _numpy_value(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.numpy_names
        if isinstance(node, ast.Call):
            return self._qualified(node.func).startswith("numpy.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self._numpy_value(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self._numpy_value(node.left), self._numpy_value(node.right)
            return (left and (right or _numeric_literal(node.right))) or (
                right and _numeric_literal(node.left)
            )
        return False

    def _jax_dtype_call(self, node, func):
        if func.startswith("jax.numpy."):
            return True
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr != "astype"
            or func.startswith("numpy.")
        ):
            return False
        return not self._numpy_value(node.func.value)

    def visit_Assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        host_value = self._numpy_value(node.value)
        self.generic_visit(node)
        if host_value:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            self.numpy_names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )

    visit_AnnAssign = visit_Assign

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.numpy_names.discard(node.id)

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _BLOCKS):
            super().generic_visit(node)
            return
        # A binding inside a block may never run. Do not carry it to other blocks.
        outer = self.numpy_names
        outer.difference_update(
            part.id
            for part in ast.walk(node)
            if isinstance(part, ast.Name) and isinstance(part.ctx, (ast.Store, ast.Del))
        )
        for _, value in ast.iter_fields(node):
            self.numpy_names = outer.copy()
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    if not isinstance(child, ast.stmt):
                        self.numpy_names = outer.copy()
                    self.visit(child)
        self.numpy_names = outer

    def visit_Call(self, node: ast.Call) -> None:
        func = self._qualified(node.func)
        self._flag_call(node, func)
        if not self.is_test:
            if self._jax_dtype_call(node, func):
                dtype_args = node.args + [
                    key.value for key in node.keywords if key.arg == "dtype"
                ]
                for arg in dtype_args:
                    if self._qualified(arg) in _DTYPES:
                        self._report(arg, "jnp-dtype", node)
            for key in node.keywords:
                if key.arg in _TOLERANCES:
                    self._tolerance(key.value, node)
            if func in _CLOSE:
                supplied = {key.arg for key in node.keywords}
                if func != "math.isclose":
                    for index, arg in enumerate(node.args[:4]):
                        if isinstance(arg, ast.Starred):
                            break
                        if index >= 2:
                            supplied.add(_CLOSE[func][index - 2])
                            self._tolerance(arg, node)
                if not supplied.issuperset(_CLOSE[func]):
                    self._report(node, "default-close", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._qualified(node) in _FLAG_ATTRS:
            self._flag(node, setting=not isinstance(node.ctx, ast.Load))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        name = self._qualified(node.value)
        if name == "os.environ" and _literal(node.slice, "JAX_ENABLE_X64"):
            self._flag(node, setting=not isinstance(node.ctx, ast.Load))
        elif name == "jax.config.values" and _literal(node.slice, "jax_enable_x64"):
            self._flag(node, setting=False)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if not self.is_test:
            for operand in (node.left, *node.comparators):
                self._tolerance(operand)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.AST) -> None:
        # Default arguments and decorators can run at import time.
        for field, child in ast.iter_fields(node):
            nodes = child if isinstance(child, list) else [child]
            outer = self.numpy_names
            if field == "body":
                self.numpy_names = set()
            self.deferred += field == "body"
            for part in nodes:
                if isinstance(part, ast.AST):
                    self.visit(part)
            self.deferred -= field == "body"
            self.numpy_names = outer
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.numpy_names.discard(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


def scan_file(path):
    """Scan one Python file under a src or tests directory.

    Args:
        path: Python file path.

    Returns:
        One (line number, rule, source line) tuple per finding.
    """
    path = Path(path)
    scope, parts = _scope(path)
    if path.suffix != ".py" or scope is None:
        return []
    try:
        with tokenize.open(path) as stream:
            source = stream.read()
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return [(getattr(exc, "lineno", None) or 0, "scan-error", str(exc))]

    comments = {
        token.start[0]: token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    guard = _Guard(tree, comments, scope, parts)
    guard.visit(tree)
    lines = source.splitlines()
    return [
        (line, rule, lines[line - 1].strip()) for line, _, rule in sorted(guard.hits)
    ]


def scan(paths):
    """Scan Python files for precision-contract violations.

    Args:
        paths: Iterable of files to scan.

    Returns:
        One (path, line number, rule, source line) tuple per finding.
    """
    return [
        (Path(path), line, rule, source)
        for path in paths
        for line, rule, source in scan_file(path)
    ]


def _tracked_python(repo):
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "src", "tests"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        repo / name
        for name in result.stdout.split("\0")
        if name.endswith(".py") and (repo / name).is_file()
    ]


def _format_hits(hits):
    lines = [
        f"{path}:{line}: {rule}: {_MESSAGES.get(rule, 'Unable to scan file.')} {source}"
        for path, line, rule, source in hits
    ]
    lines.append(f"Append '# {WAIVER} <reason>' to waive a deliberate exception.")
    return "\n".join(lines)


def main(argv=None):
    """Print findings and optionally return a failing status.

    Args:
        argv: Arguments replacing sys.argv[1:]. Defaults to tracked Python
            files under src and tests when no file arguments are supplied.

    Returns:
        One if enforcement is enabled and findings exist, otherwise zero.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--enforce", action="store_true", help="Fail on any finding.")
    parser.add_argument("paths", nargs="*", metavar="FILE")
    args = parser.parse_args(argv)
    hits = scan(args.paths or _tracked_python(Path.cwd()))
    if hits:
        print(_format_hits(hits))
    return int(bool(hits) and args.enforce)


def pytest_addoption(parser):
    """Register enforcement options while keeping report mode the default.

    Args:
        parser: Pytest command-line and ini parser.
    """
    group = parser.getgroup("precision guard")
    group.addoption(
        "--precision-guard-enforce",
        action="store_true",
        help="Fail on precision findings.",
    )
    parser.addini(
        "precision_guard_enforce",
        "Fail on precision findings.",
        type="bool",
        default=False,
    )


def pytest_configure(config):
    """Register a single repository-wide precision check with pytest.

    Args:
        config: Pytest run configuration.
    """
    # The pre-commit hook only needs the standard library.
    import pytest

    class PrecisionGuardItem(pytest.Item):
        def runtest(self):
            hits = scan(_tracked_python(self.config.rootpath))
            if hits:
                message = _format_hits(hits)
                if self.config.getoption(
                    "precision_guard_enforce"
                ) or self.config.getini("precision_guard_enforce"):
                    pytest.fail(message, pytrace=False)
                self.add_report_section("call", "precision guard", message)

        def reportinfo(self):
            return self.config.rootpath, None, self.name

    class PrecisionGuardPlugin:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, items):
            items.append(
                PrecisionGuardItem.from_parent(
                    session, name="precision-guard", nodeid="precision-guard"
                )
            )

        def pytest_terminal_summary(self, terminalreporter):
            for report in terminalreporter.getreports("passed"):
                for heading, message in report.sections:
                    if heading == "Captured precision guard call":
                        terminalreporter.write_sep("=", "precision guard (report only)")
                        terminalreporter.write_line(message)

    config.pluginmanager.register(PrecisionGuardPlugin(), "precision-guard-collector")


if __name__ == "__main__":
    sys.exit(main())
