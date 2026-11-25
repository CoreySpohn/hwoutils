"""Contract checks for the precision scanner, CLI, and pytest plugin."""

import subprocess

import pytest

from hwoutils import precision_guard

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (
            "src/example.py",
            """
import os
from jax import config as cfg
from jax import enable_x64 as wide
from hwoutils.precision import enable_x64 as set_x64
flag = cfg.x64_enabled
cfg.update("jax_enable_x64", True)
cfg.read("jax_enable_x64")
cfg.update(name="jax_enable_x64", val=True)
with wide():  # precision-ok: host-only conversion
    pass
with wide():
    pass
set_x64()
flag = os.environ.get("JAX_ENABLE_X64")
""",
            ["x64-flag"] * 7,
        ),
        (
            "src/hwoutils/precision.py",
            """
import jax
jax.config.update('jax_enable_x64', True)
flag = jax.config.x64_enabled
""",
            [],
        ),
        (
            "src/example.py",
            """
import jax.numpy as jn
from jax.numpy import array, complex128 as wide
import numpy as np
a = jn.array([1], dtype=float)
b = array([1], wide)
c = jn.zeros(1, jn.float64)
d = a.astype(complex)
e = np.array([1], dtype=float)
f = jn.array([1], dtype=jn.float32)
g = np.asarray([1]).astype(float)
h = np.array([1]).astype(complex)
i = np.fft.fftfreq(3)
k = (1 - i**2).astype(complex)
j: np.ndarray = np.asarray([1])
j.astype(float)
j = jn.asarray([1])
j.astype(float)
def use_unknown(j):
    return j.astype(float)
jn.asarray([1]).astype(float)
def conditional_cast(x, condition):
    if condition:
        x = np.asarray(x)
        x.astype(complex)
    return x.astype(float)
""",
            ["jnp-dtype"] * 8,
        ),
        (
            "src/example.py",
            """
from jax import numpy as jnp
from numpy import isclose as near
from hwoutils.numerics import isclose
if 1e-5 >= abs(value) > -1e-6:
    result = compare(atol=3e-7, tol=1e-4)
UNMARKED = 1e-7
STOP = 1e-7  # tol: algorithm, 0.1 ppm of the initial residual
a = solve(tol=UNMARKED)
b = solve(tol=STOP)
c = jnp.allclose(a, b, atol=0)
d = near(a, b, 0, 0)
e = near(a, b, within, atol=0)
f = isclose(a, b, ulps=4)
g = near(a, b, 1e-8, 0)
if x > 0.0:
    h = near(a, b, rtol=0, atol=0.0)
i = near(a, b, rtol=0, atol=1e-10)
""",
            ["small-tolerance"] * 4 + ["default-close"] + ["small-tolerance"] * 2,
        ),
        (
            "tests/test_sample.py",
            """
import os
import jax
from hwoutils import enable_x64 as set_x64
with jax.enable_x64():
    pass
class Example:
    jax.config.update("jax_enable_x64", True)
def test_it():
    jax.config.update("jax_enable_x64", False)
def default(value=set_x64()):
    pass
os.environ["JAX_ENABLE_X64"] = "1"
""",
            ["test-x64"] * 4,
        ),
        (
            "tests/conftest.py",
            """
import jax
def pytest_configure(config):
    jax.config.update("jax_enable_x64", True)
def fixture(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
def helper():
    jax.config.update("jax_enable_x64", False)  # precision-ok: subprocess case
""",
            ["test-x64", "test-x64"],
        ),
    ],
)
def test_ast_rules(tmp_path, path, source, expected):
    """Fails if a prohibited form escapes or an allowed form is reported."""
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(source)
    assert [rule for _, rule, _ in precision_guard.scan_file(file)] == expected


def test_cli_modes_and_inline_waiver(tmp_path):
    """Fails if the default blocks, enforce permits, or the waiver is not inline."""
    file = tmp_path / "src" / "sample.py"
    file.parent.mkdir()
    file.write_text(
        "import jax.numpy as jnp\n"
        "a = jnp.ones(1, dtype=float)  # precision-ok: host-only adapter\n"
        "b = jnp.ones(1, dtype=float)  # precision-ok:\n"
        "c = jnp.ones(1, dtype=float)\n"
        'note = "# precision-ok: inside a string"\n'
    )
    hook = precision_guard.__file__
    report = subprocess.run([hook, str(file)], capture_output=True, text=True)
    assert report.returncode == 0, report.stderr
    output = report.stdout
    assert f"{file}:3: jnp-dtype" in output
    assert f"{file}:4: jnp-dtype" in output
    assert f"{file}:2:" not in output
    enforced = subprocess.run(
        [hook, "--enforce", str(file)], capture_output=True, text=True
    )
    assert enforced.returncode == 1
    assert enforced.stdout == report.stdout


def test_pytest_guard_modes(pytester, monkeypatch):
    """Fails if findings disappear or enforcement cannot be activated."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest('pytest_plugins = ["hwoutils.precision_guard"]')
    source = pytester.path / "src" / "some_module.py"
    source.parent.mkdir()
    source.write_text("import jax.numpy as jnp\na = jnp.ones(1, dtype=float)\n")
    subprocess.run(["git", "init", "-q"], cwd=pytester.path, check=True)
    subprocess.run(["git", "add", "."], cwd=pytester.path, check=True)
    result = pytester.runpytest_inprocess()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*some_module.py:2: jnp-dtype*"])
    for flags in [
        ("--precision-guard-enforce",),
        ("-o", "precision_guard_enforce=true"),
    ]:
        result = pytester.runpytest_inprocess(*flags)
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*some_module.py:2: jnp-dtype*"])
