"""Select one JAX precision before test collection.

Load with `pytest_plugins = ["hwoutils.testing.precision"]` in the initial
conftest. Import the package under test from test modules or fixtures so that
its import runs after pytest has configured the lane.
"""

import numpy as np
import pytest
from _pytest.junitxml import xml_key

from hwoutils.precision import enable_x64

_LANES = ("float32", "float64")
_LANE = pytest.StashKey[str]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the precision option and its repository default.

    Args:
        parser: Pytest command-line and ini parser.
    """
    group = parser.getgroup("precision")
    group.addoption(
        "--precision", choices=_LANES, help="JAX floating-point lane for this run."
    )
    parser.addini("default_precision", "Default JAX precision lane.", default="float32")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Configure JAX before test modules can construct any arrays.

    Args:
        config: Pytest run configuration.
    """
    lane = config.getoption("precision") or config.getini("default_precision")
    if lane not in _LANES:
        raise pytest.UsageError("default_precision must be float32 or float64.")
    config.stash[_LANE] = lane
    enable_x64(lane == "float64", use_env=False)  # precision-ok: pytest lane setup
    for lane_name in _LANES:
        config.addinivalue_line(
            "markers", f"{lane_name}_only(reason): collect only in the {lane_name} lane"
        )


def pytest_report_header(config: pytest.Config) -> str:
    """Show the active lane in pytest's startup report.

    Args:
        config: Pytest run configuration.

    Returns:
        The terminal report header line.
    """
    return f"precision: {config.stash[_LANE]}"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Record the suite property even when xdist runs tests in worker processes.

    Args:
        session: Pytest session after the JUnit reporter is configured.
    """
    # record_testsuite_property runs in workers, which do not own the reporter.
    config = session.config
    xml = config.stash.get(xml_key, None)
    if xml is not None:
        xml.add_global_property("precision", config.stash[_LANE])


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect items explicitly restricted to the opposite precision.

    Args:
        config: Pytest run configuration.
        items: Collected test items, updated in place.
    """
    lane = config.stash[_LANE]
    other_marker = "float32_only" if lane == "float64" else "float64_only"
    selected, deselected = [], []
    for item in items:
        if item.get_closest_marker(other_marker) is None:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def precision(pytestconfig: pytest.Config) -> np.dtype:
    """Expose the configured lane as a dtype for test inputs.

    Args:
        pytestconfig: Pytest run configuration.

    Returns:
        The test suite's float32 or float64 dtype.
    """
    return np.dtype(pytestconfig.stash[_LANE])
