"""Select one JAX precision before initial conftests import.

Pytest loads this plugin through its installed entry point. Use
``-p hwoutils.testing.precision`` when automatic discovery is unavailable.
"""

from importlib.util import find_spec

import numpy as np
import pytest
from _pytest.junitxml import xml_key

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


def _configure_lane(config: pytest.Config, option: str | None) -> None:
    lane = option or config.getini("default_precision")
    if lane not in _LANES:
        raise pytest.UsageError("default_precision must be float32 or float64.")
    if find_spec("jax") is not None:
        from hwoutils.precision import enable_x64

        enable_x64(lane == "float64", use_env=False)  # precision-ok: pytest lane setup
    elif lane == "float64":
        raise pytest.UsageError("The float64 lane requires JAX. Install hwoutils[jax].")
    config.stash[_LANE] = lane


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config: pytest.Config) -> None:
    """Select the lane before conftests can construct any arrays.

    Args:
        early_config: Pytest configuration after the preliminary option parse.
    """
    _configure_lane(early_config, early_config.known_args_namespace.precision)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Register the lane markers and support conftest-based plugin loading.

    Args:
        config: Pytest run configuration.
    """
    if _LANE not in config.stash:
        _configure_lane(config, config.getoption("precision"))
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
