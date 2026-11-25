"""Numerical test helpers and the pytest precision plugin."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hwoutils.testing._numerical import (
        assert_precision_closed,
        precision_pair,
        ulps,
    )

__all__ = ["assert_precision_closed", "precision_pair", "ulps"]


def __getattr__(name):
    """Load numerical helpers on demand so the plugin works without Equinox.

    Args:
        name: Attribute name looked up on the package.

    Returns:
        The requested numerical test helper.

    Raises:
        AttributeError: If the name is not a public numerical helper.
    """
    if name in __all__:
        helper = getattr(import_module("hwoutils.testing._numerical"), name)
        globals()[name] = helper
        return helper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
