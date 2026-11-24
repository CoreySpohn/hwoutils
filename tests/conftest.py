"""Shared fixtures for hwoutils tests."""

import importlib.util
import os

from hypothesis import settings

# Released hwoutils versions may not include the optional plugins yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
if importlib.util.find_spec("hwoutils.leaves_contract") is not None:
    pytest_plugins.append("hwoutils.leaves_contract")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))
