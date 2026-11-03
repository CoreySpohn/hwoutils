"""Test budget guard: per-test time, memory and compile budgets for pytest suites.

Load the plugin with ``pytest_plugins = ["hwoutils.testbudget.plugin"]`` in the
top-level ``conftest.py`` and configure it under ``[tool.testbudget]`` in the
library's ``pyproject.toml``.
"""

from hwoutils.testbudget.config import BudgetConfig, load_config

__all__ = ["BudgetConfig", "load_config"]
