"""
conftest.py — pytest configuration for the FinGraph test suite.

Suppresses deprecation warnings from third-party packages (MLflow, Pydantic v1
shims, OpenTelemetry) that are not actionable and clutter the test output.
Only FinGraph's own DeprecationWarnings are surfaced.
"""

import warnings


def pytest_configure(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"mlflow\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pydantic\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"opentelemetry\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pkg_resources\..*")
