"""pytest configuration for the rapidfem Python test suite.

Marker convention:
- `slow`: runs a full transient or sweep. Default opt-out via
  `pytest -m "not slow"`; explicit opt-in via `pytest -m slow`.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that run a full TD transient or FD sweep (minutes)",
    )
