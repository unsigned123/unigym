"""Optional test-environment gates for backend lanes."""

from __future__ import annotations

import os

import pytest

_STRICT_ENV = "UNIGYM_TEST_ENV_STRICT"


def strict_lane_requested() -> bool:
    """Whether a missing optional environment must fail instead of skip."""
    return bool(os.environ.get("CI")) or os.environ.get(_STRICT_ENV) == "1"


def require_optional_module(module_name: str, *, reason: str) -> None:
    """Import an optional lane dependency, skipping locally or failing when strict.

    A local run may skip when an optional extra is not installed.  A strict
    lane (``CI`` set or ``UNIGYM_TEST_ENV_STRICT=1``) treats the missing module
    as a provisioning failure, never a skip.
    """
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        top_level = module_name.split(".", 1)[0]
        if exc.name not in {module_name, top_level}:
            raise
        if strict_lane_requested():
            pytest.fail(f"required test environment is missing {module_name!r}: {reason}")
        pytest.skip(f"{module_name!r} is not installed: {reason}")
