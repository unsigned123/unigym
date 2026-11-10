"""Optional-environment gate: missing modules skip locally and fail when strict."""

from __future__ import annotations

import pytest

from tests._support.envs import require_optional_module, strict_lane_requested

_MISSING_MODULE = "unigym_test_missing_optional_probe_phase2e"


def test_strict_lane_requested_follows_ci_and_explicit_strict(monkeypatch):
    """CI or UNIGYM_TEST_ENV_STRICT=1 marks a lane as required."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("UNIGYM_TEST_ENV_STRICT", raising=False)
    assert strict_lane_requested() is False

    monkeypatch.setenv("CI", "true")
    assert strict_lane_requested() is True

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("UNIGYM_TEST_ENV_STRICT", "1")
    assert strict_lane_requested() is True

    monkeypatch.setenv("UNIGYM_TEST_ENV_STRICT", "0")
    assert strict_lane_requested() is False


def test_missing_optional_module_skips_locally_with_a_teaching_message(monkeypatch):
    """A local optional lane skips with the reason the dependency is required."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("UNIGYM_TEST_ENV_STRICT", raising=False)

    with pytest.raises(pytest.skip.Exception, match="optional extra"):
        require_optional_module(_MISSING_MODULE, reason="the warp lane needs the optional extra")


def test_missing_optional_module_fails_in_a_strict_lane(monkeypatch):
    """A required lane fails on a missing provisioned dependency, never skips."""
    monkeypatch.setenv("CI", "true")

    with pytest.raises(pytest.fail.Exception, match="required test environment"):
        require_optional_module(_MISSING_MODULE, reason="the warp lane needs the optional extra")


def test_present_optional_module_is_accepted():
    """An installed module passes the gate without skip or failure."""
    require_optional_module("pytest", reason="the test runner itself")
