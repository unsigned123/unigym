"""Shared in-process ``isaaclab`` sensors double for contact-ref tests."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Cfg:
    """Minimal stand-in for an IsaacLab config dataclass."""

    def __init__(self, **kwargs: Any) -> None:
        self.kw: dict[str, Any] = kwargs


class _InitialStateCfg(_Cfg):
    """Config namespace nested under ``RigidObjectCfg``."""


class _RigidObjectCfg(_Cfg):
    """Rigid-object config exposing the nested initial-state class."""

    InitialStateCfg = _InitialStateCfg


class _SensorCfg(_Cfg):
    """Contact-sensor config stand-in."""


@pytest.fixture
def fake_isaaclab(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install ``isaaclab`` and ``isaaclab.sensors``, returning the sensors module."""
    package = types.ModuleType("isaaclab")
    sensors = types.ModuleType("isaaclab.sensors")
    setattr(sensors, "ContactSensorCfg", _SensorCfg)
    setattr(package, "sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    return sensors
