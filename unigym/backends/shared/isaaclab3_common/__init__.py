"""Shared IsaacLab 3 backend library (P1 skeleton).

This package holds the engine-neutral IsaacLab 3 implementation used by the
``isaaclab3_physx`` and ``isaaclab3_newton`` identities.  It must not import a
concrete backend, the MuJoCo family, or an engine at module import time.  The
engine-specific decisions are injected through
:class:`~unigym.backends.shared.isaaclab3_common.engines.EngineProfile`.
"""

from __future__ import annotations

from .engines import EngineProfile

__all__ = ["EngineProfile"]
