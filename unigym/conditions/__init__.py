"""Canonical evaluation conditions.

This package's base import is torch-free: it exports only the structural
:class:`Condition` protocol and :class:`ConditionError`.  Concrete conditions
live in :mod:`unigym.conditions.height`, :mod:`unigym.conditions.tilt` and
:mod:`unigym.conditions.joint_limit`; those modules may import torch, so domain
packages import them locally at the adapter boundary.
"""

from __future__ import annotations

from .base import Condition, ConditionError

__all__ = ["Condition", "ConditionError"]
