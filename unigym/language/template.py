"""Generic template-based language generator.

Replaces ``{key}`` placeholders in a template string with values drawn from
``states``.  When a state value is a tensor it is converted to a Python scalar
(per-env via indexing).  This class is a convenience base for simple use-cases;
specialised generators subclass it or implement ``describe`` directly.
"""
from __future__ import annotations

from typing import List

import torch

from .base import LanguageGenerator


class TemplateGenerator(LanguageGenerator):
    """Placeholder-driven template engine.

    The template uses ``str.format`` semantics, so ``{vx:.1f}`` etc. work.
    Missing keys are replaced with empty strings to avoid crashes.
    """

    def __init__(self, template: str = "", robot_name: str = "robot"):
        self.template = template
        self.robot_name = robot_name

    def _scalar(self, v, i):
        """Extract a Python scalar from tensor/list/scalar for env *i*."""
        if isinstance(v, torch.Tensor):
            return float(v[i]) if v.dim() > 0 else float(v)
        if isinstance(v, (list, tuple)):
            return v[i]
        return v

    def describe(self, states: dict, command: torch.Tensor) -> List[str]:
        n = command.shape[0]
        results: list[str] = []
        for i in range(n):
            ctx = {}
            for k, v in states.items():
                ctx[k] = self._scalar(v, i)
            ctx.setdefault("robot_name", self.robot_name)
            try:
                results.append(self.template.format(**ctx))
            except (KeyError, ValueError, IndexError):
                results.append(self.template)
        return results
