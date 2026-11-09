"""Command sampler registry (torch-only, unit-testable)."""

from __future__ import annotations

from .base import CommandSampler
from .fixed import FixedSampler
from .lhs import LHSSampler
from .uniform import UniformSampler

COMMAND_REGISTRY = {
    "fixed": FixedSampler,
    "uniform": UniformSampler,
    "lhs": LHSSampler,
}


def build_sampler(section: dict, num_envs: int, device: str) -> CommandSampler:
    params = {k: v for k, v in section.items() if k not in ("name", "dim")}
    return COMMAND_REGISTRY[section["name"]](
        dim=section["dim"], device=device, num_envs=num_envs, **params
    )
