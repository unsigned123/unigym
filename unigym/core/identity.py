"""Core-neutral run identity facts.

``RunIdentity`` is a CORE fact object: the complete, JSON-safe description of a
run.  It deliberately knows nothing about tasks, the runtime, or a concrete
backend, so it imports only the standard library and the shared JSON alias.
This module is numpy/torch-free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from unigym._typing import JsonValue

__all__ = [
    "RobotIdentity",
    "RunIdentity",
    "RUN_IDENTITY_SCHEMA_VERSION",
]

RUN_IDENTITY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RobotIdentity:
    """One articulation's stable identity (declared name + profile kind/id)."""

    name: str
    profile: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"robot identity name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError(
                f"robot identity {self.name!r}: profile must be a non-empty "
                f"string, got {self.profile!r}"
            )

    def to_json(self) -> dict[str, JsonValue]:
        """JSON-safe identity document."""
        return {"name": self.name, "profile": self.profile}


@dataclass(frozen=True)
class RunIdentity:
    """Complete run identity facts."""

    backend: str
    device: str
    num_envs: int
    seed: int
    policy_dt: float
    sim_dt: float
    reset_seed: int | None = None
    world_id: str | None = None
    articulations: tuple[RobotIdentity, ...] = ()
    observation_schema_version: int = 1
    model_revision: int = 0
    schema_version: int = RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("backend", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"run identity {name} must be a non-empty string, got {value!r}")
        if not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError(f"num_envs must be an int >= 1, got {self.num_envs!r}")
        if not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        for name in ("policy_dt", "sim_dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"run identity {name} must be finite and > 0, got {value!r}")
            object.__setattr__(self, name, value)
        if self.reset_seed is not None and not isinstance(self.reset_seed, int):
            raise ValueError(f"reset_seed must be an int or None, got {self.reset_seed!r}")
        if self.world_id is not None and not isinstance(self.world_id, str):
            raise ValueError(f"world_id must be a string or None, got {self.world_id!r}")
        arts = tuple(self.articulations)
        if not all(isinstance(a, RobotIdentity) for a in arts):
            raise TypeError("articulations must contain only RobotIdentity entries")
        object.__setattr__(self, "articulations", arts)
        for name in ("observation_schema_version", "model_revision", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"run identity {name} must be an int >= 0, got {value!r}")

    def to_json(self) -> dict[str, JsonValue]:
        """Complete JSON-safe run identity facts."""
        return {
            "schema_version": int(self.schema_version),
            "backend": self.backend,
            "device": self.device,
            "num_envs": int(self.num_envs),
            "seed": int(self.seed),
            "policy_dt": self.policy_dt,
            "sim_dt": self.sim_dt,
            "reset_seed": self.reset_seed,
            "world_id": self.world_id,
            "articulations": [a.to_json() for a in self.articulations],
            "observation_schema_version": int(self.observation_schema_version),
            "model_revision": int(self.model_revision),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RunIdentity:
        """Build from a JSON/meta dict (round-trip helper; JSON only)."""
        arts = value.get("articulations") or []
        if not isinstance(arts, Sequence):
            raise TypeError("articulations must be a sequence")
        return cls(
            backend=str(value["backend"]),
            device=str(value["device"]),
            num_envs=int(value["num_envs"]),
            seed=int(value["seed"]),
            policy_dt=float(value["policy_dt"]),
            sim_dt=float(value["sim_dt"]),
            reset_seed=(None if value.get("reset_seed") is None else int(value["reset_seed"])),
            world_id=(None if value.get("world_id") is None else str(value.get("world_id"))),
            articulations=tuple(
                RobotIdentity(name=str(a["name"]), profile=str(a["profile"]))
                for a in arts
                if isinstance(a, Mapping)
            ),
            observation_schema_version=int(value.get("observation_schema_version", 1)),
            model_revision=int(value.get("model_revision", 0)),
            schema_version=int(value.get("schema_version", RUN_IDENTITY_SCHEMA_VERSION)),
        )
