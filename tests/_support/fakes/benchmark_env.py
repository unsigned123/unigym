"""Typed benchmark-sink doubles for the live per-env evaluation boundary.

``FakeBenchmarkEnv`` exposes the run-layer facts ``BenchmarkSink`` reads when
it builds a per-env ``EvalContext``: ``num_envs``, ``device``, the backend
label, the articulation map and a lazy runtime-query property.  The query
subfaces are intentionally minimal: only ``state.root_pose`` is meaningful
(it records the env selection so a test can prove per-env scoping).  The
query handle still exposes the full protocol shape at runtime so the sink's
``isinstance(..., Query)`` availability check succeeds.

Limitations: no physics, no state writes, no scratch world, no contact or
camera data.  ``scratch()`` raises instead of inventing a world.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch

from unigym.query.base import EnvSelection, Pose
from unigym.runtime.controllers import ArticulationRuntime
from unigym.query.facade import ScratchWorld

__all__ = ["FakeBenchmarkEnv", "FakeBenchmarkQuery", "FakeQueryState"]


class FakeQueryState:
    """Records the env selection of each ``root_pose`` read."""

    def __init__(self) -> None:
        self.envs: list[EnvSelection] = []

    def root_pose(self, articulation: str, *, env: EnvSelection = None) -> Pose:
        """Return a one-row pose and record the requested env selection."""
        del articulation
        self.envs.append(env)
        return Pose(
            env_ids=torch.tensor([0], dtype=torch.int64),
            pos_w=torch.zeros(1, 3),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )


class FakeBenchmarkQuery:
    """A read-only query facade with the subfaces ``EvalQuery`` wraps."""

    def __init__(self) -> None:
        self.backend = "fake"
        self.capabilities: object = object()
        self.model: object = object()
        self.state = FakeQueryState()
        self.kinematics: object = object()
        self.geometry: object = object()
        self.contacts: object = object()
        self.actuators: object = object()

    def require(self, *ops: str) -> None:
        """Accept every capability request (the fake advertises nothing)."""
        del ops

    def scratch(self, env: EnvSelection = None) -> AbstractContextManager[ScratchWorld]:
        """Refuse a writable scratch world explicitly."""
        del env
        raise NotImplementedError("the benchmark eval fake exposes no scratch world")

    def close(self) -> None:
        """No-op: the sink never owns the fake query handle."""
        return None


class FakeBenchmarkEnv:
    """Env double carrying the run facts and a lazy query property."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.articulations: dict[str, ArticulationRuntime] = {}
        self._backend_label = "fake backend"
        self._query = FakeBenchmarkQuery()
        self.query_reads = 0

    @property
    def query(self) -> FakeBenchmarkQuery:
        """Return the query handle and count the access."""
        self.query_reads += 1
        return self._query
