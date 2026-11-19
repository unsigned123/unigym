"""The backend publishes structured ResetOutcome through a public face."""

from __future__ import annotations

import torch

from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, ResetOutcome, register

_TYPE = "test_mujoco_reset_outcome_facts"


class _OutcomeInitializer(InitializerBase):
    name = _TYPE

    def validate(self) -> None:
        self.normalized = {}

    def resolve(self) -> None:
        pass

    def apply(self) -> ResetOutcome:
        return ResetOutcome(valid=False, reason="blocked", metrics={"clearance": 0.0})


def _register() -> None:
    if _TYPE not in INITIALIZER_REGISTRY:
        register(_OutcomeInitializer)


def test_last_reset_outcomes_is_public_and_structured(tiny_env_factory) -> None:
    _register()
    try:
        env = tiny_env_factory(params={"initializer": {"type": _TYPE}})
        outcome = env.last_reset_outcomes[0]
        assert outcome.valid is False
        assert outcome.reason == "blocked"

        env.reset_envs(torch.tensor([0], dtype=torch.int64))
        assert env.last_reset_outcomes[0].valid is False
        assert env.last_reset_outcomes[0].reason == "blocked"
    finally:
        INITIALIZER_REGISTRY.pop(_TYPE, None)
