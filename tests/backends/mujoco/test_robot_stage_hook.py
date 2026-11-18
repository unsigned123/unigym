"""The generic robot-stage hook runs once after placement apply."""

from __future__ import annotations

import numpy as np
import torch

from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, register

_CALLS: list[tuple[str, float]] = []
_HOOK_TYPE = "test_robot_stage_hook"
_PLAIN_TYPE = "test_robot_stage_plain"


class _HookInitializer(InitializerBase):
    """An initializer type that declares a robot-stage hook."""

    name = _HOOK_TYPE

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    def apply(self) -> dict:
        _CALLS.append(("apply", float(self.rng.uniform())))
        return {}

    def apply_robot_stages(self) -> None:
        _CALLS.append(("stages", float(self.rng.uniform())))


class _PlainInitializer(InitializerBase):
    """An initializer type that keeps the base no-op stage hook."""

    name = _PLAIN_TYPE

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    def apply(self) -> dict:
        _CALLS.append(("apply", -1.0))
        return {}


def _register(types) -> None:
    for cls in types:
        if cls.name not in INITIALIZER_REGISTRY:
            register(cls)


def test_stage_hook_runs_once_after_each_apply(tiny_env_factory) -> None:
    _register((_HookInitializer, _PlainInitializer))
    try:
        _CALLS.clear()
        env = tiny_env_factory(params={"initializer": {"type": _HOOK_TYPE, "params": {"seed": 5}}})
        # construction runs one reset; measure from the constructed state
        consumed = len(_CALLS)
        _CALLS.clear()

        env.reset_envs(torch.tensor([0], dtype=torch.int64))
        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert [label for label, _ in _CALLS] == ["apply", "stages", "apply", "stages"]
        expected = np.random.default_rng(5).uniform(size=consumed + 4).tolist()
        assert [value for _, value in _CALLS] == expected[consumed:]
    finally:
        INITIALIZER_REGISTRY.pop(_HOOK_TYPE, None)
        INITIALIZER_REGISTRY.pop(_PLAIN_TYPE, None)


def test_initializer_without_stages_is_a_noop(tiny_env_factory) -> None:
    _register((_HookInitializer, _PlainInitializer))
    try:
        _CALLS.clear()
        env = tiny_env_factory(params={"initializer": {"type": _PLAIN_TYPE}})
        _CALLS.clear()

        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert [label for label, _ in _CALLS] == ["apply"]
    finally:
        INITIALIZER_REGISTRY.pop(_HOOK_TYPE, None)
        INITIALIZER_REGISTRY.pop(_PLAIN_TYPE, None)
