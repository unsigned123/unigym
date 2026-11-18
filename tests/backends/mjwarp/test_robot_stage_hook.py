"""Mjwarp robot-stage hook: once after apply, shared RNG, warm start kept."""

from __future__ import annotations

import numpy as np
import torch

from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, register

_CALLS: list[tuple[str, float]] = []
_HOOK_TYPE = "test_mjwarp_robot_stage_hook"
_PLAIN_TYPE = "test_mjwarp_robot_stage_plain"
_WARM_TYPE = "test_mjwarp_warmstart"
_MULTI_TYPE = "test_mjwarp_multi_warmstart"


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


class _WarmStartInitializer(InitializerBase):
    """Seed the post-settle warm start, then issue a stage joint write."""

    name = _WARM_TYPE
    capture_env = None
    seeded = None
    mirror_after = None
    device_after = None

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    def apply(self) -> dict:
        return {}

    def apply_robot_stages(self) -> None:
        self.primitives.settle_fn(2)
        data = self.capture_env.mj_datas[0]
        self.seeded = np.asarray(data.qacc_warmstart).copy()
        self.primitives.set_joint_state("tiny", ("elbow",), [0.2])
        self.mirror_after = np.asarray(data.qacc_warmstart).copy()
        self.device_after = np.asarray(self.capture_env._wdata.qacc_warmstart.numpy()[0]).copy()


class _MultiEnvWarmStartInitializer(InitializerBase):
    """Probe env 0 (active reset) vs env 1 in one N=2 joint-write call."""

    name = _MULTI_TYPE
    capture_env = None

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    def apply(self) -> dict:
        return {}

    def apply_robot_stages(self) -> None:
        env = self.capture_env
        d0, d1 = env.mj_datas[0], env.mj_datas[1]
        d0.qacc_warmstart[:] = 1.25
        d1.qacc_warmstart[:] = 2.5
        self.env0_seed = np.asarray(d0.qacc_warmstart).copy()
        self.env1_seed = np.asarray(d1.qacc_warmstart).copy()
        self.primitives.set_joint_state("tiny", ("elbow",), [0.2])
        env.set_joint_state(
            [1],
            articulation="tiny",
            joint_names=("elbow",),
            position=torch.tensor([[0.3]]),
        )
        self.env0_after = np.asarray(d0.qacc_warmstart).copy()
        self.env1_after = np.asarray(d1.qacc_warmstart).copy()
        self.device0_after = np.asarray(env._wdata.qacc_warmstart.numpy()[0]).copy()
        self.device1_after = np.asarray(env._wdata.qacc_warmstart.numpy()[1]).copy()


def _register(types) -> None:
    for cls in types:
        if cls.name not in INITIALIZER_REGISTRY:
            register(cls)


def _cleanup() -> None:
    for name in (_HOOK_TYPE, _PLAIN_TYPE, _WARM_TYPE, _MULTI_TYPE):
        INITIALIZER_REGISTRY.pop(name, None)


def test_stage_hook_runs_once_after_each_apply(tiny_env_factory) -> None:
    _register((_HookInitializer, _PlainInitializer, _WarmStartInitializer))
    try:
        _CALLS.clear()
        env = tiny_env_factory(params={"initializer": {"type": _HOOK_TYPE, "params": {"seed": 5}}})
        consumed = len(_CALLS)
        _CALLS.clear()

        env.reset_envs(torch.tensor([0], dtype=torch.int64))
        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert [label for label, _ in _CALLS] == ["apply", "stages", "apply", "stages"]
        expected = np.random.default_rng(5).uniform(size=consumed + 4).tolist()
        assert [value for _, value in _CALLS] == expected[consumed:]
    finally:
        _cleanup()


def test_initializer_without_stages_is_a_noop(tiny_env_factory) -> None:
    _register((_HookInitializer, _PlainInitializer, _WarmStartInitializer))
    try:
        _CALLS.clear()
        env = tiny_env_factory(params={"initializer": {"type": _PLAIN_TYPE}})
        _CALLS.clear()

        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert [label for label, _ in _CALLS] == ["apply"]
    finally:
        _cleanup()


def test_apply_robot_stages_preserves_the_reset_warm_start(tiny_env_factory) -> None:
    _register((_HookInitializer, _PlainInitializer, _WarmStartInitializer))
    try:
        env = tiny_env_factory(params={"initializer": {"type": _PLAIN_TYPE}})
        initializer = _WarmStartInitializer(
            {}, env._initializer.profile, env._initializer.primitives
        )
        initializer.capture_env = env
        env._initializer = initializer

        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert float(np.linalg.norm(initializer.seeded)) > 0.0
        assert initializer.mirror_after.tolist() == initializer.seeded.tolist()
        assert np.allclose(initializer.device_after, initializer.seeded, atol=1e-6)
        assert env.mj_datas[0].qacc_warmstart.tolist() == initializer.seeded.tolist()
        device = np.asarray(env._wdata.qacc_warmstart.numpy()[0], dtype=np.float64)
        assert np.allclose(device, initializer.seeded, atol=1e-6)
    finally:
        _cleanup()


def test_n2_only_the_active_reset_env_preserves_warm_start(tiny_env_factory) -> None:
    """Active reset env 0 keeps its seed; env 1 in the same call is cleared."""
    _register(
        (
            _HookInitializer,
            _PlainInitializer,
            _WarmStartInitializer,
            _MultiEnvWarmStartInitializer,
        )
    )
    try:
        env = tiny_env_factory(num_envs=2, params={"initializer": {"type": _PLAIN_TYPE}})
        initializer = _MultiEnvWarmStartInitializer(
            {}, env._initializer.profile, env._initializer.primitives
        )
        initializer.capture_env = env
        env._initializer = initializer

        env.reset_envs(torch.tensor([0], dtype=torch.int64))

        assert float(np.linalg.norm(initializer.env0_seed)) > 0.0
        assert float(np.linalg.norm(initializer.env1_seed)) > 0.0
        assert initializer.env0_after.tolist() == initializer.env0_seed.tolist()
        assert not initializer.env1_after.any()
        assert np.allclose(initializer.device0_after, initializer.env0_seed, atol=1e-6)
        assert not initializer.device1_after.any()

        assert env.mj_datas[0].qacc_warmstart.tolist() == initializer.env0_seed.tolist()
        assert not env.mj_datas[1].qacc_warmstart.any()
        device = np.asarray(env._wdata.qacc_warmstart.numpy(), dtype=np.float64)
        assert np.allclose(device[0], initializer.env0_seed, atol=1e-6)
        assert not device[1].any()
    finally:
        _cleanup()
