"""Runner layer: SyncRunner + external/gym drivers + sinks (consumers).

The runner owns the rollout control flow; sinks own everything
consumer-side (collection recording today, benchmark scoring in B1+).
See docs/contracts/runner_contract.md.
"""
from __future__ import annotations

from .base import Runner, StepResult
from .external import ExternalRunner
from .gym import GymEnvRunner, GymRunner, GymVectorRunner
from .sinks import CollectSink
from .sync import SyncRunner

_RUNNER_KINDS = {
    "sync": SyncRunner,
    "external": ExternalRunner,
    "gym": GymEnvRunner,
    "gym_vector": GymVectorRunner,
}


def make_runner(instance, kind: str = "sync", **kwargs) -> Runner:
    """Build a :class:`Runner` for one live :class:`RunInstance` (v097).

    ``kind`` selects the driver: ``sync`` (adapter-driven rollout),
    ``external`` (user-driven ``step``, ``action_fn`` optional), ``gym``
    (single-env Gymnasium ``Env``: ``reset``/``step``; ``reward_fn`` /
    ``action_fn`` / ``max_steps`` forwarded) or ``gym_vector`` (batched
    Gymnasium ``VectorEnv`` with SameStep autoreset; ``reward_fn`` /
    ``action_fn`` forwarded).  Unknown kinds fail fast and name the known
    set.
    """
    try:
        factory = _RUNNER_KINDS[kind]
    except KeyError:
        known = ", ".join(sorted(_RUNNER_KINDS))
        raise ValueError(
            f"unknown runner kind {kind!r} (known: {known})") from None
    if kind in ("external", "gym", "gym_vector"):
        return factory.from_instance(instance, **kwargs)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(
            f"runner kind {kind!r} takes no options (got: {unknown})")
    return factory.from_instance(instance)


__all__ = [
    "CollectSink", "ExternalRunner", "GymEnvRunner", "GymRunner",
    "GymVectorRunner", "Runner", "StepResult", "SyncRunner", "make_runner",
]
