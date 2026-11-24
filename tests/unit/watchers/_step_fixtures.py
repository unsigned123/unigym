"""Shared real ``EnvStepContext`` fixtures for watcher unit tests."""

from __future__ import annotations

import torch

from unigym.env.obs import ObsSnapshot, ObservationSchema
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues, ValueStore
from unigym.watchers import WatchContext, WatchDebugSnapshot


def obs_snapshot(values: dict | None = None, *, num_envs: int = 2, t: int = 0) -> ObsSnapshot:
    """Real observation snapshot (empty schema is enough for watcher reads)."""
    return ObsSnapshot(
        dict(values or {}),
        schema=ObservationSchema({}),
        num_envs=num_envs,
        device="cpu",
        t=t,
    )


def step_context(
    record: dict | None = None,
    obs: dict | None = None,
    *,
    values: ValueStore | StepValues | None = None,
    num_envs: int = 2,
    t: int = 0,
    articulations: dict | None = None,
) -> EnvStepContext:
    """Build a real core step boundary over a plain test view."""
    rec = dict(record or {})
    obs_values = dict(obs or {})
    view = EnvView(
        record=rec,
        obs=obs_snapshot(obs_values, num_envs=num_envs, t=t),
        env_id=None,
        num_envs=num_envs,
        device="cpu",
        backend="test backend",
        articulations=articulations,
    )
    if values is None:
        value_view = StepValues({})
    elif isinstance(values, ValueStore):
        value_view = values.snapshot()
    else:
        value_view = values
    return EnvStepContext(
        view=view,
        values=value_view,
        t=t,
        step_index=torch.zeros(num_envs, dtype=torch.int64),
        time_s=None,
        cmd=None,
        reset_state={},
    )


def tick_bundle(
    record: dict | None = None,
    obs: dict | None = None,
    *,
    values: ValueStore | StepValues | None = None,
    num_envs: int = 2,
    t: int = 0,
    articulations: dict | None = None,
    extra: dict | None = None,
) -> tuple[dict, EnvStepContext]:
    """The runner-shaped bundle plus its live step for one evaluated tick."""
    step = step_context(
        record,
        obs,
        values=values,
        num_envs=num_envs,
        t=t,
        articulations=articulations,
    )
    bundle: dict = {
        "record": dict(record or {}),
        "obs": step.view.obs(),
        "step": step,
    }
    bundle.update(extra or {})
    return bundle, step


def live_context(
    record: dict | None = None,
    obs: dict | None = None,
    *,
    values: ValueStore | StepValues | None = None,
    num_envs: int = 2,
    t: int = 0,
    articulations: dict | None = None,
) -> WatchContext:
    """A live ``step=`` context (keeps the step view open for view facts)."""
    step = step_context(
        record,
        obs,
        values=values,
        num_envs=num_envs,
        t=t,
        articulations=articulations,
    )
    return WatchContext(
        t=t,
        bundle={"record": dict(record or {}), "obs": step.view.obs()},
        step=step,
    )


def debug_context(
    record: dict | None = None,
    obs: dict | None = None,
    *,
    values: ValueStore | StepValues | None = None,
    num_envs: int = 2,
    t: int = 0,
) -> WatchContext:
    """Offline debug context: real step -> immutable snapshot -> closed view.

    The debug view is rebuilt from public record/observation/identity only; it
    carries no articulation/control runtime facts (use :func:`live_context`).
    """
    step = step_context(record, obs, values=values, num_envs=num_envs, t=t)
    snapshot = WatchDebugSnapshot.from_step(step)
    step.view.close()
    return WatchContext(t=t, bundle={"record": dict(record or {})}, debug=snapshot)
