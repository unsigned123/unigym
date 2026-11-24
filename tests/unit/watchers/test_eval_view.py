"""Watchers evaluate the C10 step boundary and keep only a debug snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.unit.watchers._step_fixtures import (
    debug_context,
    live_context,
    obs_snapshot,
    step_context,
    tick_bundle,
)
from unigym.evaluation import EvalViewStaleError
from unigym.runtime.controllers import ArticulationRuntime
from unigym.values import ValueStore
from unigym.watchers import WatchContext, WatchError, WatchResult, WatcherEngine
from unigym.watchers import templates as tpl


class _ContextCaptureWatcher:
    """Callable watcher double that parks its context on the tick bundle."""

    def check(self, ctx):
        ctx.bundle["_test_captured_ctx"] = ctx
        ctx.bundle["_test_captured_view"] = ctx.view
        return WatchResult(fired=ctx.view is not None)


class _ResetWatcher:
    """Callable double recording every on_reset context it receives."""

    def __init__(self) -> None:
        self.contexts: list[WatchContext] = []

    def check(self, ctx):
        del ctx
        return WatchResult(fired=False)

    def on_reset(self, ctx):
        self.contexts.append(ctx)


class _PauseBridge:
    """Bridge double exposing only the watcher pause action."""

    def __init__(self) -> None:
        self.paused = False

    def pause_for_watcher(self) -> None:
        self.paused = True


def _runtime(name: str, limits: tuple[torch.Tensor, torch.Tensor] | None) -> ArticulationRuntime:
    return ArticulationRuntime(
        name=name,
        joint_names=("joint",),
        q0=[0.0],
        joint_pos_limits=limits,
    )


def _arm_limits(lower: float = -1.5, upper: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[lower], [lower]]),
        torch.tensor([[upper], [upper]]),
    )


def _joint_ctx(record: dict, articulations: dict, *, num_envs: int = 2) -> WatchContext:
    return live_context(record, num_envs=num_envs, articulations=articulations)


def _capture_engine() -> WatcherEngine:
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    return engine


def test_on_tick_evaluates_exactly_the_bundle_step_view():
    """One live context reads the runner's step view; no second view exists."""
    engine = _capture_engine()
    engine.add(
        {
            "name": "expr",
            "expr": "WatchResult(fired=(view is bundle['_test_captured_view']))",
        }
    )
    bundle, step = tick_bundle(
        record={"x": torch.tensor([1.0, 2.0])},
        obs={"x": torch.tensor([3.0, 4.0])},
    )

    engine.on_tick(0, bundle)

    ctx = bundle["_test_captured_ctx"]
    assert ctx.step is step
    assert ctx.view is step.view
    assert ctx.view is bundle["_test_captured_view"]
    assert ctx.view.env_id is None
    assert ctx.view.num_envs == 2
    assert ctx.view.has_query is False
    assert torch.equal(ctx.record["x"], torch.tensor([1.0, 2.0]))
    assert torch.equal(ctx.obs["x"], torch.tensor([3.0, 4.0]))
    rows = {row["name"]: row for row in engine.list()}
    assert rows["cap"]["fired"] == 1
    assert rows["expr"]["fired"] == 1


def test_after_tick_the_live_context_becomes_a_debug_snapshot():
    """The runner view stays open, but the retained context is a private copy."""
    engine = _capture_engine()
    bundle, step = tick_bundle(
        record={"x": torch.tensor([1.0, 2.0])},
        obs={"x": torch.tensor([3.0, 4.0])},
    )

    engine.on_tick(0, bundle)

    retained = engine._last_ctx
    assert retained is not None
    assert retained.step is None
    assert retained.debug is not None
    assert retained.view is retained.debug.view
    assert retained.view is not step.view
    assert "step" not in retained.bundle
    assert "view" not in retained.bundle

    step.view.close()
    with pytest.raises(EvalViewStaleError):
        step.view.record()

    assert torch.equal(retained.record["x"], torch.tensor([1.0, 2.0]))
    assert torch.equal(retained.obs["x"], torch.tensor([3.0, 4.0]))
    assert retained.num_envs == 2


def test_add_and_test_between_ticks_use_the_debug_snapshot():
    """Hot add/test work after the runner closed the boundary view."""
    engine = _capture_engine()
    bundle, step = tick_bundle(
        record={"x": torch.tensor([1.0, 2.0])},
        obs={"x": torch.tensor([3.0, 4.0])},
    )
    engine.on_tick(0, bundle)
    step.view.close()

    added = engine.add(
        {
            "name": "probe",
            "expr": "WatchResult(fired=(record['x'][0].item() == 1.0))",
        }
    )
    tested = engine.test(
        {
            "name": "probe2",
            "expr": "WatchResult(fired=(obs['x'][0].item() == 3.0))",
        }
    )

    assert added["dry_run"]["fired"] is True
    assert tested["dry_run"]["fired"] is True


def test_context_exposes_values_and_never_an_env_escape():
    """The expression namespace has values/view but no env attribute."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("sink", "user/value", torch.tensor([5.0, 6.0]))
    engine = _capture_engine()
    bundle, step = tick_bundle(record={"x": torch.tensor([1.0, 2.0])}, values=store)

    engine.on_tick(0, bundle)

    ctx = bundle["_test_captured_ctx"]
    assert not hasattr(ctx, "env")
    namespace = ctx.eval_namespace()
    assert "env" not in namespace
    assert namespace["view"] is step.view
    assert namespace["values"] is ctx.values
    assert namespace["step"] == ctx.t == 0
    assert ctx.values.get("user/value").tolist() == [5.0, 6.0]

    retained = engine._last_ctx
    assert retained.values is retained.debug.values
    assert torch.equal(retained.values["user/value"], torch.tensor([5.0, 6.0]))


def test_offline_context_without_step_or_debug_keeps_the_bundle_fallback():
    """Only an explicitly constructed offline context reads the bundle snapshot."""
    ctx = WatchContext(
        t=0,
        bundle={
            "record": {"x": torch.tensor([1.0, 2.0])},
            "obs": obs_snapshot({"y": torch.tensor([3.0, 4.0])}),
        },
    )

    assert ctx.view is None
    assert len(ctx.values) == 0
    assert torch.equal(ctx.record["x"], torch.tensor([1.0, 2.0]))
    assert torch.equal(ctx.obs["y"], torch.tensor([3.0, 4.0]))
    assert ctx.num_envs == 0


def test_on_reset_with_the_final_step_uses_a_temporary_context():
    """The final step is borrowed for the hook; the debug context is retained."""
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "reset", "callable": f"{Path(__file__).resolve()}:_ResetWatcher"})
    instance = engine._states["reset"].instance
    bundle, _step = tick_bundle(record={"x": torch.tensor([1.0, 2.0])})
    engine.on_tick(0, bundle)
    debug_ctx = engine._last_ctx
    assert debug_ctx is not None
    final_step = step_context(record={"x": torch.tensor([9.0, 9.0])}, t=0)
    final_bundle = {
        "t": 0,
        "reset_facts": {0: {"reset_valid": True}},
        "episode_results": {0: "done"},
        "terminations": {0: "fall"},
    }

    engine.on_reset(torch.tensor([0]), final_step, final_bundle)

    hook_ctx = instance.contexts[-1]
    assert hook_ctx.view is final_step.view
    assert hook_ctx.bundle is final_bundle
    assert hook_ctx.bundle["reset_facts"] == {0: {"reset_valid": True}}
    assert hook_ctx.bundle["episode_results"] == {0: "done"}
    assert hook_ctx.bundle["terminations"] == {0: "fall"}
    assert engine._last_ctx is debug_ctx
    assert engine._last_ctx.view is not final_step.view

    final_step.view.close()
    engine.on_reset(torch.tensor([0]))

    assert instance.contexts[-1] is engine._last_ctx
    assert engine._last_ctx.view is engine._last_ctx.debug.view


def test_debug_snapshot_excludes_camera_observations():
    """Between-tick debug copies carry record/non-camera obs/values only."""
    engine = _capture_engine()
    bundle, _step = tick_bundle(
        record={"x": torch.tensor([1.0, 2.0])},
        obs={
            "x": torch.tensor([3.0, 4.0]),
            "cam/ego/rgb": torch.zeros(2, 2, 2, 3, dtype=torch.uint8),
        },
    )

    engine.on_tick(0, bundle)

    retained = engine._last_ctx
    assert "x" in retained.obs
    assert "cam/ego/rgb" not in retained.obs
    assert "cam/ego/rgb" not in retained.debug.obs
    assert "obs" not in retained.bundle
    assert "raw" not in retained.bundle


def test_joint_limit_accepts_an_explicit_articulation():
    """An explicit articulation reads its own joint_pos and limit facts."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = _joint_ctx(record, {"arm": _runtime("arm", _arm_limits())})

    result = tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})

    assert result.fired is True
    assert result.data["articulation"] == "arm"
    assert result.data["envs"] == [1]


def test_joint_limit_infers_the_articulation_from_the_record_leaf():
    """Without articulation the single joint_pos leaf names the articulation."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = _joint_ctx(record, {"arm": _runtime("arm", _arm_limits())})

    result = tpl.get_template("joint_limit")["check"](ctx, {})

    assert result.fired is True
    assert result.data["articulation"] == "arm"


def test_joint_limit_rejects_an_ambiguous_inference():
    """Two articulation leaves require an explicit articulation."""
    record = {
        "articulation/left/joint_pos": torch.tensor([[0.0], [0.0]]),
        "articulation/right/joint_pos": torch.tensor([[0.0], [0.0]]),
    }
    ctx = _joint_ctx(
        record,
        {
            "left": _runtime("left", _arm_limits()),
            "right": _runtime("right", _arm_limits()),
        },
    )

    with pytest.raises(WatchError, match="ambiguous"):
        tpl.get_template("joint_limit")["check"](ctx, {})


def test_joint_limit_derives_the_articulation_from_a_namespaced_key():
    """A record/articulation/<name>/joint_pos key remains a valid selector."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = _joint_ctx(record, {"arm": _runtime("arm", _arm_limits())})

    result = tpl.get_template("joint_limit")["check"](
        ctx, {"key": "record/articulation/arm/joint_pos"}
    )

    assert result.fired is True
    assert result.data["articulation"] == "arm"


def test_joint_limit_retired_key_reports_the_articulation_migration():
    """The old fixed record/joint_pos key fails fast with a migration hint."""
    errors = tpl.validate_template_params("joint_limit", {"key": "record/joint_pos"}, [])

    assert any("retired" in error and "articulation" in error for error in errors)


def test_joint_limit_rejects_explicit_articulation_conflicting_with_the_key():
    """A namespaced key must select the same articulation it declares."""
    errors = tpl.validate_template_params(
        "joint_limit",
        {"articulation": "arm", "key": "record/articulation/leg/joint_pos"},
        [],
    )

    assert any(
        "selects articulation 'leg'" in error and "declared articulation 'arm'" in error
        for error in errors
    )


def test_joint_limit_without_declared_limits_is_a_data_failure():
    """A declared articulation with no limits fails as a data-shaped error."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = _joint_ctx(record, {"arm": _runtime("arm", None)})

    with pytest.raises(WatchError, match="declares no joint limits"):
        tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})


def test_joint_limit_without_a_view_reports_the_offline_gap():
    """An offline context needs a live step for articulation runtime facts."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = WatchContext(t=0, bundle={"record": record})

    with pytest.raises(WatchError, match="live step context"):
        tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})


def test_joint_limit_debug_context_raises_a_teaching_watch_error():
    """A debug snapshot fails like offline: runtime facts need a live step."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = debug_context(record)

    with pytest.raises(WatchError, match="live step context"):
        tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})


def test_engine_test_surfaces_the_joint_limit_debug_error_as_watch_error():
    """engine.test propagates the template's teaching WatchError unchanged."""
    engine = WatcherEngine(run_id="r", out_dir=None)
    bundle, _step = tick_bundle(
        record={"articulation/arm/joint_pos": torch.tensor([[0.0, 2.0]])},
        num_envs=1,
    )
    engine.on_tick(0, bundle)

    with pytest.raises(WatchError, match="live step context"):
        engine.test(
            {
                "name": "probe",
                "template": "joint_limit",
                "params": {"articulation": "arm"},
            }
        )


def test_firing_actions_still_run_through_the_bridge():
    """The trigger/action layer is unchanged by the debug-snapshot lifecycle."""
    bridge = _PauseBridge()
    engine = WatcherEngine(run_id="r", out_dir=None, bridge=bridge)
    engine.add({"name": "pause", "expr": "WatchResult(fired=True)", "on": ["pause"]})
    bundle, _step = tick_bundle(num_envs=1)

    engine.on_tick(0, bundle)

    assert bridge.paused is True
    assert engine.list()[0]["fired"] == 1


def test_rewind_payload_does_not_checkpoint_the_view_or_debug_snapshot():
    """Views/snapshots are boundary handles, never rewind state."""
    engine = WatcherEngine(run_id="r", out_dir=None)
    bundle, _step = tick_bundle(record={"x": torch.tensor([1.0, 2.0])})
    engine.on_tick(0, bundle)

    state = engine.rewind_state()

    assert set(state) == {"specs", "states", "window_cache"}
