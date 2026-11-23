"""WatchContext converges on the read-only EnvView evaluation boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.evaluation import EnvView, EvalViewStaleError
from unigym.runtime.controllers import ArticulationRuntime
from unigym.watchers import WatchContext, WatchError, WatchResult, WatcherEngine
from unigym.watchers import templates as tpl


class _Snapshot:
    """Observation snapshot double carrying the published ``values`` face."""

    def __init__(self, values: dict, num_envs: int) -> None:
        self.values = values
        self.num_envs = num_envs


class _EnvBoundary:
    """Minimal live-env input the engine turns into one EnvView."""

    def __init__(
        self,
        *,
        num_envs: int = 2,
        articulations: dict | None = None,
        query=None,
    ) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.articulations = articulations or {}
        self._backend_label = "fake backend"
        if query is not None:
            self.query = query


class _ContextCaptureWatcher:
    """Callable watcher double that parks its context on the tick bundle."""

    def check(self, ctx):
        ctx.bundle["_test_captured_ctx"] = ctx
        ctx.bundle["_test_captured_view"] = ctx.view
        return WatchResult(fired=ctx.view is not None)


class _QuerySentinel:
    """Stand-in run-shared query that records an unwanted close."""

    def __init__(self) -> None:
        self.closed = False
        self.model = object()
        self.state = object()
        self.kinematics = object()
        self.geometry = object()
        self.contacts = object()
        self.actuators = object()

    def close(self) -> None:
        self.closed = True


class _PauseBridge:
    """Bridge double exposing only the watcher pause action."""

    def __init__(self) -> None:
        self.paused = False

    def pause_for_watcher(self) -> None:
        self.paused = True


def _bundle(
    record: dict,
    *,
    num_envs: int,
    env=None,
    obs: dict | None = None,
) -> dict:
    bundle: dict = {
        "record": record,
        "obs": _Snapshot(obs if obs is not None else {}, num_envs),
    }
    if env is not None:
        bundle["env"] = env
    return bundle


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
    view = EnvView(
        record=record,
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
        backend="fake backend",
        articulations=articulations,
    )
    return WatchContext(t=0, bundle={"record": record}, view=view)


def test_context_reads_record_obs_and_num_envs_through_the_view():
    """The view, not the bundle snapshot, is the live read path."""
    view = EnvView(
        record={"x": torch.tensor([7.0, 8.0])},
        obs=_Snapshot({"x": torch.tensor([3.0, 4.0])}, 2),
        env_id=None,
        num_envs=2,
        device="cpu",
        backend="fake backend",
    )
    bundle = _bundle(
        {"x": torch.tensor([0.0, 0.0])},
        num_envs=2,
        obs={"x": torch.tensor([0.0, 0.0])},
    )

    ctx = WatchContext(t=0, bundle=bundle, view=view)

    assert torch.equal(ctx.record["x"], torch.tensor([7.0, 8.0]))
    assert torch.equal(ctx.obs["x"], torch.tensor([3.0, 4.0]))
    assert ctx.num_envs == 2


def test_expression_namespace_exposes_view_and_not_env():
    """The expr face carries the read-only view and no raw env handle."""
    view = EnvView(
        record={},
        obs=None,
        env_id=None,
        num_envs=1,
        device="cpu",
        backend="fake backend",
    )
    ctx = WatchContext(t=0, bundle={}, view=view)

    namespace = ctx.eval_namespace()

    assert namespace["view"] is view
    assert "env" not in namespace


def test_engine_builds_one_view_for_callable_and_expression_predicates():
    """One vector-scoped view per tick, shared by callable and expression."""
    env = _EnvBoundary(num_envs=2)
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    engine.add(
        {
            "name": "expr",
            "expr": "WatchResult(fired=(view is bundle['_test_captured_view']))",
        }
    )
    bundle = _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2, env=env)

    engine.on_tick(0, bundle)

    ctx = bundle["_test_captured_ctx"]
    assert ctx.view is bundle["_test_captured_view"]
    assert ctx.view.env_id is None
    assert ctx.view.num_envs == 2
    assert ctx.view.has_query is False
    rows = {row["name"]: row for row in engine.list()}
    assert rows["cap"]["fired"] == 1
    assert rows["expr"]["fired"] == 1


def test_bundle_without_env_falls_back_to_the_engine_env():
    """A step bundle with no env still gets a view from the wired engine env."""
    env = _EnvBoundary(num_envs=2)
    engine = WatcherEngine(run_id="r", out_dir=None, env=env)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    bundle = _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2)

    engine.on_tick(0, bundle)

    ctx = bundle["_test_captured_ctx"]
    assert ctx.view is not None
    assert ctx.view.num_envs == 2
    assert ctx.num_envs == 2


def test_previous_tick_view_is_closed_when_the_next_tick_replaces_it():
    """A retained context reads live state only while its tick is current."""
    env = _EnvBoundary(num_envs=2)
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    first_bundle = _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2, env=env)

    engine.on_tick(0, first_bundle)
    first = first_bundle["_test_captured_ctx"]
    assert torch.equal(first.record["x"], torch.tensor([1.0, 2.0]))

    engine.on_tick(1, _bundle({"x": torch.tensor([3.0, 4.0])}, num_envs=2, env=env))

    with pytest.raises(EvalViewStaleError):
        _ = first.record


def test_add_and_test_between_ticks_use_the_current_open_view():
    """Hot add/test dry-runs read the retained view without closing it."""
    env = _EnvBoundary(num_envs=2)
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    bundle = _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2, env=env)
    engine.on_tick(0, bundle)
    current = bundle["_test_captured_ctx"]

    added = engine.add({"name": "expr", "expr": "WatchResult(fired=(view is not None))"})
    tested = engine.test({"name": "probe", "expr": "WatchResult(fired=(view is not None))"})

    assert added["dry_run"]["fired"] is True
    assert tested["dry_run"]["fired"] is True
    assert torch.equal(current.record["x"], torch.tensor([1.0, 2.0]))


def test_closing_a_view_does_not_close_the_run_shared_query():
    """View invalidation never closes the env-owned query handle."""
    query = _QuerySentinel()
    env = _EnvBoundary(num_envs=2, query=query)
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add({"name": "cap", "callable": f"{Path(__file__).resolve()}:_ContextCaptureWatcher"})
    first_bundle = _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2, env=env)

    engine.on_tick(0, first_bundle)
    first = first_bundle["_test_captured_ctx"]
    assert first.view.has_query is True
    assert first.view.query is not None

    engine.on_tick(1, _bundle({"x": torch.tensor([3.0, 4.0])}, num_envs=2, env=env))

    with pytest.raises(EvalViewStaleError):
        _ = first.view.query
    assert query.closed is False


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
    """An offline context has no articulation limits to read."""
    record = {"articulation/arm/joint_pos": torch.tensor([[0.0], [2.0]])}
    ctx = WatchContext(t=0, bundle={"record": record})

    with pytest.raises(WatchError, match="no live evaluation view"):
        tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})


def test_firing_actions_still_run_through_the_bridge():
    """The trigger/action layer is unchanged by the view convergence."""
    bridge = _PauseBridge()
    env = _EnvBoundary(num_envs=1)
    engine = WatcherEngine(run_id="r", out_dir=None, bridge=bridge)
    engine.add({"name": "pause", "expr": "WatchResult(fired=True)", "on": ["pause"]})

    engine.on_tick(0, _bundle({}, num_envs=1, env=env))

    assert bridge.paused is True
    assert engine.list()[0]["fired"] == 1


def test_rewind_payload_does_not_checkpoint_the_view():
    """Views are boundary handles, never rewind state."""
    env = _EnvBoundary(num_envs=2)
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.on_tick(0, _bundle({"x": torch.tensor([1.0, 2.0])}, num_envs=2, env=env))

    state = engine.rewind_state()

    assert set(state) == {"specs", "states", "window_cache"}
