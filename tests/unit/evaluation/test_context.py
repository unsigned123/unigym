"""Read-only live evaluation context: scope, lifetime and scoped facts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
import torch

from unigym.env.obs import ObsContractError, ObsSnapshot, ObservationSchema, ObservationSpec
from unigym.evaluation import (
    EnvView,
    EvalContext,
    EvalViewError,
    EvalViewStaleError,
    EvalQueryUnavailableError,
)
from unigym.query.facade import Query
from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime, ResolvedControl


class _SampleQuery:
    """A placeholder query handle with the facade member names."""

    def __init__(self) -> None:
        self.backend = "fake"
        self.capabilities = object()
        self.model = object()
        self.state = object()
        self.kinematics = object()
        self.geometry = object()
        self.contacts = object()
        self.actuators = object()
        self.closed = False
        self.required: list[tuple[str, ...]] = []

    def require(self, *ops: str) -> None:
        self.required.append(ops)

    def close(self) -> None:
        self.closed = True


def _record(num_envs: int) -> dict[str, object]:
    return {
        "articulation/g1/joint_pos": torch.arange(num_envs * 2, dtype=torch.float32).reshape(
            num_envs, 2
        ),
        "articulation/g1/base_pos_w": torch.arange(num_envs * 3, dtype=torch.float32).reshape(
            num_envs, 3
        ),
        "time_in_episode": torch.arange(num_envs, dtype=torch.float32),
        "episode_id": torch.arange(num_envs, dtype=torch.int64),
        "body_names": ("torso", "hand"),
    }


def _obs(num_envs: int) -> ObsSnapshot:
    schema = ObservationSchema(
        {"articulation/g1/joint_pos": ObservationSpec("articulation/g1/joint_pos", (2,), "f32")}
    )
    values = {"articulation/g1/joint_pos": torch.arange(num_envs * 2).reshape(num_envs, 2)}
    return ObsSnapshot(
        values,
        schema=schema,
        num_envs=num_envs,
        device="cpu",
        t=4,
        details={"feet": [{"env_id": 0, "body": "a"}]},
    )


def _view(
    *,
    num_envs: int = 3,
    env_id: int | None = None,
    query_factory: Callable[[], Query] | None = None,
    articulations: dict[str, ArticulationRuntime] | None = None,
    controls: dict[tuple[str, str], ControlRuntime] | None = None,
) -> EnvView:
    return EnvView(
        record=_record(num_envs),
        obs=_obs(num_envs),
        env_id=env_id,
        num_envs=num_envs,
        device="cpu",
        backend="fake",
        query_factory=query_factory,
        articulations=articulations,
        controls=controls,
    )


def _control(
    name: str = "legs",
    *,
    articulation: str = "g1",
    role: str = "body",
    joints: tuple[str, ...] = ("j0", "j1"),
) -> ControlRuntime:
    return ControlRuntime(
        name=name,
        adapter="manual",
        articulation=articulation,
        joint_names=joints,
        q0=(0.0,) * len(joints),
        role=role,
    )


def _controls(*rows: ControlRuntime) -> dict[tuple[str, str], ControlRuntime]:
    return {(row.articulation, row.name): row for row in rows}


@pytest.mark.parametrize("num_envs", [1, 3])
def test_vector_record_returns_every_env_row(num_envs: int) -> None:
    """A vector view exposes each env's record rows unchanged."""
    view = _view(num_envs=num_envs)
    record = view.record()

    assert record["articulation/g1/joint_pos"].shape == (num_envs, 2)
    assert record["time_in_episode"].shape == (num_envs,)
    assert record["body_names"] == ("torso", "hand")


def test_per_env_record_keeps_leading_axis_one() -> None:
    """A per-env view slices only that env's rows and keeps ``(1, ...)``."""
    view = _view(num_envs=3, env_id=1)
    record = view.record()
    source = _record(3)

    assert record["articulation/g1/joint_pos"].shape == (1, 2)
    assert record["articulation/g1/base_pos_w"].shape == (1, 3)
    assert record["time_in_episode"].shape == (1,)
    assert record["episode_id"].shape == (1,)
    assert torch.equal(
        record["articulation/g1/joint_pos"], source["articulation/g1/joint_pos"][1:2]
    )
    assert record["body_names"] == ("torso", "hand")


def test_per_env_record_does_not_mutate_the_source() -> None:
    """Scoping reads the boundary record without writing it."""
    source = _record(3)
    before = source["articulation/g1/joint_pos"].clone()
    view = EnvView(
        record=source,
        obs=None,
        env_id=2,
        num_envs=3,
        backend="fake",
    )

    view.record()

    assert torch.equal(source["articulation/g1/joint_pos"], before)


def test_record_mapping_is_read_only() -> None:
    """The returned mapping cannot be mutated by the caller."""
    view = _view()
    record = view.record()

    with pytest.raises(TypeError):
        record["time_in_episode"] = torch.zeros(3)


def test_obs_vector_and_per_env_shapes() -> None:
    """``obs()`` follows the same vector/per-env leading-axis rule."""
    vector = _view(num_envs=3)
    per_env = _view(num_envs=3, env_id=2)

    vector_obs = vector.obs()
    scoped_obs = per_env.obs()

    assert vector_obs is not None and vector_obs.value("articulation/g1/joint_pos").shape == (3, 2)
    assert scoped_obs is not None
    assert scoped_obs.value("articulation/g1/joint_pos").shape == (1, 2)
    assert int(scoped_obs.value("articulation/g1/joint_pos")[0, 0]) == 4
    assert scoped_obs.schema.keys() == vector_obs.schema.keys()
    assert scoped_obs.t == vector_obs.t


def test_obs_select_drops_per_contact_detail() -> None:
    """Detail records have no env axis, so select drops them explicitly."""
    snapshot = _obs(3)

    selected = snapshot.select([2, 0])

    assert selected.value("articulation/g1/joint_pos").shape == (2, 2)
    assert selected.schema is snapshot.schema
    assert selected.contact_detail("feet") is None


def test_obs_select_supports_empty_and_non_contiguous_ids() -> None:
    """Selection order is preserved and an empty selection is legal."""
    snapshot = _obs(3)

    empty = snapshot.select([])
    reordered = snapshot.select(torch.tensor([2, 0], dtype=torch.int64))

    assert empty.value("articulation/g1/joint_pos").shape == (0, 2)
    assert empty.num_envs == 0
    assert reordered.value("articulation/g1/joint_pos")[0, 0].item() == 4
    assert reordered.value("articulation/g1/joint_pos")[1, 0].item() == 0


def test_obs_select_rejects_ids_outside_the_snapshot() -> None:
    """An out-of-range env id is a contract error, not an empty row."""
    snapshot = _obs(3)

    with pytest.raises(ObsContractError, match="outside"):
        snapshot.select([3])


def test_articulation_parses_scoped_record_leaves() -> None:
    """``articulation`` returns the scoped ``<leaf>`` rows as a read-only map."""
    view = _view(num_envs=3, env_id=1)

    art = view.articulation("g1")

    assert sorted(art) == ["base_pos_w", "joint_pos"]
    assert art["joint_pos"].shape == (1, 2)
    with pytest.raises(TypeError):
        art["joint_pos"] = torch.zeros(1, 2)


def test_articulation_missing_name_fails_fast() -> None:
    """An unknown articulation names the available ones."""
    view = _view()

    with pytest.raises(EvalViewError, match=r"missing.*available: g1"):
        view.articulation("missing")


def _g1(limits: tuple[torch.Tensor, torch.Tensor] | None) -> dict[str, ArticulationRuntime]:
    return {
        "g1": ArticulationRuntime(
            name="g1",
            root="fixed",
            joint_names=("j0", "j1"),
            q0=(0.0, 0.0),
            joint_pos_limits=limits,
        )
    }


def test_joint_pos_limits_vector_scope_preserves_order() -> None:
    """Vector limits keep ``(N, J)`` and the lower/upper order."""
    lower = torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    view = _view(num_envs=3, articulations=_g1((lower, upper)))

    got_lower, got_upper = view.joint_pos_limits("g1")

    assert torch.equal(got_lower, lower)
    assert torch.equal(got_upper, upper)


def test_joint_pos_limits_per_env_scope_keeps_one_row() -> None:
    """Per-env limits are scoped to the view's env."""
    lower = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    view = _view(num_envs=2, env_id=1, articulations=_g1((lower, upper)))

    got_lower, got_upper = view.joint_pos_limits("g1")

    assert got_lower.shape == (1, 2)
    assert got_upper.shape == (1, 2)
    assert got_lower.tolist() == [[-3.0, -4.0]]
    assert got_upper.tolist() == [[3.0, 4.0]]


def test_joint_pos_limits_missing_fact_returns_none() -> None:
    """A declared articulation without limits reports ``None``."""
    view = _view(articulations=_g1(None))

    assert view.joint_pos_limits("g1") is None


def test_joint_pos_limits_unknown_articulation_fails_fast() -> None:
    """An articulation with no runtime fact names the available ones."""
    view = _view(articulations=_g1(None))

    with pytest.raises(EvalViewError, match=r"missing.*available: g1"):
        view.joint_pos_limits("missing")


def test_scope_identity_properties() -> None:
    """Vector and per-env views report the same totals with different scope."""
    vector = _view(num_envs=3)
    per_env = _view(num_envs=3, env_id=2)

    assert (vector.num_envs, vector.env_id, vector.scope_size) == (3, None, 3)
    assert (per_env.num_envs, per_env.env_id, per_env.scope_size) == (3, 2, 1)
    assert vector.device == "cpu"


def test_close_is_idempotent_and_every_accessor_is_stale() -> None:
    """Closing invalidates record, obs, identity, limits and query access."""
    view = _view(
        num_envs=2,
        articulations=_g1(None),
        query_factory=lambda: _SampleQuery(),
    )
    view.close()
    view.close()

    for accessor in (
        lambda: view.record(),
        lambda: view.obs(),
        lambda: view.articulation("g1"),
        lambda: view.articulation_facts("g1"),
        lambda: view.joint_pos_limits("g1"),
        lambda: view.num_envs,
        lambda: view.env_id,
        lambda: view.scope_size,
        lambda: view.device,
        lambda: view.controls,
        lambda: view.articulation_names,
        lambda: view.with_record({}),
        lambda: view.for_env(0),
        lambda: view.has_query,
        lambda: view.query,
    ):
        with pytest.raises(EvalViewStaleError):
            accessor()


def test_close_does_not_close_the_run_shared_query() -> None:
    """The view drops its reference but never closes the query handle."""
    fake = _SampleQuery()
    view = _view(query_factory=lambda: fake)

    query = view.query
    view.close()

    assert fake.closed is False
    with pytest.raises(EvalViewStaleError):
        _ = query.backend


def test_missing_query_raises_a_typed_unavailable_error() -> None:
    """No factory means an explicit error naming the backend and the hint."""
    view = _view()

    assert view.has_query is False
    with pytest.raises(EvalQueryUnavailableError, match=r"fake.*record\(\)"):
        _ = view.query


def test_query_is_lazy_and_cached() -> None:
    """Building a view and reading record/obs never constructs the query."""
    calls: list[int] = []

    def factory() -> _SampleQuery:
        calls.append(1)
        return _SampleQuery()

    view = _view(query_factory=factory)

    view.record()
    view.obs()
    assert calls == []
    assert view.has_query is True

    first = view.query
    second = view.query

    assert first is second
    assert calls == [1]


def test_eval_context_is_frozen_and_delegates() -> None:
    """The context carries only ``env``/``t`` and closes through the view."""
    view = _view(num_envs=3, env_id=1)
    ctx = EvalContext(env=view, t=7)

    assert ctx.env_id == 1
    assert ctx.num_envs == 3
    with pytest.raises(FrozenInstanceError):
        ctx.t = 8

    ctx.close()
    with pytest.raises(EvalViewStaleError):
        _ = ctx.env_id


def test_invalid_construction_fails_fast() -> None:
    """Out-of-range env ids and bad counts are rejected at the boundary."""
    with pytest.raises(EvalViewError, match="num_envs"):
        EnvView(record={}, obs=None, env_id=None, num_envs=0)
    with pytest.raises(EvalViewError, match="outside"):
        EnvView(record={}, obs=None, env_id=3, num_envs=3)
    with pytest.raises(EvalViewError, match="num_envs"):
        EnvView(record={}, obs=_obs(2), env_id=None, num_envs=3)


def test_controls_project_to_frozen_resolved_rows() -> None:
    """Runtime controls are exposed only as frozen ``ResolvedControl`` rows."""
    control = _control()
    view = _view(articulations=_g1(None), controls=_controls(control))

    row = view.controls[("g1", "legs")]

    assert isinstance(row, ResolvedControl)
    assert (row.name, row.articulation) == ("legs", "g1")
    assert row.joint_names == ("j0", "j1")
    assert row.role == "body"
    with pytest.raises(TypeError):
        view.controls[("g1", "legs")] = row


def test_controls_key_must_match_control_identity() -> None:
    """A mapping key that is not ``(articulation, name)`` fails at construction."""
    control = _control()

    with pytest.raises(EvalViewError, match=r"controls key.*does not match"):
        _view(articulations=_g1(None), controls={("g1", "arm"): control})


def test_controls_must_be_a_mapping_or_none() -> None:
    """A non-mapping controls payload fails at construction."""
    with pytest.raises(EvalViewError, match="controls must be a mapping"):
        EnvView(
            record={},
            obs=None,
            env_id=None,
            num_envs=1,
            controls=[_control()],  # type: ignore[arg-type]
        )


def test_controls_values_must_be_control_runtimes() -> None:
    """A raw object in the controls mapping fails at construction."""
    with pytest.raises(EvalViewError, match="must be a ControlRuntime"):
        EnvView(
            record={},
            obs=None,
            env_id=None,
            num_envs=1,
            controls={("g1", "legs"): object()},  # type: ignore[dict-item]
        )


def test_articulation_names_are_declared_and_sorted() -> None:
    """The names come from the declared articulation facts, sorted."""
    articulations = _g1(None)
    articulations["arm"] = ArticulationRuntime(
        name="arm", root="fixed", joint_names=("a",), q0=(0.0,)
    )

    view = _view(articulations=articulations)

    assert view.articulation_names == ("arm", "g1")


def test_articulation_facts_selects_the_single_body_control() -> None:
    """``control=None`` uses the articulation's one body-role control."""
    view = _view(articulations=_g1(None), controls=_controls(_control()))

    facts = view.articulation_facts("g1")

    assert facts["joint_names"] == ["j0", "j1"]
    assert facts["asset_joint_names"] == ["j0", "j1"]
    assert facts["action_group"] == "legs"
    with pytest.raises(TypeError):
        facts["action_group"] = "other"


def test_articulation_facts_explicit_control_selects_that_row() -> None:
    """An explicit control name selects its ``(articulation, control)`` row."""
    view = _view(
        articulations=_g1(None),
        controls=_controls(
            _control("legs", role="body"),
            _control("arm", role="aux", joints=("j0",)),
        ),
    )

    facts = view.articulation_facts("g1", "arm")

    assert facts["action_group"] == "arm"
    assert facts["joint_names"] == ["j0"]


def test_articulation_facts_requires_a_body_control_when_implicit() -> None:
    """Zero body-role controls fail instead of guessing an aux row."""
    view = _view(articulations=_g1(None), controls=_controls(_control(role="aux")))

    with pytest.raises(EvalViewError, match=r"no body-role control.*available controls: legs"):
        view.articulation_facts("g1")


def test_articulation_facts_rejects_ambiguous_body_controls() -> None:
    """Multiple body-role controls require an explicit control name."""
    view = _view(
        articulations=_g1(None),
        controls=_controls(
            _control("legs", role="body"),
            _control("torso", role="body", joints=("j0",)),
        ),
    )

    with pytest.raises(EvalViewError, match=r"multiple body-role controls"):
        view.articulation_facts("g1")


def test_articulation_facts_rejects_unknown_control_and_articulation() -> None:
    """Unknown facts names fail fast with the available set."""
    view = _view(articulations=_g1(None), controls=_controls(_control()))

    with pytest.raises(EvalViewError, match=r"no control 'arm'.*available controls: legs"):
        view.articulation_facts("g1", "arm")
    with pytest.raises(EvalViewError, match=r"no runtime facts.*available: g1"):
        view.articulation_facts("missing")


def test_with_record_substitutes_only_the_record() -> None:
    """The child keeps scope/facts/query and shares the record value objects."""
    parent = _view(
        num_envs=3,
        articulations=_g1(None),
        controls=_controls(_control()),
        query_factory=lambda: _SampleQuery(),
    )
    new_record = {
        "articulation/g1/joint_pos": torch.zeros(3, 2),
        "extra": torch.ones(3),
    }

    child = parent.with_record(new_record)

    assert (child.env_id, child.scope_size) == (None, 3)
    assert "extra" in child.record()
    assert "time_in_episode" not in child.record()
    assert child.record()["extra"] is new_record["extra"]
    assert "time_in_episode" in parent.record()
    assert child.controls is parent.controls
    assert child.articulation_names == ("g1",)
    assert child.articulation_facts("g1")["action_group"] == "legs"
    assert child.has_query is True


def test_with_record_preserves_per_env_scope() -> None:
    """A per-env parent keeps its one-row scope over the substituted record."""
    parent = _view(num_envs=3, env_id=1)

    child = parent.with_record({"x": torch.arange(3, dtype=torch.float32)})

    assert (child.env_id, child.scope_size) == (1, 1)
    assert child.record()["x"].tolist() == [1.0]


def test_for_env_scopes_record_obs_and_limits() -> None:
    """A vector child for one env slices record, obs and joint limits."""
    lower = torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    parent = _view(
        num_envs=3,
        articulations=_g1((lower, upper)),
        controls=_controls(_control()),
    )

    child = parent.for_env(1)

    assert (child.env_id, child.scope_size, child.num_envs) == (1, 1, 3)
    assert child.record()["articulation/g1/joint_pos"].shape == (1, 2)
    assert child.record()["articulation/g1/joint_pos"].tolist() == [[2.0, 3.0]]
    obs = child.obs()
    assert obs is not None and obs.value("articulation/g1/joint_pos").shape == (1, 2)
    got_lower, got_upper = child.joint_pos_limits("g1")
    assert got_lower.tolist() == [[-3.0, -4.0]]
    assert got_upper.tolist() == [[3.0, 4.0]]
    assert child.controls is parent.controls
    assert child.articulation_facts("g1", "legs")["action_group"] == "legs"


def test_for_env_rejects_per_env_and_out_of_range_views() -> None:
    """``for_env`` is vector-only and validates the env id."""
    vector = _view(num_envs=3)
    per_env = _view(num_envs=3, env_id=1)

    with pytest.raises(EvalViewError, match="only valid on a vector view"):
        per_env.for_env(0)
    for bad in (-1, 3, True):
        with pytest.raises(EvalViewError):
            vector.for_env(bad)


def test_child_close_is_independent_of_the_parent() -> None:
    """Closing a derived child leaves the parent usable and is idempotent."""
    parent = _view(num_envs=3)
    child = parent.for_env(0)

    child.close()
    child.close()

    with pytest.raises(EvalViewStaleError):
        child.record()
    assert parent.record()["time_in_episode"].shape == (3,)


def test_closing_the_parent_invalidates_every_child() -> None:
    """Parent close reaches children and grandchildren through the chain."""
    parent = _view(num_envs=3)
    child = parent.with_record(dict(_record(3)))
    grandchild = child.for_env(1)

    parent.close()

    for derived in (child, grandchild):
        with pytest.raises(EvalViewStaleError):
            derived.record()


def test_child_query_is_independent_and_never_closed_with_the_view() -> None:
    """Each derived view lazily builds its own query; close drops only its own."""
    raw: list[_SampleQuery] = []

    def factory() -> _SampleQuery:
        handle = _SampleQuery()
        raw.append(handle)
        return handle

    parent = _view(num_envs=3, query_factory=factory)
    child = parent.for_env(1)

    child_query = child.query
    parent_query = parent.query
    assert child_query is not parent_query
    assert raw[0].closed is False and raw[1].closed is False

    child.close()
    assert raw[0].closed is False
    assert parent_query is parent.query
