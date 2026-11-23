"""EvalQuery: read-only subset and per-env scope enforcement."""

from __future__ import annotations

import inspect

import pytest
import torch

from unigym.evaluation import EnvView, EvalViewError, EvalViewStaleError
from unigym.query.base import ContactPoint


class _StateRecorder:
    def __init__(self) -> None:
        self.body_pose_calls: list[object] = []
        self.root_pose_calls: list[object] = []
        self.joint_position_calls: list[object] = []

    def body_pose(self, name: str, *, articulation=None, env=None):
        self.body_pose_calls.append((name, articulation, env))
        return None

    def root_pose(self, articulation: str, *, env=None):
        self.root_pose_calls.append((articulation, env))
        return None

    def joint_position(self, articulation: str, *, joint_names=None, env=None):
        self.joint_position_calls.append((articulation, joint_names, env))
        return None


class _ContactsRecorder:
    def __init__(self) -> None:
        self.manifold_calls: list[object] = []
        self.scan_calls: list[object] = []
        self.point_calls: list[object] = []

    def manifold(self, group: str, *, env=None):
        self.manifold_calls.append((group, env))
        return None

    def scan(self, *, env=None):
        self.scan_calls.append(env)
        return None

    def jacobian(self, point: ContactPoint, articulation: str):
        self.point_calls.append((point, articulation))
        return None

    def wrench_to_joint(self, point: ContactPoint, articulation: str):
        self.point_calls.append((point, articulation))
        return None


class _ActuatorsRecorder:
    def __init__(self) -> None:
        self.state_calls: list[object] = []

    def state(self, name: str, *, articulation=None, env=None):
        self.state_calls.append((name, articulation, env))
        return None


class _SampleQuery:
    def __init__(self) -> None:
        self.backend = "fake"
        self.capabilities = object()
        self.model = object()
        self.state = _StateRecorder()
        self.kinematics = object()
        self.geometry = object()
        self.contacts = _ContactsRecorder()
        self.actuators = _ActuatorsRecorder()
        self.required: list[tuple[str, ...]] = []

    def require(self, *ops: str) -> None:
        self.required.append(ops)


def _view(env_id: int | None, num_envs: int = 3) -> tuple[EnvView, _SampleQuery]:
    fake = _SampleQuery()
    view = EnvView(
        record={"time_in_episode": torch.zeros(num_envs)},
        obs=None,
        env_id=env_id,
        num_envs=num_envs,
        backend="fake",
        query_factory=lambda: fake,
    )
    return view, fake


def _point(env_id: int) -> ContactPoint:
    return ContactPoint(
        env_id=env_id,
        group="feet",
        body1="left",
        body2="right",
        geom1=None,
        geom2=None,
        pos_w=torch.zeros(3, dtype=torch.float64),
        normal_w=None,
        frame_w=None,
        separation_m=0.0,
        normal_force_n=0.0,
        friction_force_w=None,
        contact_id=0,
    )


def test_per_env_state_read_defaults_to_the_scope() -> None:
    """A read with no explicit env uses the view's env id."""
    view, fake = _view(env_id=2)

    view.query.state.body_pose("torso")

    assert fake.state.body_pose_calls == [("torso", None, 2)]


def test_per_env_state_read_accepts_the_scope_explicitly() -> None:
    """An explicit selection that equals the scope is accepted."""
    view, fake = _view(env_id=2)

    view.query.state.root_pose("g1", env=2)

    assert fake.state.root_pose_calls == [("g1", 2)]


@pytest.mark.parametrize(
    "selection",
    [0, 3, [0, 2], torch.tensor([0, 2], dtype=torch.int64), [2, 2]],
)
def test_per_env_state_read_rejects_out_of_scope_selection(selection) -> None:
    """Any selection other than exactly the view's env is rejected."""
    view, _ = _view(env_id=2)

    with pytest.raises(EvalViewError, match=r"outside.*env_id=2"):
        view.query.state.joint_position("g1", env=selection)


def test_vector_state_read_passes_the_selection_through() -> None:
    """A vector view does not narrow an explicit caller selection."""
    view, fake = _view(env_id=None)

    view.query.state.body_pose("torso", env=[0, 2])
    view.query.state.body_pose("torso")

    assert fake.state.body_pose_calls == [("torso", None, [0, 2]), ("torso", None, None)]


def test_per_env_contact_manifold_and_scan_use_the_scope() -> None:
    """Grouped and full contact reads fix the per-env selection."""
    view, fake = _view(env_id=1)

    view.query.contacts.manifold("feet")
    view.query.contacts.scan(env=1)

    assert fake.contacts.manifold_calls == [("feet", 1)]
    assert fake.contacts.scan_calls == [1]


def test_per_env_contact_point_outside_scope_is_rejected() -> None:
    """A contact point from another env cannot be mapped through this view."""
    view, fake = _view(env_id=1)

    with pytest.raises(EvalViewError, match=r"env_id 2 is outside.*env_id=1"):
        view.query.contacts.jacobian(_point(2), "g1")
    with pytest.raises(EvalViewError, match=r"env_id 0 is outside.*env_id=1"):
        view.query.contacts.wrench_to_joint(_point(0), "g1")

    assert fake.contacts.point_calls == []


def test_per_env_contact_point_inside_scope_is_delegated() -> None:
    """A point owned by the scoped env is passed through unchanged."""
    view, fake = _view(env_id=1)
    point = _point(1)

    view.query.contacts.jacobian(point, "g1")

    assert fake.contacts.point_calls == [(point, "g1")]


def test_vector_contact_point_is_never_env_checked() -> None:
    """A vector view allows any contact point's own env."""
    view, fake = _view(env_id=None)

    point = _point(2)
    view.query.contacts.wrench_to_joint(point, "g1")

    assert len(fake.contacts.point_calls) == 1
    assert fake.contacts.point_calls[0][0] is point
    assert fake.contacts.point_calls[0][1] == "g1"


def test_per_env_actuator_state_uses_the_scope() -> None:
    """Actuator runtime reads obey the same per-env selection."""
    view, fake = _view(env_id=2)

    view.query.actuators.state("motor")

    assert fake.actuators.state_calls == [("motor", None, 2)]


def test_eval_query_exposes_only_the_read_subset() -> None:
    """No scratch world, no close and no raw-query attribute forwarding."""
    view, _ = _view(env_id=1)
    query = view.query

    public = {name for name in dir(query) if not name.startswith("_")}

    assert public == {
        "actuators",
        "backend",
        "capabilities",
        "contacts",
        "geometry",
        "kinematics",
        "model",
        "require",
        "state",
    }
    assert not hasattr(query, "scratch")
    assert not hasattr(query, "close")
    assert "__getattr__" not in type(query).__dict__


def test_geometry_has_no_scratch_parameter() -> None:
    """The evaluation geometry face cannot accept a writable scratch world."""
    view, _ = _view(env_id=1)

    signature = inspect.signature(view.query.geometry.distance)

    assert "scratch" not in signature.parameters


def test_require_and_identity_delegate_to_the_query() -> None:
    """Backend, capabilities and the capability gate are the query's own."""
    view, fake = _view(env_id=1)
    query = view.query

    assert query.backend == "fake"
    assert query.capabilities is fake.capabilities
    query.require("state.body_pose")
    assert fake.required == [("state.body_pose",)]


def test_eval_query_is_stale_after_the_view_closes() -> None:
    """Every query read fails after close instead of reading the next episode."""
    view, _ = _view(env_id=1)
    query = view.query

    view.close()

    with pytest.raises(EvalViewStaleError):
        _ = query.backend
    with pytest.raises(EvalViewStaleError):
        query.state.body_pose("torso")
    with pytest.raises(EvalViewStaleError):
        query.contacts.scan()
