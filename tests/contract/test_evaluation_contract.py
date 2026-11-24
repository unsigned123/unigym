"""Public contract: the additive read-only evaluation context surface."""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys

import pytest

import unigym.evaluation as evaluation
import unigym.terminations as terminations
from unigym.evaluation import EnvView, EvalContext, EvalQuery
from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime, ResolvedControl

_FORBIDDEN_VIEW_ATTRS = frozenset(
    {
        "env",
        "native",
        "step",
        "reset",
        "reset_envs",
        "forward",
        "set_root_state",
        "set_joint_state",
        "scratch",
    }
)


def test_public_names_are_stable() -> None:
    """The new public module exports exactly the documented names."""
    assert set(evaluation.__all__) == {
        "EnvStepContext",
        "EnvView",
        "EvalContext",
        "EvalQuery",
        "EvalQueryUnavailableError",
        "EvalViewError",
        "EvalViewStaleError",
    }
    for name in evaluation.__all__:
        assert getattr(evaluation, name) is not None


def test_env_view_is_a_concrete_class_not_a_protocol() -> None:
    """The view is a concrete owner, so a leaking substitute is not accepted."""
    assert isinstance(EnvView, type)
    assert getattr(EnvView, "_is_protocol", False) is False
    assert EnvView.__slots__


def test_env_view_exposes_no_mutable_or_raw_surface() -> None:
    """No backend, native handle, step/reset/write or scratch is public."""
    public = {name for name in dir(EnvView) if not name.startswith("_")}
    assert not (public & _FORBIDDEN_VIEW_ATTRS)
    assert {
        "articulation",
        "articulation_facts",
        "articulation_names",
        "close",
        "controls",
        "device",
        "env_id",
        "for_env",
        "has_query",
        "joint_pos_limits",
        "num_envs",
        "obs",
        "query",
        "record",
        "scope_size",
        "with_record",
    } <= public


def test_env_view_exposes_only_frozen_control_facts_and_derivation() -> None:
    """Controls project to ``ResolvedControl`` and derived views stay read-only."""
    art = ArticulationRuntime(name="g1", root="fixed", joint_names=("j",), q0=(0.0,))
    control = ControlRuntime(
        name="legs",
        adapter="manual",
        articulation="g1",
        joint_names=("j",),
        q0=(0.0,),
        role="body",
    )
    view = EnvView(
        record={},
        obs=None,
        env_id=None,
        num_envs=2,
        articulations={"g1": art},
        controls={("g1", "legs"): control},
    )

    row = view.controls[("g1", "legs")]
    assert isinstance(row, ResolvedControl)
    assert not isinstance(row, ControlRuntime)
    assert view.articulation_names == ("g1",)
    assert view.articulation_facts("g1")["action_group"] == "legs"

    child = view.with_record({"x": 1})
    per_env = view.for_env(1)
    assert child.env_id is None
    assert per_env.env_id == 1
    assert isinstance(inspect.getattr_static(EnvView, "controls"), property)
    assert isinstance(inspect.getattr_static(EnvView, "articulation_names"), property)


def test_eval_query_has_no_scratch_or_close() -> None:
    """The adapter cannot open a writable world or close the shared query."""
    public = {name for name in dir(EvalQuery) if not name.startswith("_")}
    assert "scratch" not in public
    assert "close" not in public
    assert not hasattr(EvalQuery, "__getattr__")
    assert {
        "actuators",
        "backend",
        "capabilities",
        "contacts",
        "geometry",
        "kinematics",
        "model",
        "require",
        "state",
    } <= public


def test_eval_context_has_only_env_and_optional_t() -> None:
    """The context does not duplicate record/obs/episode payload fields."""
    fields = {field.name for field in dataclasses.fields(EvalContext)}
    assert fields == {"env", "t"}
    assert EvalContext.__dataclass_params__.frozen is True


@pytest.mark.subprocess
def test_import_evaluation_loads_no_heavy_backend() -> None:
    """A fresh interpreter imports the context without an engine."""
    code = (
        "import sys\n"
        "import unigym.evaluation\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('mujoco', 'isaaclab', 'omni', 'warp'))\n"
        "assert not heavy, heavy\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def _positional_parameters(func) -> list[str]:
    return [
        name
        for name, parameter in inspect.signature(func).parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def test_termination_check_takes_one_live_context() -> None:
    """The public termination API is ``check(ctx)`` with no recorded state."""
    assert _positional_parameters(terminations.Termination.check) == ["self", "ctx"]
    assert _positional_parameters(terminations.TerminationGroup.check) == ["self", "ctx"]
    assert _positional_parameters(terminations.TerminationGroup.check_detail) == [
        "self",
        "ctx",
    ]


def test_termination_state_is_not_exported() -> None:
    """The recorded/offline termination state concept is gone."""
    assert "TerminationState" not in terminations.__all__
    assert not hasattr(terminations, "TerminationState")


def test_condition_package_exports_only_the_torch_free_protocol() -> None:
    """The base conditions import exposes the protocol, not concrete torch classes."""
    import unigym.conditions as conditions

    assert conditions.__all__ == ["Condition", "ConditionError"]
    assert _positional_parameters(conditions.Condition.evaluate) == ["self", "ctx"]
    assert issubclass(conditions.ConditionError, ValueError)
    assert not hasattr(conditions, "HeightCondition")
    assert not hasattr(conditions, "TiltCondition")
    assert not hasattr(conditions, "JointLimitCondition")
