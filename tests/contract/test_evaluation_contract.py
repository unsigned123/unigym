"""Public contract: the additive read-only evaluation context surface."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

import unigym.evaluation as evaluation
from unigym.evaluation import EnvView, EvalContext, EvalQuery

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
        "close",
        "device",
        "env_id",
        "has_query",
        "joint_pos_limits",
        "num_envs",
        "obs",
        "query",
        "record",
        "scope_size",
    } <= public


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
