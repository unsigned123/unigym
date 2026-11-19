"""Run facade model-edit gating, delegation and read-only ``run.model``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unigym._typing import JsonValue
from unigym.model import Body, ModelIR, ModelSet, Table
from unigym.run.api import Run
from unigym.run.instance import RunInstance


def _cfg() -> dict[str, JsonValue]:
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


def _model_set() -> ModelSet:
    ir = ModelIR(
        name="arm",
        backend="mujoco",
        compiled=True,
        bodies=Table(
            [
                Body(name="world", body_id=0),
                Body(name="link", body_id=1, mass=1.0),
            ]
        ),
    )
    return ModelSet({"arm": ir}, merged=ir)


class _EditEnv:
    def __init__(self, *, status: str = "ok") -> None:
        self.status = status
        self.calls: list[str] = []
        self.steps = 0
        self.model_set = _model_set()

    def model_edit(self):
        self.calls.append("model_edit")
        if self.status == "missing":
            raise NotImplementedError("no model_edit")
        return "editor"

    def step(self, *args, **kwargs):  # pragma: no cover - must not run
        self.steps += 1
        raise AssertionError("opening an editor must not step")

    def close(self) -> None:
        self.calls.append("close")


def _instance(backend: str, env: object) -> RunInstance:
    return RunInstance(
        spec=SimpleNamespace(backend=backend),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=env,
        bridge=None,
    )


def test_run_instance_delegates_to_the_live_env() -> None:
    """RunInstance.model_edit -> env.model_edit without stepping."""
    env = _EditEnv()
    instance = _instance("mujoco", env)

    assert instance.model_edit() == "editor"
    assert env.calls == ["model_edit"]
    assert env.steps == 0


def test_run_instance_rejects_a_backend_without_the_trait() -> None:
    """IsaacLab3 declares no MODEL_EDIT; the facade names the trait."""
    instance = _instance("isaaclab3", _EditEnv())
    with pytest.raises(NotImplementedError, match=r"isaaclab3.*MODEL_EDIT"):
        instance.model_edit()


def test_run_instance_rejects_a_closed_run() -> None:
    """A closed run cannot open a model-edit transaction."""
    instance = _instance("mujoco", _EditEnv())
    instance.close()
    with pytest.raises(RuntimeError, match="closed: model edits"):
        instance.model_edit()


def test_run_instance_rejects_a_missing_env_method() -> None:
    """A declared trait with no env method still fails loudly."""
    env = SimpleNamespace(model_set=_model_set())
    instance = _instance("mujoco", env)
    with pytest.raises(NotImplementedError, match="no model_edit"):
        instance.model_edit()


def test_run_model_edit_builds_lazily_and_delegates(monkeypatch) -> None:
    """Run.model_edit builds once and delegates to the instance."""
    run = Run(_cfg())
    env = _EditEnv()
    instance = _instance("mujoco", env)
    builds: list[str] = []

    def build(inspect: bool = False) -> RunInstance:
        builds.append("build")
        return instance

    monkeypatch.setattr(run, "build", build)

    assert run.model_edit() == "editor"
    assert builds == ["build"]
    assert env.calls == ["model_edit"]


def test_run_model_edit_closed_run_fails_before_building(monkeypatch) -> None:
    """A closed Run refuses the editor and never triggers a build."""
    run = Run(_cfg())
    monkeypatch.setattr(
        run,
        "build",
        lambda inspect=False: pytest.fail("closed Run must not build"),
    )
    run.close()

    with pytest.raises(RuntimeError, match="closed: model edits"):
        run.model_edit()


def test_run_model_stays_a_read_only_modelset(monkeypatch) -> None:
    """``run.model`` is the published ModelSet, never the editor."""
    run = Run(_cfg())
    env = _EditEnv()
    instance = _instance("mujoco", env)
    monkeypatch.setattr(run, "build", lambda inspect=False: instance)

    model = run.model
    assert isinstance(model, ModelSet)
    assert model is env.model_set
    assert not hasattr(model, "set")
    assert run.model_edit() == "editor"
