"""The SimulationBuilder live transition: once-only, sealing and failure cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.sim import BuilderSealedError, Control, Simulation, SimulationBuilder
from unigym.world import World
from tests._support.fakes.runner import FakeRunner
from tests._support.sim_fixtures import write_sim_pack


class _RunSeam:
    """A Run-shaped seam: one FakeRunner, build/close recording, optional failure."""

    instances: list[_RunSeam] = []
    fail_build = False

    def __init__(self, cfg: object, **kwargs: object) -> None:
        self.cfg = cfg
        self.kwargs = kwargs
        self.closed = False
        self.build_calls = 0
        self.runner_obj = FakeRunner()
        self.instance = SimpleNamespace(env=self.runner_obj.env)
        _RunSeam.instances.append(self)

    def build(self) -> SimpleNamespace:
        """Return the live instance or fail through the injected build fault."""
        self.build_calls += 1
        if _RunSeam.fail_build:
            raise RuntimeError("run build failure")
        return self.instance

    def runner(self, kind: str | None = None) -> FakeRunner:
        """Return the one runner the builder must accept as ExternalRunner."""
        if kind != "external":
            raise AssertionError(f"unexpected runner kind {kind!r}")
        return self.runner_obj

    def close(self) -> None:
        """Record the failure-path close without touching engine state."""
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_builder_state() -> Iterator[None]:
    """Restore the robot registry and reset the run seam between cases."""
    snapshot = snapshot_robot_registry()
    _RunSeam.instances.clear()
    _RunSeam.fail_build = False
    yield
    _RunSeam.instances.clear()
    _RunSeam.fail_build = False
    restore_robot_registry(snapshot)


def _builder(tmp_path: Path) -> SimulationBuilder:
    """A declarable builder with one world and one articulation."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.base(mjcf="world.xml")
    builder.add_articulation(
        ref,
        control=Control.position_pd(joint_set="body", kp=1.0, kd=0.1),
    )
    return builder


def _patch_run(monkeypatch: pytest.MonkeyPatch, *, external: bool = True) -> None:
    """Route the builder's ``Run`` construction through the seam."""
    monkeypatch.setattr("unigym.run.api.Run", _RunSeam)
    if external:
        monkeypatch.setattr("unigym.runner.external.ExternalRunner", FakeRunner)


def test_build_is_a_once_only_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The second build returns the cached simulation without a second Run."""
    _patch_run(monkeypatch)
    builder = _builder(tmp_path)

    first = builder.build()
    second = builder.build()

    assert isinstance(first, Simulation)
    assert second is first
    assert len(_RunSeam.instances) == 1
    assert _RunSeam.instances[0].build_calls == 1


def test_declaration_after_build_is_refused_by_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built builder refuses mutation and leaves the world unchanged."""
    _patch_run(monkeypatch)
    builder = _builder(tmp_path)
    original = builder.world
    builder.build()

    with pytest.raises(BuilderSealedError, match="sealed"):
        builder.world = World.base(mjcf="replacement.xml")

    assert builder.world is original


def test_failed_build_closes_the_run_and_does_not_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live-transition failure closes the Run and keeps declarations open."""
    _patch_run(monkeypatch)
    _RunSeam.fail_build = True
    builder = _builder(tmp_path)
    original = builder.world

    with pytest.raises(RuntimeError, match="run build failure"):
        builder.build()

    assert len(_RunSeam.instances) == 1
    assert _RunSeam.instances[0].closed is True
    replacement = World.base(mjcf="replacement.xml")
    builder.world = replacement
    assert builder.world is replacement
    assert replacement is not original


def test_non_external_runner_is_refused_and_closes_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner of the wrong kind fails the live transition without sealing."""
    _patch_run(monkeypatch, external=False)
    builder = _builder(tmp_path)

    with pytest.raises(RuntimeError, match="expected ExternalRunner"):
        builder.build()

    assert len(_RunSeam.instances) == 1
    assert _RunSeam.instances[0].closed is True
    builder.world = World.base(mjcf="replacement.xml")
