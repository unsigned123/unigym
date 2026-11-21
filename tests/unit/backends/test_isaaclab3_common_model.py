"""Shared IsaacLab 3 model provider: factory, declared build, duck-typed describe."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from unigym.backends.shared.isaaclab3_common import model as model_module
from unigym.backends.shared.isaaclab3_common.model import (
    IsaacLabModelProvider,
    _resolve_root_prim,
    provider_for,
)
from unigym.core.quat import to_wxyz
from unigym.model.errors import ModelIRError
from unigym.model.ir import ModelIR


@pytest.mark.parametrize("backend", ["isaaclab3_physx", "isaaclab3_newton"])
def test_provider_for_returns_a_zero_arg_constructible_class(backend):
    """The registry factory shape is a zero-arg class bound to one label."""
    cls = provider_for(backend)

    assert issubclass(cls, IsaacLabModelProvider)
    provider = cls()
    assert provider.backend == backend


def test_build_emits_an_empty_declared_model_without_an_engine():
    """An empty scene cfg yields an uncompiled, empty declared IR."""
    provider = provider_for("isaaclab3_physx")()

    ir = provider.build({"robot": None, "props": {}, "cameras": {}})

    assert isinstance(ir, ModelIR)
    assert ir.backend == "isaaclab3_physx"
    assert ir.compiled is False
    assert len(ir.bodies) == 0
    assert len(ir.joints) == 0
    assert len(ir.cameras) == 0
    assert len(ir.actuators) == 0


@dataclass
class _CameraCfg:
    prim_path: str = "/World/envs/env_0/Camera"
    offset: object | None = None
    width: int = 640
    height: int = 480


def test_build_reorders_camera_rotation_from_xyzw_to_wxyz():
    """IsaacLab 3 ``offset.rot`` is xyzw; the IR stores wxyz."""
    rot_xyzw = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    camera = _CameraCfg(
        offset=SimpleNamespace(pos=(0.1, 0.2, 0.3), rot=rot_xyzw),
    )
    provider = provider_for("isaaclab3_physx")()

    ir = provider.build({"robot": None, "props": {}, "cameras": {"wrist": camera}})

    row = ir.cameras["wrist"]
    expected = tuple(float(value) for value in to_wxyz(rot_xyzw))
    assert row.quat == pytest.approx(expected)
    assert row.pos == pytest.approx((0.1, 0.2, 0.3))
    assert row.body == "world"


@dataclass
class _NewtonData:
    """A Newton-like data surface with no body/joint tensors declared."""

    joint_names: list[str] = field(default_factory=list)
    body_names: list[str] = field(default_factory=list)


@dataclass
class _NewtonArticulation:
    """An articulation with no ``root_physx_view`` (the Newton shape)."""

    body_names: list[str]
    joint_names: list[str] = field(default_factory=list)
    data: object | None = None
    cfg: object | None = None


def test_describe_without_root_physx_view_yields_none_mass():
    """Newton has no PhysX view: describe must not invent masses."""
    articulation = _NewtonArticulation(body_names=["base"], data=_NewtonData())
    provider = provider_for("isaaclab3_newton")()

    ir = provider.describe(articulation)

    assert isinstance(ir, ModelIR)
    assert ir.backend == "isaaclab3_newton"
    assert ir.compiled is True
    assert ir.bodies["base"].mass is None
    assert ir.bodies["base"].inertia is None


@pytest.mark.parametrize(
    "declared",
    [
        "{ENV_REGEX_NS}/Robot",
        "/World/envs/env_.*/Robot",
        "/World/envs/env_[^/]+/Robot",
        "/World/envs/env_7/Robot",
    ],
)
def test_resolve_root_prim_normalizes_every_env_regex(monkeypatch, declared):
    """Every IsaacLab 3 env-regex spelling resolves to the env-0 asset root."""
    monkeypatch.setattr(model_module, "_usd_stage", lambda: None)
    articulation = SimpleNamespace(cfg=SimpleNamespace(prim_path=declared))

    assert _resolve_root_prim(articulation) == "/World/envs/env_0/Robot"


class _InvalidStage:
    """A stage whose prims never resolve."""

    def GetPrimAtPath(self, path):
        del path
        return SimpleNamespace(IsValid=lambda: False)


def test_resolve_root_prim_falls_back_to_the_live_asset_root(monkeypatch):
    """An invalid declared prim falls back to ``root_physx_view.prim_paths[0]``."""
    monkeypatch.setattr(model_module, "_usd_stage", lambda: _InvalidStage())
    view = SimpleNamespace(prim_paths=["/World/envs/env_0/Robot"])
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/Missing"),
        root_physx_view=view,
    )

    assert _resolve_root_prim(articulation) == "/World/envs/env_0/Robot"


def test_resolve_root_prim_keeps_the_declared_path_without_a_view(monkeypatch):
    """Without a PhysX view the normalized declared prim is retained."""
    monkeypatch.setattr(model_module, "_usd_stage", lambda: _InvalidStage())
    articulation = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"))

    assert _resolve_root_prim(articulation) == "/World/envs/env_0/Robot"


def test_describe_refuses_to_emit_unknown_joint_types(monkeypatch):
    """An unresolved root prim is a loud failure, never an ``unknown`` joint IR."""
    monkeypatch.setattr(model_module, "_usd_stage", lambda: None)
    articulation = _NewtonArticulation(body_names=["base"], joint_names=["j1"], data=_NewtonData())

    with pytest.raises(ModelIRError, match="unknown"):
        IsaacLabModelProvider("isaaclab3_newton").describe(articulation)
