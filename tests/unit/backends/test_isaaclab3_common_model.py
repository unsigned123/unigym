"""Shared IsaacLab 3 model provider: factory, declared build, duck-typed describe."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from unigym.backends.shared.isaaclab3_common.model import (
    IsaacLabModelProvider,
    provider_for,
)
from unigym.core.quat import to_wxyz
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
