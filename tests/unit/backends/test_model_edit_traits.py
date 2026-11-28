"""Registry/trait matrix, docs parity and the IsaacLab COM/iquat publication."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from unigym.backends.compose import BackendRuntime, backend_runtime, validate_backend_runtime
from unigym.capabilities.traits import (
    ISAACLAB2_TRAITS,
    ISAACLAB3_NEWTON_TRAITS,
    ISAACLAB3_PHYSX_TRAITS,
    MJWARP_TRAITS,
    MUJOCO_TRAITS,
    ModelEditField,
    Trait,
)
from unigym.env.backends.base import BackendCapabilities

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "contracts" / "model_edit_contract.md"


def test_model_edit_is_declared_on_the_three_real_backends() -> None:
    """MuJoCo/mjwarp/IsaacLab2 declare MODEL_EDIT; the IsaacLab 3 stubs do not."""
    assert Trait.MODEL_EDIT in MUJOCO_TRAITS
    assert Trait.MODEL_EDIT in MJWARP_TRAITS
    assert Trait.MODEL_EDIT in ISAACLAB2_TRAITS
    assert Trait.MODEL_EDIT not in ISAACLAB3_PHYSX_TRAITS
    assert Trait.MODEL_EDIT not in ISAACLAB3_NEWTON_TRAITS

    assert Trait.MODEL_EDIT in BackendCapabilities(model_edit=True).traits()
    assert Trait.MODEL_EDIT not in BackendCapabilities().traits()
    assert Trait.MODEL_EDIT in BackendCapabilities.full().traits()


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp", "isaaclab2"])
def test_provider_fields_equal_the_spec_field_set(backend) -> None:
    """A backend's writer declaration is the spec field set."""
    runtime = backend_runtime(backend)
    provider = runtime.bindings.model_edit_provider
    assert provider is not None
    assert runtime.spec.model_edit_fields
    assert provider.fields == runtime.spec.model_edit_fields
    loaded = provider.loader()
    assert getattr(loaded, "fields") == runtime.spec.model_edit_fields


def test_isaaclab3_has_no_model_edit_registration() -> None:
    """The P1 skeletons declare neither the trait nor provider/fields."""
    for name in ("isaaclab3_physx", "isaaclab3_newton"):
        runtime = backend_runtime(name)
        assert runtime.bindings.model_edit_provider is None
        assert runtime.spec.model_edit_fields == frozenset()


def test_model_edit_field_sets_match_the_documented_matrix() -> None:
    """docs/contracts/model_edit_contract.md is generated from the registry."""
    text = DOC.read_text(encoding="utf-8")
    matrix: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not cells[0].startswith(("body.", "geom.")):
            continue
        matrix[cells[0]] = {
            "mujoco": cells[1],
            "mjwarp": cells[2],
            "isaaclab2": cells[3],
            "isaaclab3_physx": cells[4],
            "isaaclab3_newton": cells[4],
        }
    assert set(matrix) == {field.value for field in ModelEditField}
    for backend in ("mujoco", "mjwarp", "isaaclab2", "isaaclab3_physx", "isaaclab3_newton"):
        fields = backend_runtime(backend).spec.model_edit_fields
        for field in ModelEditField:
            documented = matrix[field.value][backend]
            supported = documented != "no"
            assert (field in fields) == supported, (backend, field, documented)


def test_validate_backend_runtime_rejects_a_field_provider_mismatch() -> None:
    """A provider/fields/trait inconsistency fails at composition validation."""
    runtime = backend_runtime("mujoco")
    spec = runtime.spec
    bindings = runtime.bindings
    assert bindings.model_edit_provider is not None
    broken = replace(spec, model_edit_fields=frozenset({ModelEditField.BODY_MASS}))
    with pytest.raises(RuntimeError, match="model_edit_provider fields"):
        validate_backend_runtime(BackendRuntime(spec=broken, bindings=bindings))
    missing_provider = replace(bindings, model_edit_provider=None)
    with pytest.raises(RuntimeError, match="model_edit_provider is None"):
        validate_backend_runtime(BackendRuntime(spec=spec, bindings=missing_provider))
    sans_trait = replace(spec, traits=frozenset(spec.traits - {Trait.MODEL_EDIT}))
    with pytest.raises(RuntimeError, match="without the MODEL_EDIT trait"):
        validate_backend_runtime(BackendRuntime(spec=sans_trait, bindings=bindings))


def test_model_edit_module_stays_engine_free() -> None:
    """The neutral editor imports no engine module (static AST check)."""
    source = (ROOT / "unigym" / "model" / "edit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports & {"mujoco", "mujoco_warp", "warp", "isaaclab", "torch"}


class _PhysXViewLike:
    """A PhysX-view-shaped getter source (no IsaacLab import)."""

    def __init__(self) -> None:
        self.body_names = ["link", "tip"]
        self._masses = [[2.0, 0.5]]
        self._coms = [
            [
                [0.01, 0.02, 0.03],
                [-0.01, 0.0, 0.04],
            ]
        ]
        self._inertias = [
            [
                [2.0, 0.5, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0],
                [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3],
            ]
        ]

    def get_masses(self):
        return self._masses

    def get_coms(self):
        return self._coms

    def get_inertias(self):
        return self._inertias


def test_isaaclab2_describe_publishes_com_and_inertia_quat() -> None:
    """describe() fills ``ipos`` from get_coms and a principal ``iquat``."""
    import numpy as np

    from unigym.backends.isaaclab2.model import IsaacLabModelProvider

    data = SimpleNamespace(
        joint_names=[],
        body_names=["link", "tip"],
        body_pos_w=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
        body_quat_w=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        joint_pos_limits=[],
        joint_vel_limits=[],
        default_joint_pos=[],
    )
    view = _PhysXViewLike()
    articulation = SimpleNamespace(
        data=data,
        joint_names=[],
        body_names=["link", "tip"],
        root_physx_view=view,
        actuators={},
    )

    ir = IsaacLabModelProvider().describe(articulation)
    link = ir.bodies["link"]
    tip = ir.bodies["tip"]
    assert link.ipos == (0.01, 0.02, 0.03)
    assert tip.ipos == (-0.01, 0.0, 0.04)
    assert link.inertia is not None and link.iquat is not None
    # Rebuild the full tensor from the published principal pair.
    w, x, y, z = link.iquat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    reconstructed = rotation @ np.diag(link.inertia) @ rotation.T
    expected = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert np.allclose(reconstructed, expected, atol=1e-9)
    assert tip.iquat == (1.0, 0.0, 0.0, 0.0)


def test_isaaclab2_describe_body_universe_is_the_live_data_set() -> None:
    """describe() publishes the measured live PhysX bodies, not a static list."""
    from unigym.backends.isaaclab2.model import IsaacLabModelProvider

    live = ["pelvis", "left_hand/left_hand_palm_link"]
    data = SimpleNamespace(
        joint_names=[],
        body_names=list(live),
        body_pos_w=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
        body_quat_w=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        joint_pos_limits=[],
        joint_vel_limits=[],
        default_joint_pos=[],
    )
    view = SimpleNamespace(
        body_names=list(live),
        get_masses=lambda: [[1.0, 1.0]],
        get_inertias=lambda: [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * 2],
        get_coms=lambda: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    articulation = SimpleNamespace(
        data=data,
        joint_names=[],
        # A stale/static authored list must never define the runtime universe.
        body_names=["stale_authored_only_body"],
        root_physx_view=view,
        actuators={},
    )

    ir = IsaacLabModelProvider().describe(articulation)

    assert tuple(ir.bodies.names()) == tuple(live)
    assert "stale_authored_only_body" not in ir.bodies


def test_isaaclab2_describe_requires_live_data_body_names() -> None:
    """An empty live data block fails instead of reading articulation names."""
    from unigym.backends.isaaclab2.model import IsaacLabModelProvider
    from unigym.model.errors import ModelIRError

    articulation = SimpleNamespace(
        data=SimpleNamespace(body_names=[]),
        joint_names=[],
        body_names=["stale_authored_only_body"],
    )

    with pytest.raises(ModelIRError, match="data.body_names"):
        IsaacLabModelProvider().describe(articulation)


class _PhysXViewProbe:
    """PhysX-view-shaped arrays with a one-shot inertia setter failure."""

    def __init__(self) -> None:
        import torch

        self.body_names = ["first", "second"]
        self.masses = torch.full((1, 2), 1.0)
        self.coms = torch.zeros(1, 2, 7)
        self.coms[:, :, 6] = 1.0  # xyzw identity quaternion
        self.inertias = torch.eye(3).reshape(1, 1, 9).repeat(1, 2, 1)
        self.fail_inertia = True
        self.writes: list[str] = []

    def get_masses(self):
        return self.masses

    def get_coms(self):
        return self.coms

    def get_inertias(self):
        return self.inertias

    def set_masses(self, data, indices):
        self.writes.append("masses")
        self.masses = data.clone()

    def set_coms(self, data, indices):
        self.writes.append("coms")
        self.coms = data.clone()

    def set_inertias(self, data, indices):
        self.writes.append("inertias")
        if self.fail_inertia:
            self.fail_inertia = False
            raise RuntimeError("injected inertia setter failure")
        self.inertias = data.clone()


def test_isaaclab2_mid_loop_failure_restores_every_touched_row() -> None:
    """A setter failure after an earlier body write restores the whole snapshot."""
    from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider
    from unigym.model.edit import BodyEdit
    from unigym.model.errors import ModelEditError

    view = _PhysXViewProbe()
    asset = SimpleNamespace(
        body_names=["stale_asset_only_body"],
        data=SimpleNamespace(body_names=["first", "second"]),
        root_physx_view=view,
    )
    rebuild_calls: list[str] = []
    env = SimpleNamespace(
        robots={"cartpole": asset},
        _model_revision=0,
        _rebuild_model_set=lambda: rebuild_calls.append("rebuild"),
    )
    masses_before = view.masses.clone()
    coms_before = view.coms.clone()
    inertias_before = view.inertias.clone()
    edits = (
        BodyEdit("cartpole", "first", None, {ModelEditField.BODY_MASS: 2.0}),
        BodyEdit(
            "cartpole",
            "second",
            None,
            {
                ModelEditField.BODY_INERTIA: (0.2, 0.3, 0.4),
                ModelEditField.BODY_INERTIA_QUAT: (1.0, 0.0, 0.0, 0.0),
            },
        ),
    )

    with pytest.raises(ModelEditError, match="injected inertia setter failure"):
        IsaacLab2ModelEditProvider().apply(env, edits, ())

    assert view.masses.tolist() == masses_before.tolist()
    assert view.coms.tolist() == coms_before.tolist()
    assert view.inertias.tolist() == inertias_before.tolist()
    assert rebuild_calls == []


def test_isaaclab2_model_edit_requires_live_data_body_names() -> None:
    """An empty live data block fails instead of reading view/asset names."""
    from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider
    from unigym.model.edit import BodyEdit
    from unigym.model.errors import ModelEditCapabilityError

    view = _PhysXViewProbe()
    asset = SimpleNamespace(
        body_names=["first", "second"],
        data=SimpleNamespace(body_names=[]),
        root_physx_view=view,
    )
    env = SimpleNamespace(
        robots={"cartpole": asset},
        _model_revision=0,
        _rebuild_model_set=lambda: None,
    )
    edits = (BodyEdit("cartpole", "first", None, {ModelEditField.BODY_MASS: 2.0}),)

    with pytest.raises(ModelEditCapabilityError, match="data.body_names"):
        IsaacLab2ModelEditProvider().apply(env, edits, ())
    assert env._model_revision == 0


def test_backend_capability_bits_project_model_edit() -> None:
    """Each real backend's capability object exposes Trait.MODEL_EDIT."""
    from unigym.backends.mjwarp.env import BACKEND_CAPABILITIES as MJWARP_CAPABILITIES
    from unigym.backends.mujoco.env import BACKEND_CAPABILITIES as MUJOCO_CAPABILITIES

    assert Trait.MODEL_EDIT in MUJOCO_CAPABILITIES.traits()
    assert Trait.MODEL_EDIT in MJWARP_CAPABILITIES.traits()
    assert Trait.MODEL_EDIT not in BackendCapabilities().traits()
