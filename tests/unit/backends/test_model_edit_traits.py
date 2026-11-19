"""Registry/trait matrix, docs parity and the IsaacLab COM/iquat publication."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from unigym.backends import registry
from unigym.backends.registry import (
    ISAACLAB2_TRAITS,
    ISAACLAB3_TRAITS,
    MJWARP_TRAITS,
    MUJOCO_TRAITS,
    ModelEditField,
    Trait,
    backend_info,
    validate_trait_rules,
)
from unigym.env.backends.base import BackendCapabilities

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "contracts" / "model_edit_contract.md"


def test_model_edit_is_declared_on_the_three_real_backends() -> None:
    """MuJoCo/mjwarp/IsaacLab2 declare MODEL_EDIT; IsaacLab3 does not."""
    assert Trait.MODEL_EDIT in MUJOCO_TRAITS
    assert Trait.MODEL_EDIT in MJWARP_TRAITS
    assert Trait.MODEL_EDIT in ISAACLAB2_TRAITS
    assert Trait.MODEL_EDIT not in ISAACLAB3_TRAITS

    assert Trait.MODEL_EDIT in BackendCapabilities(model_edit=True).traits()
    assert Trait.MODEL_EDIT not in BackendCapabilities().traits()
    assert Trait.MODEL_EDIT in BackendCapabilities.full().traits()


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp", "isaaclab2"])
def test_provider_fields_equal_the_registry_field_set(backend) -> None:
    """A backend's writer declaration is the registry field set."""
    info = backend_info(backend)
    assert info.model_edit_provider is not None
    assert info.model_edit_fields
    assert info.model_edit_provider.fields == info.model_edit_fields
    provider = info.model_edit_provider.loader()
    assert getattr(provider, "fields") == info.model_edit_fields


def test_isaaclab3_has_no_model_edit_registration() -> None:
    """The EA skeleton declares neither the trait nor provider/fields."""
    info = backend_info("isaaclab3")
    assert info.model_edit_provider is None
    assert info.model_edit_fields == frozenset()


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
            "isaaclab3": cells[4],
        }
    assert set(matrix) == {field.value for field in ModelEditField}
    for backend in ("mujoco", "mjwarp", "isaaclab2", "isaaclab3"):
        fields = backend_info(backend).model_edit_fields
        for field in ModelEditField:
            documented = matrix[field.value][backend]
            supported = documented != "no"
            assert (field in fields) == supported, (backend, field, documented)


def test_validate_trait_rules_rejects_a_field_provider_mismatch() -> None:
    """A provider/fields/trait inconsistency fails at registration validation."""
    info = registry._STATIC_INFO["mujoco"]
    provider = info.model_edit_provider
    assert provider is not None
    broken = replace(info, model_edit_fields=frozenset({ModelEditField.BODY_MASS}))
    with pytest.raises(RuntimeError, match="model_edit_provider fields"):
        validate_trait_rules({"mujoco": broken})
    missing_provider = replace(info, model_edit_provider=None)
    with pytest.raises(RuntimeError, match="model_edit_provider is None"):
        validate_trait_rules({"mujoco": missing_provider})
    sans_trait = replace(
        info,
        traits=frozenset(info.traits - {Trait.MODEL_EDIT}),
    )
    with pytest.raises(RuntimeError, match="without the MODEL_EDIT trait"):
        validate_trait_rules({"mujoco": sans_trait})


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
