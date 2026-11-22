"""P2-D backend-gap classifications stay declarations, not new support.

The unit layer does not start an engine.  No cheap MuJoCo geometry-distance
fixture exists here, so the mesh-exact check is kept at the capability/doc
level and asserts the result-type default instead of reading a live
``DistanceResult`` from a simulator.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from unigym.backends.registry import Trait
from unigym.env.backends.base import BackendCapabilities
from unigym.query import DistanceResult
from unigym.query.capabilities import QueryCapabilities


def _backend_capabilities() -> dict[str, QueryCapabilities]:
    """Return the five backend capability declarations by canonical name."""
    from unigym.backends.isaaclab2.query import ISAACLAB2_QUERY_CAPABILITIES
    from unigym.backends.mjwarp.query import MJWARP_QUERY_CAPABILITIES
    from unigym.backends.mujoco.query import MUJOCO_QUERY_CAPABILITIES
    from unigym.backends.shared.isaaclab3_common.query import (
        ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
        ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
    )

    return {
        "mujoco": MUJOCO_QUERY_CAPABILITIES,
        "mjwarp": MJWARP_QUERY_CAPABILITIES,
        "isaaclab2": ISAACLAB2_QUERY_CAPABILITIES,
        "isaaclab3_physx": ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
        "isaaclab3_newton": ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
    }


def test_mujoco_contact_jacobian_detail_declares_dense_only() -> None:
    """MuJoCo names native contact rows and states sparse ``efc_J`` is unmapped."""
    from unigym.backends.mujoco.query import MUJOCO_QUERY_CAPABILITIES

    caps = MUJOCO_QUERY_CAPABILITIES
    detail = caps.contacts.jacobian.detail

    assert caps.supports("contacts.jacobian") == "native"
    assert "sparse efc_J" in detail
    assert "not mapped" in detail
    assert "dense" in caps.contacts.wrench_to_joint.detail


def test_mjwarp_contact_jacobian_detail_declares_synthetic_dense_only() -> None:
    """mjWarp names synthetic CPU-mirror rows and states sparse is unmapped."""
    from unigym.backends.mjwarp.query import MJWARP_QUERY_CAPABILITIES

    caps = MJWARP_QUERY_CAPABILITIES
    detail = caps.contacts.jacobian.detail

    assert caps.supports("contacts.jacobian") == "synthetic"
    assert "sparse efc_J" in detail
    assert "not mapped" in detail
    assert "dense" in caps.contacts.wrench_to_joint.detail


def test_geometry_distance_result_never_claims_mesh_exact() -> None:
    """A distance result defaults ``mesh_exact`` to false; no engine is started."""
    from unigym.backends.mujoco.query import MUJOCO_QUERY_CAPABILITIES

    result = DistanceResult(
        env_ids=torch.tensor([0], dtype=torch.int64),
        signed_distance_m=torch.tensor([0.1], dtype=torch.float64),
        closest_a_w=torch.zeros((1, 3), dtype=torch.float64),
        closest_b_w=torch.zeros((1, 3), dtype=torch.float64),
        normal_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        capped=torch.tensor([False], dtype=torch.bool),
    )

    assert MUJOCO_QUERY_CAPABILITIES.supports("geometry.distance") == "native"
    assert result.mesh_exact is False


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_isaaclab_gap_operations_stay_declared_not_native(backend: str) -> None:
    """IL contact/Newton/actuator/scratch gaps keep a non-native declared level."""
    caps = _backend_capabilities()[backend]
    unsupported = (
        "model.geom_contact",
        "state.root_state",
        "state.com_state",
        "state.object_state",
        "state.object_com_state",
        "jacobian.object",
        "contacts.jacobian",
        "contacts.wrench_to_joint",
        "actuators.state",
    )

    for op in unsupported:
        group, _, member = op.partition(".")
        feature = getattr(getattr(caps, group), member)
        assert feature.support == "unsupported", op
        assert feature.detail.strip(), op

    assert caps.supports("actuators.facts") == "partial"
    assert caps.supports("scratch.geometry") == "unsupported"
    assert caps.contacts.manifold.support in {"partial", "unsupported"}
    assert caps.contacts.manifold.detail.strip()
    if backend == "isaaclab2":
        assert caps.scratch.mode == "copy"
        assert caps.scratch.isolation == "bit_exact"
        assert caps.supports("scratch.copy") == "unsupported"
        assert caps.supports("contacts.manifold") == "unsupported"
    else:
        assert caps.scratch.mode == "transaction"
        assert caps.scratch.isolation == "restored"
        assert caps.supports("scratch.live_untouched") == "partial"
        assert caps.supports("scratch.set_root_state") == "synthetic"


def test_physx_contact_manifold_detail_stays_partial() -> None:
    """PhysX keeps the P1C-3 partial manifold without normal/friction/geom ids."""
    manifold = _backend_capabilities()["isaaclab3_physx"].contacts.manifold
    detail = manifold.detail.lower()

    assert manifold.support == "partial"
    assert "normal" in detail
    assert "friction" in detail
    assert "geom" in detail


def test_newton_contact_gap_is_not_inferred_from_physx() -> None:
    """PhysX's partial manifold never promotes the Newton contact claim."""
    caps = _backend_capabilities()

    assert caps["isaaclab3_physx"].supports("contacts.manifold") == "partial"
    assert caps["isaaclab3_newton"].supports("contacts.manifold") == "unsupported"
    assert "Newton" in caps["isaaclab3_newton"].contacts.manifold.detail


def test_query_capabilities_add_no_per_operation_traits() -> None:
    """Query stays one trait plus one capability catalog, with no gap operations."""
    query_traits = [trait.name for trait in Trait if "QUERY" in trait.name]
    trait_names = {trait.name for trait in Trait}
    capability_fields = {field.name for field in fields(BackendCapabilities)}

    assert query_traits == ["RUNTIME_QUERY"]
    assert not any("query" in name.lower() for name in capability_fields)

    catalogs = {name: caps.ops() for name, caps in _backend_capabilities().items()}
    reference = catalogs["mujoco"]
    for op in reference:
        group, _, member = op.partition(".")
        assert group and member, op
        assert op not in trait_names, op

    for name, ops in catalogs.items():
        assert ops == reference, name
        assert not any("sparse" in op or "mesh" in op or "gap" in op for op in ops), name
