"""Exact P2 capability matrix across the five supported backend families.

This is a release gate, not a second implementation test: it reads the
capability declarations only.  No env is constructed and no engine is started;
the backend modules are imported inside a module-scoped fixture so collection
never pulls a backend runtime.

Every ``<group>.<operation>`` must report exactly the level recorded in
``_MATRIX``.  A changed capability must change this table in the same review,
which keeps the docs/contract matrix and the code from drifting apart.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from unigym.backends.registry import Trait
from unigym.env.backends.base import BackendCapabilities
from unigym.query.capabilities import QueryCapabilities
from unigym.query.errors import QueryCapabilityError

BACKENDS = (
    "mujoco",
    "mjwarp",
    "isaaclab2",
    "isaaclab3_physx",
    "isaaclab3_newton",
)

_GROUPS = (
    "model",
    "state",
    "scratch",
    "jacobian",
    "geometry",
    "contacts",
    "actuators",
)

_STATE_MEMBERS = (
    "body_pose",
    "site_pose",
    "com_pose",
    "root_pose",
    "joint_position",
    "joint_velocity",
    "root_state",
    "com_state",
    "object_state",
    "object_com_state",
)

_JACOBIAN_MEMBERS = ("body", "site", "com", "point", "object")

_OBJECT_OPS = (
    "state.root_state",
    "state.com_state",
    "state.object_state",
    "state.object_com_state",
    "jacobian.object",
)

# Exact declared support per backend and operation.  The five new P2 object
# operations are the release claim under test; the remaining rows pin the P1
# matrix so a P2 change cannot silently move an existing support level.
_MATRIX: dict[str, dict[str, str]] = {
    "mujoco": {
        "model.ir": "native",
        "model.body": "native",
        "model.geom": "native",
        "model.site": "native",
        "model.joint": "native",
        "model.actuator": "native",
        "model.geom_contact": "native",
        "state.body_pose": "native",
        "state.site_pose": "native",
        "state.com_pose": "native",
        "state.root_pose": "native",
        "state.joint_position": "native",
        "state.joint_velocity": "native",
        "state.root_state": "native",
        "state.com_state": "native",
        "state.object_state": "native",
        "state.object_com_state": "native",
        "scratch.copy": "native",
        "scratch.live_untouched": "native",
        "scratch.set_root_state": "native",
        "scratch.set_joint_state": "native",
        "scratch.forward": "native",
        "scratch.body_pose": "native",
        "scratch.site_pose": "native",
        "scratch.com_pose": "native",
        "scratch.geometry": "native",
        "scratch.jacobian": "native",
        "scratch.contacts": "native",
        "jacobian.body": "native",
        "jacobian.site": "native",
        "jacobian.com": "partial",
        "jacobian.point": "native",
        "jacobian.object": "native",
        "geometry.distance": "native",
        "geometry.aabb": "native",
        "contacts.manifold": "native",
        "contacts.jacobian": "native",
        "contacts.wrench_to_joint": "native",
        "actuators.facts": "native",
        "actuators.state": "native",
    },
    "mjwarp": {
        "model.ir": "native",
        "model.body": "native",
        "model.geom": "native",
        "model.site": "native",
        "model.joint": "native",
        "model.actuator": "native",
        "model.geom_contact": "native",
        "state.body_pose": "native",
        "state.site_pose": "native",
        "state.com_pose": "native",
        "state.root_pose": "native",
        "state.joint_position": "native",
        "state.joint_velocity": "native",
        "state.root_state": "native",
        "state.com_state": "native",
        "state.object_state": "native",
        "state.object_com_state": "native",
        "scratch.copy": "synthetic",
        "scratch.live_untouched": "synthetic",
        "scratch.set_root_state": "synthetic",
        "scratch.set_joint_state": "synthetic",
        "scratch.forward": "synthetic",
        "scratch.body_pose": "synthetic",
        "scratch.site_pose": "synthetic",
        "scratch.com_pose": "synthetic",
        "scratch.geometry": "synthetic",
        "scratch.jacobian": "synthetic",
        "scratch.contacts": "synthetic",
        "jacobian.body": "synthetic",
        "jacobian.site": "synthetic",
        "jacobian.com": "partial",
        "jacobian.point": "synthetic",
        "jacobian.object": "synthetic",
        "geometry.distance": "synthetic",
        "geometry.aabb": "synthetic",
        "contacts.manifold": "synthetic",
        "contacts.jacobian": "synthetic",
        "contacts.wrench_to_joint": "synthetic",
        "actuators.facts": "synthetic",
        "actuators.state": "synthetic",
    },
    "isaaclab2": {
        "model.ir": "native",
        "model.body": "native",
        "model.geom": "unsupported",
        "model.site": "unsupported",
        "model.joint": "native",
        "model.actuator": "native",
        "model.geom_contact": "unsupported",
        "state.body_pose": "native",
        "state.site_pose": "partial",
        "state.com_pose": "partial",
        "state.root_pose": "native",
        "state.joint_position": "native",
        "state.joint_velocity": "native",
        "state.root_state": "unsupported",
        "state.com_state": "unsupported",
        "state.object_state": "unsupported",
        "state.object_com_state": "unsupported",
        "scratch.copy": "unsupported",
        "scratch.live_untouched": "unsupported",
        "scratch.set_root_state": "unsupported",
        "scratch.set_joint_state": "unsupported",
        "scratch.forward": "unsupported",
        "scratch.body_pose": "unsupported",
        "scratch.site_pose": "unsupported",
        "scratch.com_pose": "unsupported",
        "scratch.geometry": "unsupported",
        "scratch.jacobian": "unsupported",
        "scratch.contacts": "unsupported",
        "jacobian.body": "unsupported",
        "jacobian.site": "unsupported",
        "jacobian.com": "unsupported",
        "jacobian.point": "unsupported",
        "jacobian.object": "unsupported",
        "geometry.distance": "unsupported",
        "geometry.aabb": "unsupported",
        "contacts.manifold": "unsupported",
        "contacts.jacobian": "unsupported",
        "contacts.wrench_to_joint": "unsupported",
        "actuators.facts": "partial",
        "actuators.state": "unsupported",
    },
    "isaaclab3_physx": {
        "model.ir": "native",
        "model.body": "native",
        "model.geom": "unsupported",
        "model.site": "unsupported",
        "model.joint": "native",
        "model.actuator": "native",
        "model.geom_contact": "unsupported",
        "state.body_pose": "native",
        "state.site_pose": "partial",
        "state.com_pose": "partial",
        "state.root_pose": "native",
        "state.joint_position": "native",
        "state.joint_velocity": "native",
        "state.root_state": "unsupported",
        "state.com_state": "unsupported",
        "state.object_state": "unsupported",
        "state.object_com_state": "unsupported",
        "scratch.copy": "synthetic",
        "scratch.live_untouched": "partial",
        "scratch.set_root_state": "synthetic",
        "scratch.set_joint_state": "synthetic",
        "scratch.forward": "synthetic",
        "scratch.body_pose": "synthetic",
        "scratch.site_pose": "synthetic",
        "scratch.com_pose": "synthetic",
        "scratch.geometry": "unsupported",
        "scratch.jacobian": "synthetic",
        "scratch.contacts": "unsupported",
        "jacobian.body": "native",
        "jacobian.site": "synthetic",
        "jacobian.com": "partial",
        "jacobian.point": "synthetic",
        "jacobian.object": "unsupported",
        "geometry.distance": "unsupported",
        "geometry.aabb": "unsupported",
        "contacts.manifold": "partial",
        "contacts.jacobian": "unsupported",
        "contacts.wrench_to_joint": "unsupported",
        "actuators.facts": "partial",
        "actuators.state": "unsupported",
    },
    "isaaclab3_newton": {
        "model.ir": "native",
        "model.body": "native",
        "model.geom": "unsupported",
        "model.site": "unsupported",
        "model.joint": "native",
        "model.actuator": "native",
        "model.geom_contact": "unsupported",
        "state.body_pose": "native",
        "state.site_pose": "partial",
        "state.com_pose": "partial",
        "state.root_pose": "native",
        "state.joint_position": "native",
        "state.joint_velocity": "native",
        "state.root_state": "unsupported",
        "state.com_state": "unsupported",
        "state.object_state": "unsupported",
        "state.object_com_state": "unsupported",
        "scratch.copy": "synthetic",
        "scratch.live_untouched": "partial",
        "scratch.set_root_state": "synthetic",
        "scratch.set_joint_state": "synthetic",
        "scratch.forward": "synthetic",
        "scratch.body_pose": "synthetic",
        "scratch.site_pose": "synthetic",
        "scratch.com_pose": "synthetic",
        "scratch.geometry": "unsupported",
        "scratch.jacobian": "synthetic",
        "scratch.contacts": "unsupported",
        "jacobian.body": "native",
        "jacobian.site": "synthetic",
        "jacobian.com": "partial",
        "jacobian.point": "synthetic",
        "jacobian.object": "unsupported",
        "geometry.distance": "unsupported",
        "geometry.aabb": "unsupported",
        "contacts.manifold": "unsupported",
        "contacts.jacobian": "unsupported",
        "contacts.wrench_to_joint": "unsupported",
        "actuators.facts": "partial",
        "actuators.state": "unsupported",
    },
}


@pytest.fixture(scope="module")
def capabilities() -> dict[str, QueryCapabilities]:
    """Import and return the five capability declarations without an engine."""
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


def _feature(caps: QueryCapabilities, op: str):
    """Return one declared ``Feature`` by ``<group>.<operation>``."""
    group, _, member = op.partition(".")
    return getattr(getattr(caps, group), member)


def test_matrix_rows_cover_the_same_complete_operation_set() -> None:
    """The fixture matrix is complete: five backends, the same 40 operations."""
    assert tuple(_MATRIX) == BACKENDS
    reference = set(_MATRIX["mujoco"])
    assert len(reference) == 40
    for backend in BACKENDS:
        assert set(_MATRIX[backend]) == reference, backend
    assert _MATRIX["mujoco"]["state.object_state"] == "native"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", sorted(_MATRIX["mujoco"]))
def test_capability_matrix_is_exact(
    capabilities: dict[str, QueryCapabilities],
    backend: str,
    op: str,
) -> None:
    """Every backend reports exactly the matrix level for every operation."""
    expected = _MATRIX[backend][op]
    actual = capabilities[backend].supports(op)
    assert actual == expected, f"{backend} {op}: expected {expected!r}, got {actual!r}"


def test_mujoco_declares_native_object_state_and_object_jacobian(
    capabilities: dict[str, QueryCapabilities],
) -> None:
    """MuJoCo answers every new object read natively, including the Jacobian."""
    caps = capabilities["mujoco"]
    for op in _OBJECT_OPS:
        assert caps.supports(op) == "native", op


def test_mjwarp_declares_native_object_state_and_synthetic_object_jacobian(
    capabilities: dict[str, QueryCapabilities],
) -> None:
    """mjWarp reads state from the CPU mirror and computes the Jacobian on it."""
    caps = capabilities["mjwarp"]
    for op in (
        "state.root_state",
        "state.com_state",
        "state.object_state",
        "state.object_com_state",
    ):
        assert caps.supports(op) == "native", op
    assert caps.supports("jacobian.object") == "synthetic"


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_isaaclab_declares_object_state_and_object_jacobian_unsupported(
    capabilities: dict[str, QueryCapabilities],
    backend: str,
) -> None:
    """No IsaacLab identity claims an object state, COM state or object Jacobian."""
    caps = capabilities[backend]
    for op in _OBJECT_OPS:
        assert caps.supports(op) == "unsupported", f"{backend} {op}"


def test_physx_manifold_is_partial_and_newton_manifold_is_unsupported(
    capabilities: dict[str, QueryCapabilities],
) -> None:
    """PhysX keeps the P1C-3 partial manifold; Newton is never inferred from it."""
    assert capabilities["isaaclab3_physx"].supports("contacts.manifold") == "partial"
    assert capabilities["isaaclab3_newton"].supports("contacts.manifold") == "unsupported"


@pytest.mark.parametrize(
    ("backend", "mode", "isolation"),
    (
        ("mujoco", "copy", "bit_exact"),
        ("mjwarp", "copy", "bit_exact"),
        ("isaaclab2", "copy", "bit_exact"),
        ("isaaclab3_physx", "transaction", "restored"),
        ("isaaclab3_newton", "transaction", "restored"),
    ),
)
def test_scratch_mode_and_isolation_match_the_declaration(
    capabilities: dict[str, QueryCapabilities],
    backend: str,
    mode: str,
    isolation: str,
) -> None:
    """The MuJoCo family is a bit-exact copy; IsaacLab 3 is a restored transaction."""
    scratch = capabilities[backend].scratch
    assert (scratch.mode, scratch.isolation) == (mode, isolation)


def test_object_operations_have_a_teaching_detail(
    capabilities: dict[str, QueryCapabilities],
) -> None:
    """Every declared object operation carries a non-empty teaching detail."""
    for backend, caps in capabilities.items():
        for op in _OBJECT_OPS:
            assert _feature(caps, op).detail.strip(), f"{backend} {op}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ops_catalog_is_the_exact_matrix_without_duplicates(
    capabilities: dict[str, QueryCapabilities],
    backend: str,
) -> None:
    """``ops()`` is exactly the matrix, with no duplicate or extra operation."""
    ops = capabilities[backend].ops()
    assert len(ops) == len(set(ops)), backend
    assert set(ops) == set(_MATRIX[backend]), backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_state_and_jacobian_members_are_exactly_the_contract(
    capabilities: dict[str, QueryCapabilities],
    backend: str,
) -> None:
    """No unknown ``state``/``jacobian`` member can appear without a matrix change."""
    ops = capabilities[backend].ops()
    state_members = tuple(op.partition(".")[2] for op in ops if op.startswith("state."))
    jacobian_members = tuple(op.partition(".")[2] for op in ops if op.startswith("jacobian."))

    assert state_members == _STATE_MEMBERS
    assert jacobian_members == _JACOBIAN_MEMBERS


@pytest.mark.parametrize(
    "op",
    (
        "state.bogus",
        "state.object_twist",
        "state.object_jacobian",
        "jacobian.bogus",
        "jacobian.geom",
        "jacobian.object_com",
    ),
)
def test_unknown_state_and_jacobian_members_are_rejected(
    capabilities: dict[str, QueryCapabilities],
    op: str,
) -> None:
    """An unknown subface member is a capability error, never a silent ``unsupported``."""
    for backend, caps in capabilities.items():
        with pytest.raises(QueryCapabilityError):
            caps.supports(op)


def test_query_adds_no_per_operation_trait_or_backend_capability_bit(
    capabilities: dict[str, QueryCapabilities],
) -> None:
    """Query stays one trait plus one catalog; no operation becomes a trait/bit."""
    query_traits = [trait.name for trait in Trait if "QUERY" in trait.name]
    trait_names = {trait.name for trait in Trait}
    capability_fields = {field.name for field in fields(BackendCapabilities)}

    assert query_traits == ["RUNTIME_QUERY"]
    assert not any("query" in name.lower() for name in capability_fields)

    for backend, caps in capabilities.items():
        for op in caps.ops():
            group, dot, member = op.partition(".")
            assert dot and member, f"{backend} {op}"
            assert group in _GROUPS, f"{backend} {op}"
            assert op not in trait_names, f"{backend} {op}"
        assert not any("sparse" in op or "mesh" in op or "gap" in op for op in caps.ops()), backend
