"""Runtime-query capability support levels and gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from unigym.query.capabilities import (
    ActuatorCaps,
    ContactCaps,
    Feature,
    GeometryCaps,
    JacobianCaps,
    ModelCaps,
    QueryCapabilities,
    ScratchCaps,
    StateCaps,
)
from unigym.query.errors import QueryCapabilityError


def _native() -> Feature:
    return Feature("native")


def _capabilities() -> QueryCapabilities:
    native = _native()
    return QueryCapabilities(
        api_version=1,
        backend="sample",
        model=ModelCaps(
            ir=native,
            body=native,
            geom=native,
            site=native,
            joint=native,
            actuator=native,
        ),
        state=StateCaps(
            body_pose=native,
            site_pose=native,
            com_pose=native,
            root_pose=native,
            joint_position=native,
            joint_velocity=native,
        ),
        scratch=ScratchCaps(
            copy=native,
            live_untouched=native,
            set_root_state=native,
            set_joint_state=native,
            forward=native,
            body_pose=native,
            site_pose=native,
            com_pose=native,
            geometry=native,
            jacobian=native,
        ),
        jacobian=JacobianCaps(body=native, site=native, com=native, point=native),
        geometry=GeometryCaps(distance=native, aabb=native),
        contacts=ContactCaps(
            manifold=Feature("unsupported", "arrives in P1"),
            jacobian=Feature("unsupported", "arrives in P1"),
            wrench_to_joint=Feature("unsupported", "arrives in P1"),
        ),
        actuators=ActuatorCaps(
            facts=Feature("unsupported", "arrives in P1"),
            state=Feature("unsupported", "arrives in P1"),
        ),
        notes={"engine_thread": "only between steps"},
    )


def test_supports_returns_the_declared_level() -> None:
    """A native and an unsupported operation report their declared support."""
    caps = _capabilities()

    assert caps.supports("model.body") == "native"
    assert caps.supports("contacts.manifold") == "unsupported"


def test_require_passes_for_supported_operations() -> None:
    """Requiring only supported operations never raises."""
    _capabilities().require("model.ir", "geometry.distance", "jacobian.body")


def test_require_names_the_backend_operation_and_support() -> None:
    """An unsupported operation fails with a teaching capability error."""
    with pytest.raises(QueryCapabilityError) as caught:
        _capabilities().require("contacts.manifold")

    error = caught.value
    assert error.backend == "sample"
    assert error.op == "contacts.manifold"
    assert error.support == "unsupported"
    assert "contacts.manifold" in str(error)
    assert "arrives in P1" in str(error)


def test_require_reports_the_first_unsupported_operation() -> None:
    """Mixed requirements name the first unsupported op and list the rest."""
    with pytest.raises(QueryCapabilityError) as caught:
        _capabilities().require("model.ir", "actuators.state", "contacts.jacobian")

    assert caught.value.op == "actuators.state"
    assert "contacts.jacobian" in str(caught.value)


@pytest.mark.parametrize("op", ["model.bogus", "bogus.body", "model"])
def test_unknown_operation_is_a_capability_error(op: str) -> None:
    """An unknown group or member fails loudly instead of defaulting."""
    with pytest.raises(QueryCapabilityError, match=r"cannot query"):
        _capabilities().supports(op)


def test_feature_rejects_an_unknown_support_level() -> None:
    """Feature support is a closed vocabulary."""
    with pytest.raises(ValueError, match="Feature.support"):
        Feature("bogus")  # type: ignore[arg-type]


def test_scratch_caps_mode_and_isolation_are_a_closed_vocabulary() -> None:
    """A scratch copy defaults to bit-exact copy and rejects unknown modes."""
    scratch = _capabilities().scratch

    assert scratch.mode == "copy"
    assert scratch.isolation == "bit_exact"
    with pytest.raises(ValueError, match="ScratchCaps.mode"):
        replace(scratch, mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ScratchCaps.isolation"):
        replace(scratch, isolation="bogus")  # type: ignore[arg-type]


def test_capability_notes_are_read_only() -> None:
    """Callers cannot mutate the shared note table."""
    caps = _capabilities()

    with pytest.raises(TypeError):
        caps.notes["engine_thread"] = "changed"  # type: ignore[index]


def test_ops_enumerates_every_group_member() -> None:
    """The operation catalog covers each declared subface."""
    ops = _capabilities().ops()

    assert "model.ir" in ops
    assert "model.geom_contact" in ops
    assert "state.root_pose" in ops
    assert "scratch.forward" in ops
    assert "jacobian.point" in ops
    assert "geometry.aabb" in ops
    assert "contacts.jacobian" in ops
    assert "actuators.facts" in ops


def test_object_members_default_to_unsupported_not_declared() -> None:
    """An existing constructor keeps working; new members stay undeclared."""
    caps = _capabilities()
    object_ops = (
        "state.root_state",
        "state.com_state",
        "state.object_state",
        "state.object_com_state",
        "jacobian.object",
    )

    for op in object_ops:
        assert caps.supports(op) == "unsupported"
        assert op in caps.ops()
    with pytest.raises(QueryCapabilityError) as caught:
        caps.require("state.object_state")
    assert caught.value.detail == "not declared"


def test_geom_contact_member_defaults_to_unsupported_not_declared() -> None:
    """An existing constructor keeps working; geom_contact stays undeclared."""
    caps = _capabilities()

    assert caps.supports("model.geom_contact") == "unsupported"
    with pytest.raises(QueryCapabilityError) as caught:
        caps.require("model.geom_contact")
    assert caught.value.detail == "not declared"

    declared = replace(caps, model=replace(caps.model, geom_contact=Feature("native")))
    assert declared.supports("model.geom_contact") == "native"


def test_object_members_can_be_declared_explicitly() -> None:
    """A backend may declare the new members without touching the old ones."""
    caps = _capabilities()
    declared = replace(
        caps,
        state=replace(caps.state, object_state=Feature("native")),
        jacobian=replace(caps.jacobian, object=Feature("synthetic")),
    )

    assert declared.supports("state.object_state") == "native"
    assert declared.supports("jacobian.object") == "synthetic"
    assert declared.supports("state.object_com_state") == "unsupported"
    assert declared.supports("state.root_pose") == "native"
