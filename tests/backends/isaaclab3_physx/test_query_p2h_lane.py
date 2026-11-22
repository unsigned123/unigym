"""Real IsaacLab 3 PhysX typed-unsupported P2h query surfaces (P2h lane evidence)."""

from __future__ import annotations

import pytest

from unigym.query import ObjectRef, QueryCapabilityError

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]


def test_new_live_query_surfaces_raise_typed_capability_errors(cartpole_env) -> None:
    """Each P2h live surface is typed-unsupported, never an AttributeError."""
    query = cartpole_env.query

    with pytest.raises(QueryCapabilityError) as geom_contact:
        query.model.geom_contact("crate")
    assert geom_contact.value.op == "model.geom_contact"

    with pytest.raises(QueryCapabilityError) as all_pairs:
        query.geometry.all_pairs()
    assert all_pairs.value.op == "geometry.all_pairs"

    with pytest.raises(QueryCapabilityError) as scan:
        query.contacts.scan()
    assert scan.value.op == "contacts.scan"

    # The capability gate runs before object resolution, so an arbitrary name is
    # never resolved and cannot raise a reference error instead.
    with pytest.raises(QueryCapabilityError) as object_state:
        query.state.object_state(ObjectRef("crate"))
    assert object_state.value.op == "state.object_state"


def test_scratch_contacts_and_new_capabilities_are_typed_unsupported(cartpole_env) -> None:
    """Scratch contacts reject with ``scratch.contacts`` and the matrix is unsupported."""
    query = cartpole_env.query

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError) as contacts:
            scratch.contacts()
    assert contacts.value.op == "scratch.contacts"

    for op in (
        "model.geom_contact",
        "geometry.all_pairs",
        "contacts.scan",
        "scratch.contacts",
    ):
        assert query.capabilities.supports(op) == "unsupported"
