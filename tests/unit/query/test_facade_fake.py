"""Protocol-shape checks for the runtime-query facade.

The sample below is a real, tiny implementation of the facade protocols (the
unit layer must not start an engine).  It proves the runtime-checkable facade
shape and the capability gate, not backend physics.
"""

from __future__ import annotations

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
from unigym.query.facade import Query


def _feature() -> Feature:
    return Feature("native")


def _capabilities() -> QueryCapabilities:
    native = _feature()
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
        ),
        actuators=ActuatorCaps(
            facts=Feature("unsupported", "arrives in P1"),
            state=Feature("unsupported", "arrives in P1"),
        ),
    )


class _SampleSubface:
    """A placeholder subface with no behavior (shape check only)."""


class _SampleQuery:
    """A minimal object carrying exactly the facade member set."""

    def __init__(self) -> None:
        self.backend = "sample"
        self.capabilities = _capabilities()
        self.model = _SampleSubface()
        self.state = _SampleSubface()
        self.kinematics = _SampleSubface()
        self.geometry = _SampleSubface()
        self.contacts = _SampleSubface()
        self.actuators = _SampleSubface()

    def scratch(self, env: int | None = None) -> None:
        del env
        raise NotImplementedError

    def require(self, *ops: str) -> None:
        self.capabilities.require(*ops)

    def close(self) -> None:
        pass


class _IncompleteQuery:
    """A facade missing every subface (must be rejected by the protocol)."""

    def __init__(self) -> None:
        self.backend = "sample"
        self.capabilities = _capabilities()

    def scratch(self, env: int | None = None) -> None:
        del env
        raise NotImplementedError

    def require(self, *ops: str) -> None:
        self.capabilities.require(*ops)

    def close(self) -> None:
        pass


def test_query_protocol_recognizes_the_complete_facade_shape() -> None:
    """The runtime-checkable facade requires every subface attribute."""
    assert isinstance(_SampleQuery(), Query)


def test_query_protocol_rejects_a_facade_missing_subfaces() -> None:
    """A backend that omits a subface is not a Query."""
    assert not isinstance(_IncompleteQuery(), Query)


def test_sample_facade_delegates_require_to_capabilities() -> None:
    """The facade gate is the capability object's gate."""
    with pytest.raises(QueryCapabilityError, match="contacts.manifold"):
        _SampleQuery().require("contacts.manifold")
