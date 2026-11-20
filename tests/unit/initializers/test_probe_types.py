"""Neutral probe request types: field validation and canonical values."""

from __future__ import annotations

import pytest

from unigym.initializers import (
    CollisionPairSpec,
    ProbeCapabilityError,
    ProbePose,
    ProbeRequest,
    ProbeResult,
)


def test_probe_capability_error_is_a_neutral_runtime_error() -> None:
    """Packs catch capability failures through ``unigym.initializers`` only."""
    assert issubclass(ProbeCapabilityError, RuntimeError)


def _pose() -> ProbePose:
    return ProbePose(
        owner="crate",
        body=None,
        pos_w=(0.1, 0.2, 0.3),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def test_probe_pose_keeps_canonical_tuple_values() -> None:
    """A valid pose stores finite tuples and preserves the body scope."""
    pose = ProbePose(
        owner="crate",
        body="base",
        pos_w=(0.1, 0.2, 0.3),
        quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )

    assert pose.pos_w == (0.1, 0.2, 0.3)
    assert pose.quat_wxyz == (0.0, 1.0, 0.0, 0.0)
    assert pose.body == "base"
    assert all(isinstance(component, float) for component in pose.pos_w)


@pytest.mark.parametrize("owner", ["", "  "])
def test_probe_pose_rejects_an_empty_owner(owner: str) -> None:
    """The owner is a non-empty declared name."""
    with pytest.raises(ValueError, match="ProbePose.owner"):
        ProbePose(owner=owner, body=None, pos_w=(0.0, 0.0, 0.0), quat_wxyz=(1, 0, 0, 0))


def test_probe_pose_rejects_an_empty_body_scope() -> None:
    """A body scope is either None or a non-empty name."""
    with pytest.raises(ValueError, match="ProbePose.body"):
        ProbePose(owner="crate", body="", pos_w=(0.0, 0.0, 0.0), quat_wxyz=(1, 0, 0, 0))


def test_probe_names_are_stripped() -> None:
    """Owner/body/geom names are canonicalized without surrounding blanks."""
    pose = ProbePose(
        owner=" crate ",
        body=" base ",
        pos_w=(0.0, 0.0, 0.0),
        quat_wxyz=(1, 0, 0, 0),
    )
    pair = CollisionPairSpec(" crate_geom ", " floor ")

    assert pose.owner == "crate"
    assert pose.body == "base"
    assert pair.candidate_geom == "crate_geom"
    assert pair.other_geom == "floor"


@pytest.mark.parametrize(
    ("pos_w", "match"),
    [
        ((0.0, 0.0), r"ProbePose.pos_w must have shape \(3,\)"),
        ((0.0, 0.0, float("nan")), "ProbePose.pos_w must be finite"),
    ],
)
def test_probe_pose_rejects_an_invalid_position(pos_w, match: str) -> None:
    """Position shape and finiteness are checked at construction."""
    with pytest.raises(ValueError, match=match):
        ProbePose(owner="crate", body=None, pos_w=pos_w, quat_wxyz=(1, 0, 0, 0))


@pytest.mark.parametrize(
    ("quat_wxyz", "match"),
    [
        ((0.0, 0.0, 0.0), r"ProbePose.quat_wxyz must have shape \(4,\)"),
        ((0.0, 0.0, 0.0, 0.0), "ProbePose.quat_wxyz must be a non-zero"),
        ((1.0, float("inf"), 0.0, 0.0), "ProbePose.quat_wxyz must be finite"),
    ],
)
def test_probe_pose_rejects_an_invalid_quaternion(quat_wxyz, match: str) -> None:
    """The world-frame wxyz quaternion must be finite and non-zero."""
    with pytest.raises(ValueError, match=match):
        ProbePose(owner="crate", body=None, pos_w=(0.0, 0.0, 0.0), quat_wxyz=quat_wxyz)


def test_collision_pair_spec_defaults_and_validation() -> None:
    """The pair defaults are the documented tolerance and no support exception."""
    pair = CollisionPairSpec(candidate_geom="crate_geom", other_geom="floor")

    assert pair.allow_support_contact is False
    assert pair.penetration_tolerance_m == pytest.approx(1e-7)


@pytest.mark.parametrize(
    ("candidate_geom", "other_geom", "match"),
    [
        ("", "floor", "CollisionPairSpec.candidate_geom"),
        ("crate_geom", " ", "CollisionPairSpec.other_geom"),
    ],
)
def test_collision_pair_spec_rejects_empty_geom_names(
    candidate_geom: str, other_geom: str, match: str
) -> None:
    """Both geom references are non-empty names."""
    with pytest.raises(ValueError, match=match):
        CollisionPairSpec(candidate_geom=candidate_geom, other_geom=other_geom)


@pytest.mark.parametrize("tolerance", [-1e-3, float("inf"), float("nan")])
def test_collision_pair_spec_rejects_an_invalid_tolerance(tolerance) -> None:
    """The penetration tolerance is finite and non-negative."""
    with pytest.raises(ValueError, match="penetration_tolerance_m"):
        CollisionPairSpec(
            candidate_geom="crate_geom",
            other_geom="floor",
            penetration_tolerance_m=tolerance,
        )


@pytest.mark.parametrize("flag", [1])
def test_collision_pair_spec_rejects_a_non_bool_support_flag(flag) -> None:
    """The support-contact flag is a real bool, not truthy filler."""
    with pytest.raises(TypeError, match="allow_support_contact must be bool"):
        CollisionPairSpec(
            candidate_geom="crate_geom", other_geom="floor", allow_support_contact=flag
        )


def test_probe_request_accepts_minimal_and_full_shapes() -> None:
    """Defaults are no explicit pairs, zero clearance and no support body."""
    request = ProbeRequest(overrides=(_pose(),))

    assert request.overrides == (_pose(),)
    assert request.pairs is None
    assert request.min_clearance_m == 0.0
    assert request.support_body is None

    pair = CollisionPairSpec("crate_geom", "floor")
    full = ProbeRequest(
        overrides=(_pose(),),
        pairs=(pair,),
        min_clearance_m=0.02,
        support_body="floor",
    )

    assert full.pairs == (pair,)
    assert full.min_clearance_m == pytest.approx(0.02)
    assert full.support_body == "floor"


def test_probe_request_accepts_an_explicit_empty_pair_set() -> None:
    """An explicit empty pair set is a legal degenerate request."""
    request = ProbeRequest(overrides=(_pose(),), pairs=())

    assert request.pairs == ()


def test_probe_request_rejects_empty_overrides() -> None:
    """At least one candidate free-root pose is required."""
    with pytest.raises(ValueError, match="overrides must be a non-empty"):
        ProbeRequest(overrides=())


@pytest.mark.parametrize("wrong", [object()])
def test_probe_request_rejects_a_non_pose_override(wrong) -> None:
    """Every override is a ProbePose, named by its index."""
    with pytest.raises(TypeError, match=r"overrides\[0\] must be a ProbePose"):
        ProbeRequest(overrides=(wrong,))


def test_probe_request_rejects_duplicate_owner_body_overrides() -> None:
    """Two overrides for the same free root are ambiguous."""
    with pytest.raises(ValueError, match=r"overrides\[1\] duplicates owner/body"):
        ProbeRequest(overrides=(_pose(), _pose()))


@pytest.mark.parametrize("wrong", [object()])
def test_probe_request_rejects_a_non_pair_pairs_entry(wrong) -> None:
    """Every explicit pair is a CollisionPairSpec, named by its index."""
    with pytest.raises(TypeError, match=r"pairs\[0\] must be a CollisionPairSpec"):
        ProbeRequest(overrides=(_pose(),), pairs=(wrong,))


@pytest.mark.parametrize("clearance", [-0.1, float("nan"), float("inf")])
def test_probe_request_rejects_an_invalid_clearance(clearance) -> None:
    """Surface clearance is finite and non-negative."""
    with pytest.raises(ValueError, match="min_clearance_m"):
        ProbeRequest(overrides=(_pose(),), min_clearance_m=clearance)


def test_probe_request_rejects_an_empty_support_body() -> None:
    """An ignored support body is either None or a non-empty name."""
    with pytest.raises(ValueError, match="ProbeRequest.support_body"):
        ProbeRequest(overrides=(_pose(),), support_body="")


def test_probe_result_keeps_canonical_distances() -> None:
    """Distances are floats in a tuple and the reason stays optional."""
    result = ProbeResult(accepted=False, reason="penetration", distances_m=(-0.1, 0.2))

    assert result.accepted is False
    assert result.reason == "penetration"
    assert result.distances_m == (-0.1, 0.2)
    assert all(isinstance(distance, float) for distance in result.distances_m)


@pytest.mark.parametrize("verdict", [1])
def test_probe_result_rejects_a_non_bool_verdict(verdict) -> None:
    """A malformed backend verdict fails loudly, not silently accepted."""
    with pytest.raises(TypeError, match="accepted must be bool"):
        ProbeResult(accepted=verdict, reason=None, distances_m=())


@pytest.mark.parametrize("reason", [3])
def test_probe_result_rejects_a_non_string_reason(reason) -> None:
    """A malformed reason is a result contract error."""
    with pytest.raises(TypeError, match="reason must be str or None"):
        ProbeResult(accepted=True, reason=reason, distances_m=())


def test_probe_result_rejects_a_non_finite_distance() -> None:
    """Distances are finite metres."""
    with pytest.raises(ValueError, match="distances_m must be finite"):
        ProbeResult(accepted=True, reason=None, distances_m=(float("nan"),))


def test_probe_result_rejects_a_reason_on_an_accepted_probe() -> None:
    """An accepted verdict has no failure reason."""
    with pytest.raises(ValueError, match="reason must be None when accepted"):
        ProbeResult(accepted=True, reason="penetration", distances_m=())


def test_probe_result_requires_a_reason_on_a_rejected_probe() -> None:
    """A rejected verdict names the failure reason."""
    with pytest.raises(ValueError, match="reason must be set when rejected"):
        ProbeResult(accepted=False, reason=None, distances_m=())
