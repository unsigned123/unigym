"""Control runtime construction: validation, derivation and coverage."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from unigym.actuator_law import (
    ActuatorLawDescriptor,
    register_actuator_law,
    restore_actuator_laws,
    snapshot_actuator_laws,
)
from unigym.actuators.ir import ActuationGroup, resolve_actuation_groups
from unigym.registration import reset_registration
from unigym.robots.base import RobotProfile
from unigym.runtime.actions import (
    OrderedActionGroup,
    OrderedActionSpec,
    PolicyBinding,
    resolve_control_dt,
)
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
    build_control_runtimes,
    drive_for_target_laws,
    validate_action_coverage,
    validate_policy_bindings,
)
from unigym.runtime.controls import ControlError
from unigym.scenes.articulation import ArticulationSpec, JointSpec


@pytest.fixture(autouse=True)
def _isolate_actuator_laws():
    """Restore the actuator-law registry so no test law leaks."""
    laws = snapshot_actuator_laws()
    reset_registration()
    yield
    restore_actuator_laws(laws)
    reset_registration()


def _group(
    name="body",
    articulation="arm",
    joints=("shoulder", "elbow"),
    target="position",
    role="aux",
):
    return OrderedActionGroup(
        name=name,
        articulation=articulation,
        joints=joints,
        target=target,
        role=role,
    )


def _binding(adapter="manual", groups=(("arm", "body"),), control_dt=None):
    return PolicyBinding(adapter=adapter, groups=groups, control_dt=control_dt)


def _plans(group, art, *, law="standard_pd"):
    gains = {"kp": 1.0, "kd": 0.1} if law in ("standard_pd", "computed_torque_pd") else {}
    actuation = ActuationGroup(
        name="act",
        law=law,
        joints=tuple(group.joints),
        gains=gains,
    )
    return {
        art.name: resolve_actuation_groups((actuation,), articulation=art.name, joints=group.joints)
    }


def _build(groups, art, profile, *, policies=None, law="standard_pd"):
    groups = tuple(groups)
    if policies is None:
        policies = (_binding(groups=tuple((g.articulation, g.name) for g in groups)),)
    plans = _plans(groups[0], art, law=law) if groups else {}
    return build_control_runtimes(
        OrderedActionSpec(groups),
        tuple(policies),
        (art,),
        {"arm": profile},
        plans,
    )


def _profile(joint_sets, defaults):
    return RobotProfile(
        name="arm",
        joint_sets=joint_sets,
        default_joint_pos=defaults,
        assets=None,
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _runtime(name: str, articulation: str, joints: tuple[str, ...], role="aux") -> ControlRuntime:
    return ControlRuntime(
        name=name,
        adapter="manual",
        articulation=articulation,
        joint_names=joints,
        q0=(0.0,) * len(joints),
        role=role,
    )


@pytest.mark.parametrize(
    ("drive", "target", "match"),
    [
        ("bogus", "position", "drive must be one of"),
        ("torque", "position", "requires target 'torque'"),
        ("position_pd", "velocity", "requires target 'position'"),
    ],
)
def test_control_runtime_rejects_invalid_drive_target_pair(drive, target, match):
    """A runtime accepts only a legal drive and its required target kind."""
    with pytest.raises(ControlError, match=match):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder",),
            q0=(0.0,),
            drive=drive,
            target=target,
        )


def test_control_runtime_rejects_invalid_role():
    """The body/aux role is explicit and closed."""
    with pytest.raises(ControlError, match="role must be one of"):
        _runtime("body", "arm", ("shoulder",), role="prop")


@pytest.mark.parametrize(
    ("joint_names", "match"),
    [((), "owns no joints"), (("shoulder", "shoulder"), "contains duplicates")],
    ids=["empty", "duplicate"],
)
def test_control_runtime_rejects_empty_or_duplicate_joint_names(joint_names, match):
    """A control runtime owns a non-empty, duplicate-free joint list."""
    with pytest.raises(ControlError, match=match):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=joint_names,
            q0=(0.0,) * len(joint_names),
        )


@pytest.mark.parametrize("q0", [None, (0.0,)], ids=["missing", "short"])
def test_control_runtime_requires_q0_aligned_with_joint_names(q0):
    """The q0 vector must have exactly one entry per owned joint."""
    with pytest.raises(ControlError, match=r"q0 has \("):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=q0,
        )


def test_build_derives_drive_from_action_target_and_actuation_law():
    """The runtime drive is target + law, never an authored control field."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    profile = _profile({"body": ("a",)}, {"a": 0.0})
    group = _group(joints=("a",))

    position = _build((group,), arm, profile, law="standard_pd")[0]
    servo = _build((group,), arm, profile, law="native_servo")[0]

    assert (position.target, position.drive) == ("position", "position_pd")
    assert (servo.target, servo.drive) == ("position", "position_servo")


def test_build_derives_torque_drive_for_direct_torque():
    """A torque target requires (and resolves to) the direct_torque drive."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    group = _group(joints=("a",), target="torque")

    runtime = _build((group,), arm, _profile({"body": ("a",)}, {"a": 0.0}), law="direct_torque")[0]

    assert (runtime.target, runtime.drive) == ("torque", "torque")


def test_build_rejects_incompatible_target_and_law():
    """An action target whose law cannot serve it fails teachingly."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    group = _group(joints=("a",), target="torque")

    with pytest.raises(ControlError, match="requires the direct_torque"):
        _build((group,), arm, _profile({"body": ("a",)}, {"a": 0.0}), law="standard_pd")


def _register_law(name: str, drives: frozenset[str]) -> None:
    register_actuator_law(ActuatorLawDescriptor(name=name, drives=drives))


@pytest.mark.parametrize(
    ("target", "laws", "expected"),
    [
        ("position", ("standard_pd",), "position_pd"),
        ("position", ("computed_torque_pd",), "position_pd"),
        ("position", ("standard_pd", "computed_torque_pd"), "position_pd"),
        ("position", ("native_servo",), "position_servo"),
        ("velocity", ("direct_velocity",), "velocity"),
        ("torque", ("direct_torque",), "torque"),
    ],
)
def test_drive_for_target_laws_maps_builtin_laws(target, laws, expected):
    """Every built-in law still resolves to its declared drive."""
    assert drive_for_target_laws(target, laws, what="group 'body'") == expected


def test_drive_for_target_laws_accepts_a_registered_law_for_an_existing_drive():
    """A pack law declaring an existing drive is derived like a builtin."""
    _register_law("acme.pd", frozenset({"position_pd"}))

    drive = drive_for_target_laws("position", ("acme.pd",), what="group 'body'")

    assert drive == "position_pd"


def test_build_derives_position_drive_from_a_registered_pack_law():
    """A pack-registered position_pd law builds a position control runtime."""
    _register_law("acme.pd", frozenset({"position_pd"}))
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))

    runtime = _build(
        (_group(joints=("a",)),),
        arm,
        _profile({"body": ("a",)}, {"a": 0.0}),
        law="acme.pd",
    )[0]

    assert (runtime.target, runtime.drive) == ("position", "position_pd")


def test_drive_for_target_laws_intersects_registered_and_builtin_laws():
    """A pack law and a builtin law sharing a drive resolve to that drive."""
    _register_law("acme.pd", frozenset({"position_pd"}))

    drive = drive_for_target_laws("position", ("acme.pd", "standard_pd"), what="group 'body'")

    assert drive == "position_pd"


def test_drive_for_target_laws_rejects_incompatible_mixed_laws():
    """Laws whose drives do not intersect fail instead of guessing."""
    with pytest.raises(ControlError, match=r"target 'position' requires one of the"):
        drive_for_target_laws("position", ("standard_pd", "native_servo"), what="group 'body'")


def test_drive_for_target_laws_rejects_an_unregistered_law():
    """An unknown law fails naming the law and the registered set."""
    with pytest.raises(ControlError, match=r"unregistered actuation law 'ghost'.*standard_pd"):
        drive_for_target_laws("position", ("ghost",), what="group 'body'")


def test_drive_for_target_laws_rejects_ambiguous_common_drives():
    """A law serving two drives for one target is ambiguous, not first-wins."""
    _register_law("acme.dual", frozenset({"position_pd", "position_servo"}))

    with pytest.raises(ControlError, match="ambiguous across the common drives"):
        drive_for_target_laws("position", ("acme.dual",), what="group 'body'")


def test_drive_for_target_laws_rejects_a_drive_outside_the_closed_vocabulary():
    """A descriptor drive absent from DRIVE_TARGET can never be selected."""
    _register_law("acme.warp_drive", frozenset({"mystery"}))

    with pytest.raises(ControlError, match=r"target 'position'.*acme\.warp_drive"):
        drive_for_target_laws("position", ("acme.warp_drive",), what="group 'body'")


def test_build_control_runtimes_uses_joint_and_profile_q0_defaults():
    """q0 comes from explicit joint inits, profile defaults, then position zero."""
    arm = ArticulationSpec(
        name="arm",
        joints=(JointSpec("b", init=0.4),),
    )
    profile = _profile({"body": ("a", "b", "c")}, {"a": 0.1})
    group = _group(joints=("a", "b", "c"))

    runtime = _build((group,), arm, profile)[0]

    assert runtime.joint_names == ("a", "b", "c")
    assert runtime.q0.tolist() == pytest.approx([0.1, 0.4, 0.0])
    assert runtime.source == "action_spec"
    assert runtime.strategy_name == "position_pd"


def test_explicit_joint_init_wins_over_the_profile_default():
    """A declared joint init is the explicit q0 fact; the profile is the fallback."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a", init=0.9),))
    profile = _profile({"body": ("a",)}, {"a": 0.1})
    group = _group(joints=("a",))

    runtime = _build((group,), arm, profile)[0]

    assert runtime.q0.tolist() == pytest.approx([0.9])


def test_build_control_runtimes_copies_role_and_policy_metadata():
    """The action group role and policy control_dt flow into the runtime."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    profile = _profile({"body": ("a",)}, {"a": 0.0})
    group = _group(joints=("a",), role="body")

    runtime = _build(
        (group,),
        arm,
        profile,
        policies=(_binding(groups=(("arm", "body"),), control_dt=0.02),),
    )[0]

    assert runtime.role == "body"
    assert runtime.control_dt == 0.02


def test_build_control_runtimes_rejects_uncovered_joint():
    """An action joint with no resolved actuation law is rejected."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"), JointSpec("b")))
    group = _group(joints=("a",))

    with pytest.raises(ControlError, match="no actuation group covers"):
        build_control_runtimes(
            OrderedActionSpec((group,)),
            (_binding(),),
            (arm,),
            {"arm": _profile({"body": ("a",)}, {"a": 0.0})},
            {},
        )


def _unused_spec():
    """A spec with no group on the target articulation (zero driven joints)."""
    return OrderedActionSpec((OrderedActionGroup("noop", "other", ("x",), "position"),))


def _coverage_case(kind: str):
    if kind == "missing":
        arm = ArticulationSpec(name="arm", joints=(JointSpec("j"), JointSpec("k")))
        return (
            (arm,),
            OrderedActionSpec((_group(joints=("j",)),)),
            "neither driven by an action group",
        )
    if kind == "passive-conflict":
        arm = ArticulationSpec(
            name="arm",
            joints=(JointSpec("j"),),
            passive_joints=("j",),
        )
        return (
            (arm,),
            OrderedActionSpec((_group(joints=("j",)),)),
            "both driven by an action group",
        )
    if kind == "unknown":
        arm = ArticulationSpec(name="arm", joints=(JointSpec("j"),))
        return (
            (arm,),
            OrderedActionSpec((_group(joints=("j", "z")),)),
            "not in the articulation's actuated joint vocabulary",
        )
    raise AssertionError(f"unknown coverage case {kind!r}")


@pytest.mark.parametrize(
    "kind",
    ["missing", "passive-conflict", "unknown"],
)
def test_validate_action_coverage_rejects_incomplete_or_conflicting(kind):
    """Spec-driven coverage is exact and disjoint against the actuated vocabulary."""
    arts, spec, match = _coverage_case(kind)

    with pytest.raises(ControlError, match=match):
        validate_action_coverage(arts, {}, spec)


def test_build_control_runtimes_falls_back_to_the_adapter_control_dt():
    """A binding that declares no period mirrors the explicit adapter fallback."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    profile = _profile({"body": ("a",)}, {"a": 0.0})
    group = _group(joints=("a",), role="body")

    runtime = build_control_runtimes(
        OrderedActionSpec((group,)),
        (_binding(groups=(("arm", "body"),)),),
        (arm,),
        {"arm": profile},
        _plans(group, arm),
        default_control_dt=0.05,
    )[0]

    assert runtime.control_dt == 0.05


def test_build_control_runtimes_binding_control_dt_wins_over_the_fallback():
    """The policy binding's period is authoritative over the adapter fallback."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"),))
    profile = _profile({"body": ("a",)}, {"a": 0.0})
    group = _group(joints=("a",), role="body")

    runtime = build_control_runtimes(
        OrderedActionSpec((group,)),
        (_binding(groups=(("arm", "body"),), control_dt=0.1),),
        (arm,),
        {"arm": profile},
        _plans(group, arm),
        default_control_dt=0.05,
    )[0]

    assert runtime.control_dt == 0.1


def test_validate_action_coverage_rejects_an_undriven_profile_articulation():
    """An omitted profile-backed articulation cannot build silently unactuated."""
    arm = ArticulationSpec(name="arm")
    profile = _profile({"body": ("a", "b")}, {"a": 0.0, "b": 0.0})

    with pytest.raises(ControlError, match="declares no action group"):
        validate_action_coverage((arm,), {"arm": profile}, _unused_spec())


def test_validate_action_coverage_allows_a_fully_passive_profile_articulation():
    """A zero-action articulation is legal when every actuated joint is passive."""
    arm = ArticulationSpec(name="arm", passive_joints=("a", "b"))
    profile = _profile({"body": ("a", "b")}, {"a": 0.0, "b": 0.0})

    validate_action_coverage((arm,), {"arm": profile}, _unused_spec())


def test_validate_policy_bindings_rejects_unknown_group_reference():
    """A binding may only reference declared action groups."""
    group = _group()

    with pytest.raises(ControlError, match="undeclared action group"):
        validate_policy_bindings((group,), (_binding(groups=(("arm", "hand"),)),))


def test_validate_policy_bindings_rejects_duplicate_policy_ownership():
    """One action group cannot be owned by two policies."""
    group = _group()

    with pytest.raises(ControlError, match="owned by two policies"):
        validate_policy_bindings(
            (group,),
            (
                _binding(adapter="a", groups=(("arm", "body"),)),
                _binding(adapter="b", groups=(("arm", "body"),)),
            ),
        )


def test_validate_policy_bindings_requires_every_group_owned():
    """Every action group must appear in exactly one policy binding."""
    first = _group(name="body")
    second = _group(name="aux", joints=("knee",))

    with pytest.raises(ControlError, match="not owned by any policy"):
        validate_policy_bindings((first, second), (_binding(groups=(("arm", "body"),)),))


def test_resolve_control_dt_prefers_the_binding_over_the_adapter():
    """The binding period is authoritative; the adapter value is the fallback."""
    binding = PolicyBinding("manual", (("arm", "body"),), control_dt=0.1)

    implicit = PolicyBinding("manual", (("arm", "body"),))

    assert resolve_control_dt((binding,), "manual", 0.02) == 0.1
    assert resolve_control_dt((implicit,), "manual", 0.02) == 0.02


def test_resolve_control_dt_rejects_conflicting_binding_periods():
    """One adapter has one clock; disagreeing bindings fail loud."""
    policies = (
        PolicyBinding("manual", (("arm", "body"),), control_dt=0.1),
        PolicyBinding("manual", (("arm", "hand"),), control_dt=0.05),
    )

    with pytest.raises(ValueError, match="conflicting control_dt"):
        resolve_control_dt(policies, "manual", 0.02)


def test_resolve_control_dt_requires_an_explicit_period():
    """Neither a binding nor the adapter declaring a period is an error."""
    with pytest.raises(ValueError, match="must be explicit"):
        resolve_control_dt((PolicyBinding("manual", (("arm", "body"),)),), "manual", None)


def test_go1_go2_same_joint_names_share_one_adapter():
    """Two articulations with identical joint names are disambiguated by identity."""
    joints = ("hip", "knee")
    go1 = ArticulationSpec(name="go1", joints=tuple(JointSpec(j) for j in joints))
    go2 = ArticulationSpec(name="go2", joints=tuple(JointSpec(j) for j in joints))
    groups = (
        _group(name="legs", articulation="go1", joints=joints, role="body"),
        _group(name="legs", articulation="go2", joints=joints, role="body"),
    )
    actuation = ActuationGroup(
        name="legs", law="standard_pd", joints=joints, gains={"kp": 1.0, "kd": 0.1}
    )
    plans = {
        spec.name: resolve_actuation_groups((actuation,), articulation=spec.name, joints=joints)
        for spec in (go1, go2)
    }

    runtimes = build_control_runtimes(
        OrderedActionSpec(groups),
        (PolicyBinding("dog_policy", (("go1", "legs"), ("go2", "legs"))),),
        (go1, go2),
        {"go1": _profile({"body": joints}, {}), "go2": _profile({"body": joints}, {})},
        plans,
    )

    assert [(runtime.articulation, runtime.name) for runtime in runtimes] == [
        ("go1", "legs"),
        ("go2", "legs"),
    ]
    assert all(runtime.joint_names == joints for runtime in runtimes)
    assert all(runtime.role == "body" for runtime in runtimes)


def test_composite_adapter_owns_both_go1_and_go2_groups():
    """The v0 composite-adapter path is one binding over both identities."""
    action = OrderedActionSpec(
        (
            _group(name="legs", articulation="go1", joints=("hip",), role="body"),
            _group(name="legs", articulation="go2", joints=("hip",), role="body"),
        )
    )
    composite = PolicyBinding("two_dog", (("go1", "legs"), ("go2", "legs")))

    validate_policy_bindings(action.groups, (composite,))

    assert composite.adapter == "two_dog"
    assert composite.groups == (("go1", "legs"), ("go2", "legs"))


def test_build_copies_explicit_body_and_aux_roles():
    """Roles are copied from the action group, never inferred from the name."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("a"), JointSpec("b")))
    profile = _profile({"body": ("a",), "hand": ("b",)}, {"a": 0.0, "b": 0.0})
    groups = (
        _group(name="legs", articulation="arm", joints=("a",), role="body"),
        _group(name="gripper", articulation="arm", joints=("b",), role="aux"),
    )
    actuation = ActuationGroup(
        name="all", law="standard_pd", joints=("a", "b"), gains={"kp": 1.0, "kd": 0.1}
    )
    plans = {"arm": resolve_actuation_groups((actuation,), articulation="arm", joints=("a", "b"))}

    runtimes = build_control_runtimes(
        OrderedActionSpec(groups),
        (PolicyBinding("manual", (("arm", "legs"), ("arm", "gripper"))),),
        (arm,),
        {"arm": profile},
        plans,
    )

    assert [(runtime.name, runtime.role) for runtime in runtimes] == [
        ("legs", "body"),
        ("gripper", "aux"),
    ]


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_control_runtime_rejects_non_finite_q0(bad):
    """A control runtime rejects NaN/inf q0 naming the runtime and q0 field."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be finite"):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=(0.0, bad),
        )


def test_articulation_runtime_rejects_non_finite_q0():
    """An articulation runtime uses the same non-finite q0 rejection."""
    with pytest.raises(ControlError, match=r"articulation 'arm' q0 must be finite"):
        ArticulationRuntime(
            name="arm",
            joint_names=("shoulder",),
            q0=(float("nan"),),
        )


def test_control_runtime_rejects_non_numeric_q0():
    """A non-numeric q0 entry fails as a ControlError, not a raw TypeError."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be numeric"):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder",),
            q0=("not-a-number",),
        )


def test_control_runtime_rejects_non_finite_tensor_q0():
    """The tensor q0 path applies the same finiteness check."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be finite"):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=torch.tensor([0.0, float("nan")]),
        )


def test_control_runtime_checks_q0_shape_before_finiteness():
    """A wrong q0 shape reports shape, not a downstream finite error."""
    with pytest.raises(ControlError, match=r"q0 has \(1,\) entries"):
        ControlRuntime(
            name="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=(float("nan"),),
        )


def test_control_runtime_has_no_action_face_member():
    """Identity is ``(articulation, name)``; there is no separate action-face field."""
    assert "action_face" not in {field.name for field in fields(ControlRuntime)}
    assert not hasattr(ControlRuntime, "action_face")


def test_action_routing_source_has_no_action_specific_face_terminology():
    """The routed action path names groups, never a separate action face."""
    root = Path(__file__).resolve().parents[3]
    files = (
        "unigym/runtime/controllers.py",
        "unigym/env/backends/base.py",
        "unigym/run/build.py",
        "unigym/run/api.py",
        "unigym/runner/sinks/collect.py",
        "unigym/runner/sync.py",
        "unigym/robots/base.py",
        "unigym/world.py",
        "unigym/backends/shared/mjfamily/articulation_actions.py",
        "unigym/backends/shared/mjfamily/articulation_runtime.py",
        "unigym/backends/isaaclab2/env.py",
        "unigym/backends/isaaclab2/articulation_runtime.py",
    )
    forbidden = (
        "action_face",
        "action-face",
        "action face",
        "action faces",
        "control face",
        "control faces",
        "accepted faces",
    )
    offenders = []
    for relative in files:
        text = (root / relative).read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{relative}: {token!r}")
    assert not offenders, f"action-specific face terminology remains: {offenders}"


def test_articulation_runtime_keeps_native_and_canonical_base_body():
    """base_body is the backend-native name; base_body_id is the canonical id."""
    from unigym.scenes.articulation import ContactSensorInventory

    runtime = ArticulationRuntime(
        name="g1",
        joint_names=("j",),
        q0=(0.0,),
        foot_bodies=("left_ankle_roll",),
        base_body="pelvis",
        base_body_id="base",
        body_sets={"grasp": ("finger",)},
        contacts=ContactSensorInventory(mode="none"),
    )

    assert runtime.base_body == "pelvis"
    assert runtime.base_body_id == "base"
    assert runtime.foot_bodies == ("left_ankle_roll",)
    assert runtime.body_sets == {"grasp": ("finger",)}
    assert runtime.contacts is not None and runtime.contacts.mode == "none"
    assert not hasattr(runtime, "feet")
    assert not hasattr(runtime, "body_parts")
