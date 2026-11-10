"""Articulation IR invariants: identity, joints, passive joints and normalization."""

from __future__ import annotations

import math

import pytest

from unigym.scenes.articulation import (
    ArticulationError,
    ArticulationSpec,
    AssetRef,
    JointSpec,
    articulation_key,
    normalize_articulations,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name must be non-empty"),
        ({"name": "a", "root": "floating"}, "root must be one of"),
        ({"name": "a", "root": None}, "root must be one of"),
    ],
)
def test_articulation_spec_rejects_empty_name_and_unknown_root(kwargs, match):
    """An articulation requires a name and one of the declared root modes."""
    with pytest.raises(ArticulationError, match=match):
        ArticulationSpec(**kwargs)


def test_articulation_spec_normalizes_joint_mapping_and_passive_list():
    """A joint mapping and a passive list become immutable ordered IR values."""
    spec = ArticulationSpec(
        name="arm",
        joints={"shoulder": {"init": 0.25, "range": [-1.0, 1.0]}, "elbow": {}},
        passive_joints=["elbow"],
    )

    assert [joint.name for joint in spec.joints] == ["shoulder", "elbow"]
    assert spec.joints[0].init == 0.25
    assert spec.joints[0].range == (-1.0, 1.0)
    assert spec.joints[1].init == 0.0
    assert spec.passive_joints == ("elbow",)


def test_articulation_spec_rejects_duplicate_joint_names_in_sequence():
    """A sequence joint declaration may not repeat a joint name."""
    with pytest.raises(ArticulationError, match=r"duplicate joint names \['j'\]"):
        ArticulationSpec(name="a", joints=(JointSpec(name="j"), JointSpec(name="j")))


def test_articulation_spec_rejects_duplicate_joint_names_in_mapping():
    """Mapping keys that normalize to one joint name are duplicate names."""
    with pytest.raises(ArticulationError, match=r"duplicate joint names \['1'\]"):
        ArticulationSpec(name="a", joints={1: {}, "1": {}})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"range": (1.0, 0.0)}, "lo > hi"),
        ({"range": (0.0, 1.0), "init": 2.0}, "outside the declared range"),
        ({"init": math.nan}, "must be finite"),
        ({"range": (0.0,)}, "must be"),
    ],
)
def test_joint_spec_rejects_invalid_range_and_init(kwargs, match):
    """A joint range must be ordered and its initial value inside that range."""
    with pytest.raises(ArticulationError, match=match):
        JointSpec(name="j", **kwargs)


@pytest.mark.parametrize("params", [{"mode": "policy"}, {"stiffness": 1.0}, {"latch": True}])
def test_joint_spec_rejects_removed_v104_control_keys(params):
    """Removed per-joint control keys fail with a teaching error, never an alias."""
    with pytest.raises(ArticulationError, match="removed v104 key"):
        JointSpec.from_mapping("j", params)


@pytest.mark.parametrize(
    ("passive_joints", "match"),
    [
        (("j", "j"), r"passive_joints contains duplicates"),
        (("ghost",), r"passive_joints \['ghost'\] are not declared joints"),
    ],
)
def test_articulation_spec_rejects_invalid_passive_joints(passive_joints, match):
    """Passive joints are unique and must name declared joints when joints exist."""
    with pytest.raises(ArticulationError, match=match):
        ArticulationSpec(
            name="a",
            joints=(JointSpec(name="j"),),
            passive_joints=passive_joints,
        )


def test_articulation_spec_allows_passive_joints_when_joints_are_recipe_derived():
    """A joint-less recipe articulation may still declare passive joints."""
    spec = ArticulationSpec(name="a", passive_joints=("wheel",))

    assert spec.joints == ()
    assert spec.passive_joints == ("wheel",)


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        (None, "declares no articulations"),
        ([], "non-empty list"),
        ([{"name": ""}], "name must be non-empty"),
        ([{"root": "free"}], "missing 'name'"),
        ([{"name": "a"}, {"name": "a"}], "names must be unique"),
        (
            [{"name": "a", "joints": [{"name": "j"}, {"name": "j"}]}],
            "duplicate joint names",
        ),
        (
            [{"name": "a", "joints": {1: {}, "1": {}}}],
            "duplicate joint names",
        ),
        ([{"name": "a", "bogus": 1}], r"unknown keys \['bogus'\]"),
        ([{"name": "a", "control": {}}], "removed v104"),
    ],
)
def test_normalize_articulations_rejects_malformed_entries(entries, match):
    """Normalization rejects empty, duplicate, unnamed, unknown and removed shapes."""
    with pytest.raises(ArticulationError, match=match):
        normalize_articulations(entries)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: AssetRef.from_mapping({"mjcf": "a", "glb": "b"}), r"unknown keys \['glb'\]"),
        (lambda: AssetRef.from_mapping("robot.urdf"), "must be a mapping"),
        (lambda: articulation_key("", "k"), "name must be non-empty"),
        (lambda: articulation_key("a", ""), "observation key must be non-empty"),
    ],
)
def test_malformed_asset_ref_and_articulation_key_fail_fast(call, match):
    """Asset references and observation keys reject malformed input explicitly."""
    with pytest.raises(ArticulationError, match=match):
        call()
