"""``unigym.robot/1`` loader builds the exact neutral profile or fails closed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unigym.assets.asset_ref import AssetRef
from unigym.robots.base import BodyBinding
from unigym.robots.data import RobotDataError, load_robot_profile


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "unigym.robot/1",
        "name": "acme.arm",
        "joint_sets": {"body": ["shoulder", "elbow"]},
        "default_joint_pos": {"shoulder": 0.0, "elbow": 0.0},
        "base_body": "base",
        "base_height": 0.5,
        "foot_bodies": [],
        "body_sets": {},
        "body_bindings": {"base": {"mjcf": "base", "usd": "/World/base"}},
        "assets": {"usd": "pack:acme/robots/acme_arm.usd"},
        "action_faces": {},
        "full_joint_order": ["shoulder", "elbow"],
    }
    document.update(overrides)
    return document


def _write(path: Path, document: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_loader_builds_the_exact_profile(tmp_path: Path) -> None:
    """A valid document maps every field (bindings and AssetRef included)."""
    path = _write(tmp_path / "robot.yaml", _document())

    profile = load_robot_profile(path)

    assert profile.name == "acme.arm"
    assert profile.joint_order == ["shoulder", "elbow"]
    assert dict(profile.default_joint_pos) == {"shoulder": 0.0, "elbow": 0.0}
    assert profile.assets == AssetRef(usd="pack:acme/robots/acme_arm.usd")
    assert profile.assets.usd == "pack:acme/robots/acme_arm.usd"
    assert profile.base_body == "base"
    assert profile.base_height == pytest.approx(0.5)
    assert profile.foot_bodies == ()
    assert profile.body_sets == {}
    assert profile.body_bindings["base"] == BodyBinding(
        "base", {"mjcf": "base", "usd": "/World/base"}
    )
    assert profile.action_faces == {}
    assert profile.full_joint_names == ["shoulder", "elbow"]


@pytest.mark.parametrize("key", ["schema", "name", "joint_sets", "default_joint_pos"])
def test_loader_rejects_a_missing_required_field(tmp_path: Path, key: str) -> None:
    """Each required field is a teaching error naming the field."""
    document = _document()
    del document[key]
    path = _write(tmp_path / "robot.yaml", document)

    with pytest.raises(RobotDataError, match=key):
        load_robot_profile(path)


def test_loader_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    """Only top-level unknown keys fail; nested binding formats are data."""
    path = _write(tmp_path / "robot.yaml", _document(unknown_field=True))

    with pytest.raises(RobotDataError, match="unknown key"):
        load_robot_profile(path)


def test_loader_rejects_a_wrong_schema_id(tmp_path: Path) -> None:
    """The schema id is exact, not a compatible prefix."""
    path = _write(tmp_path / "robot.yaml", _document(schema="unigym.robot/2"))

    with pytest.raises(RobotDataError, match="schema"):
        load_robot_profile(path)


@pytest.mark.parametrize(
    "bindings",
    [
        {"base": {}},
        {"base": "root"},
        {"": {"mjcf": "root"}},
    ],
)
def test_loader_wraps_invalid_body_bindings(tmp_path: Path, bindings: object) -> None:
    """A malformed binding fails with the file and the binding field named."""
    path = _write(tmp_path / "robot.yaml", _document(body_bindings=bindings))

    with pytest.raises(RobotDataError, match="body_bindings"):
        load_robot_profile(path)


def test_loader_rejects_an_unknown_asset_format(tmp_path: Path) -> None:
    """Assets use the closed AssetRef format vocabulary."""
    path = _write(tmp_path / "robot.yaml", _document(assets={"obj": "robot.obj"}))

    with pytest.raises(RobotDataError, match="assets"):
        load_robot_profile(path)


@pytest.mark.parametrize("assets", [None, "omitted"])
def test_loader_passes_none_for_null_or_omitted_assets(tmp_path: Path, assets: object) -> None:
    """``assets: null`` and a missing key both reach RobotProfile as None."""
    document = _document()
    if assets == "omitted":
        del document["assets"]
    else:
        document["assets"] = None
    path = _write(tmp_path / "robot.yaml", document)

    profile = load_robot_profile(path)

    assert profile.assets is None


def test_loader_wraps_profile_validation_errors(tmp_path: Path) -> None:
    """RobotProfile invariants stay owned by the profile, wrapped with the file."""
    path = _write(tmp_path / "robot.yaml", _document(joint_sets={"body": []}))

    with pytest.raises(RobotDataError, match="joint set"):
        load_robot_profile(path)


def test_loader_rejects_unreadable_and_malformed_files(tmp_path: Path) -> None:
    """A missing file and invalid YAML are both teaching RobotDataErrors."""
    with pytest.raises(RobotDataError, match="unreadable robot profile"):
        load_robot_profile(tmp_path / "missing.yaml")

    path = tmp_path / "malformed.yaml"
    path.write_text("schema: [unterminated\n", encoding="utf-8")
    with pytest.raises(RobotDataError, match="unreadable robot profile"):
        load_robot_profile(path)


def test_load_robot_profile_is_lazily_exported() -> None:
    """The package exposes the loader through PEP 562 without caching it."""
    import unigym.robots as robots

    assert "load_robot_profile" in robots.__all__
    assert "load_robot_profile" not in robots.__dict__

    from unigym.robots.data import load_robot_profile as data_loader

    assert robots.load_robot_profile is data_loader
