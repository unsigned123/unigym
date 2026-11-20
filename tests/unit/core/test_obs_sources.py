"""Manager-driven observation sources: selected cameras and single contact reads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.backends.obs_schema import build_site_inventory
from unigym.backends.shared.mjfamily.articulation_obs import refresh_obs_sources
from unigym.cameras.base import CameraSpec, rgb_key
from unigym.cameras.manager import CameraManager
from unigym.contacts.manager import ContactManager
from unigym.env.obs import ObservationConfig, ObservationRegistry, build_observation_schema
from unigym.sites import SiteSpec
from unigym.props.asset_ir import AssetDefinition, Body, Site
from unigym.robots.base import BodyBinding, RobotProfile


class _Camera:
    def __init__(self) -> None:
        self.calls = 0

    def tick_and_read(self) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"cam/ego/rgb": torch.zeros(1, 3, 2, 2, dtype=torch.uint8)}


class _Contacts:
    def __init__(self) -> None:
        self.calls = 0
        self.detail_calls = 0

    def read(self) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"contact/feet": torch.tensor([[True, False]])}

    def has_detail(self, name: str) -> bool:
        del name
        return True

    def read_detail(self, name: str) -> list[dict]:
        self.detail_calls += 1
        return [{"group": name}]


class _Props:
    def get_world_poses(self, env) -> dict:
        del env
        return {"cube": (torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))}

    def get_joint_states(self, env) -> dict:
        del env
        return {"cube": {"lid": torch.zeros(1, 1)}}


def test_contact_is_read_once_per_refresh_and_only_selected_groups_publish() -> None:
    config = ObservationConfig.from_mapping(
        {"robot": ["arm"], "contact": ["feet"], "camera": ["ego"]}
    )
    schema = build_observation_schema(
        config,
        full_joint_names={"arm": ("shoulder",)},
        contacts={"feet": ((2,), "bool"), "other": ((2,), "bool")},
        cameras={"ego": {"cam/ego/rgb": ((3, 2, 2), "u8")}},
    )
    registry = ObservationRegistry(config, schema, num_envs=1, device="cpu")
    camera = _Camera()
    contacts = _Contacts()
    env = SimpleNamespace(
        obs=registry,
        camera_manager=camera,
        contact_manager=contacts,
        prop_backend=None,
        num_envs=1,
        _obs_sites=(),
        _data=None,
    )

    refresh_obs_sources(env)
    refresh_obs_sources(env)

    assert contacts.calls == 2  # exactly once per refresh
    assert contacts.detail_calls == 0  # detail is lazy (explicit access only)
    assert camera.calls == 2
    assert "contact/feet" in registry.snapshot().values
    assert "contact/other" not in registry.snapshot().values


def test_contact_manager_detail_capability_is_explicit() -> None:
    spec = SimpleNamespace(name="feet", format="contact")

    class _NoDetail:
        def setup_scene(self):
            return {}

        def bind_env(self, env):
            pass

        def read_raw(self, spec):
            return torch.zeros(1, 1, dtype=torch.bool)

        def close(self):
            pass

    class _Detail(_NoDetail):
        def read_detail(self, spec):
            return [{"group": spec.name}]

    assert ContactManager([spec], _NoDetail()).has_detail("feet") is False
    assert ContactManager([spec], _Detail()).has_detail("feet") is True


def test_object_observation_publishes_root_pose_and_declared_joints() -> None:
    config = ObservationConfig.from_mapping({"robot": ["arm"], "object": ["cube"]})
    schema = build_observation_schema(
        config,
        full_joint_names={"arm": ("shoulder",)},
        objects={"cube": ("lid",)},
    )
    registry = ObservationRegistry(config, schema, num_envs=1, device="cpu")
    env = SimpleNamespace(
        obs=registry,
        camera_manager=None,
        contact_manager=None,
        prop_backend=_Props(),
        num_envs=1,
        _obs_sites=(),
        _data=None,
    )

    refresh_obs_sources(env)

    cube = registry.snapshot().object("cube")
    assert torch.equal(cube.pos_w, torch.zeros(1, 3))
    assert torch.equal(cube.quat_w, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(cube.joints["lid"], torch.zeros(1, 1))


def test_camera_manager_obs_rig_split_ignores_video_only_cameras() -> None:
    rigs = [
        SimpleNamespace(spec=SimpleNamespace(name="ego")),
        SimpleNamespace(spec=SimpleNamespace(name="video")),
    ]
    manager = CameraManager([], rigs, obs_served={"ego"})

    assert [rig.spec.name for rig in manager._obs_rigs] == ["ego"]


def test_video_only_camera_renders_every_call() -> None:
    """Two video captures render twice and return distinct frames."""
    spec = CameraSpec(name="video", types=("rgb",), state="per_instance", width=2, height=2)

    class _Rig:
        spec = None
        render_spec = None

        def __init__(self) -> None:
            self.calls = 0

        def read(self, env_ids=None):
            del env_ids
            self.calls += 1
            return {rgb_key("video"): torch.full((1, 2, 2, 3), self.calls, dtype=torch.uint8)}

    rig = _Rig()
    rig.spec = spec
    manager = CameraManager([spec], [rig], obs_served=set())
    manager.num_envs = 1
    assert manager._obs_rigs == []

    first = manager.video_rgb("video")
    second = manager.video_rgb("video")
    assert rig.calls == 2
    assert first.shape == second.shape == (2, 2, 3)
    assert first.dtype.name == "uint8"
    assert not (first == second).all()
    assert int(first.max()) == 1 and int(second.max()) == 2


def _profile(name: str, site: SiteSpec) -> RobotProfile:
    return RobotProfile(
        name=name,
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "root", "usd": "base"})},
        sites={site.name: site},
    )


def test_site_inventory_records_articulation_and_prop_owners() -> None:
    profile = _profile("arm", SiteSpec("tcp", "base"))
    asset = AssetDefinition(
        name="cup",
        root=Body("cup", sites=(Site("handle", pos=(0.0, 0.0, 0.01)),)),
    )

    inventory = build_site_inventory({"arm": profile}, {"cup": asset})

    assert inventory["tcp"].kind == "articulation"
    assert inventory["tcp"].owner == "arm"
    assert inventory["tcp"].spec == profile.sites["tcp"]
    assert inventory["handle"].kind == "prop"
    assert inventory["handle"].owner == "cup"
    assert inventory["handle"].spec.body == "cup"


def test_site_inventory_rejects_duplicate_names_across_owners() -> None:
    left = _profile("left", SiteSpec("tcp", "base"))
    right = _profile("right", SiteSpec("tcp", "base"))

    with pytest.raises(ValueError, match="duplicate site name 'tcp'.*'left'.*'right'"):
        build_site_inventory({"left": left, "right": right}, {})


def test_site_inventory_rejects_owner_identity_mismatch() -> None:
    asset = AssetDefinition(name="cup", root=Body("cup", sites=(Site("handle"),)))

    with pytest.raises(ValueError, match="owner and payload identity"):
        build_site_inventory({}, {"other": asset})
