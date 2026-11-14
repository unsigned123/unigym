"""The shared mjfamily composer handles file assets in the MWarp lane."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._support.envs import require_optional_module
from unigym.assets.asset_ref import AssetRef
from unigym.scenes.artifacts import ASSET_SCENE_DIGEST_VERSION
from unigym.scenes.scene_ir import AssetInstance, Pose, Scene

_WARP_REASON = (
    "MuJoCo Warp is an optional extra (unigym[warp]); install it or run this "
    "lane in the warp environment"
)

STATIC_MJCF = """<mujoco model="static_asset">
  <worldbody>
    <body name="root"><geom name="body" type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""

DYNAMIC_MJCF = """<mujoco model="dynamic_asset">
  <worldbody>
    <body name="root" pos="0 0 0.5">
      <freejoint/>
      <geom name="body" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _require_warp_lane() -> None:
    """Skip locally (or fail in a strict lane) when the warp lane is absent."""
    for module_name in ("mujoco_warp", "warp"):
        require_optional_module(module_name, reason=_WARP_REASON)


def _asset_plan():
    """A real empty asset boundary (absolute fixture paths need no pack)."""
    from unigym.assets.plan import AssetPlan
    from unigym.packs.runtime import AssetContext, discover

    return AssetPlan(AssetContext(discover([])))


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text(text, encoding="utf-8")
    return path


def _scene(path: Path, kind: str) -> Scene:
    return Scene(
        name="bench",
        asset_instances=(
            AssetInstance(
                id="crate",
                asset=AssetRef(mjcf=str(path)),
                pose=Pose(xyz=(1.0, 2.0, 3.0)),
                source="scene.yaml:1",
                kind=kind,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("name", "mjcf", "kind"),
    [
        ("static", STATIC_MJCF, "static_rigid"),
        ("dynamic", DYNAMIC_MJCF, "dynamic_rigid"),
    ],
)
def test_shared_engine_composes_and_describes_an_asset(
    tmp_path, name: str, mjcf: str, kind: str
) -> None:
    """The MWarp lane composes the shared asset spec and describes its rows."""
    _require_warp_lane()
    from unigym.backends.shared.mjfamily.scene.scene_engine import MujocoSceneEngine
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write(tmp_path, name, mjcf)
    scene = _scene(path, kind)
    composed = compose_scene(scene, _asset_plan())

    assert dict(composed.asset_paths) == {"crate": str(path)}
    model = composed.compile()
    assert int(model.nq) == (0 if kind == "static_rigid" else 7)

    described = MujocoSceneEngine().describe(composed)
    assert described["asset_scene"]["digest_version"] == ASSET_SCENE_DIGEST_VERSION
    rows = described["asset_scene"]["instances"]
    assert [row["id"] for row in rows] == ["crate"]
    assert rows[0]["kind"] == kind
    assert rows[0]["format"] == "mjcf"
    assert "resolved" not in rows[0]
