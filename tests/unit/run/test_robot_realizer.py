"""The run-layer robot realizer seam is backend-explicit."""

from __future__ import annotations

from unigym.actuators import ActuationGroup
from unigym.backends.registry import backend_info
from unigym.run.build import _realize_profiles
from unigym.scenes.articulation import ArticulationSpec
from tests._support.profiles import fixture_profile


def test_mujoco_has_no_robot_realization() -> None:
    """Mujoco consumes neutral profiles and builds no engine robot object."""
    builds = {
        "fixture_arm": {
            "profile": fixture_profile(),
            "params": {},
            "spec": ArticulationSpec(
                name="fixture_arm",
                actuation=(
                    ActuationGroup(
                        name="body",
                        law="standard_pd",
                        joint_set="body",
                        gains={"kp": 1.0, "kd": 1.0},
                    ),
                ),
            ),
        }
    }

    assert _realize_profiles(backend_info("mujoco"), builds, None) is None


def test_realizer_receives_profile_spec_params_and_asset_plan() -> None:
    """A backend realizer gets the profile/spec/params/asset plan."""
    seen: list[tuple[object, object, object, object]] = []

    class _Info:
        robot_realizer = staticmethod(lambda: realizer)

    def realizer(profile, *, spec, params, asset_plan):
        seen.append((profile, spec, params, asset_plan))
        return "realized"

    spec = ArticulationSpec(
        name="arm",
        spawn={"pos": [0.0, 0.0, 1.0]},
        actuation=(
            ActuationGroup(
                name="body", law="standard_pd", joints=("shoulder",), gains={"kp": 1.0, "kd": 1.0}
            ),
        ),
    )
    builds = {
        "arm": {
            "profile": fixture_profile(),
            "params": {"x": 1},
            "spec": spec,
        }
    }

    result = _realize_profiles(_Info(), builds, None)

    assert result == {"arm": "realized"}
    assert seen[0][0] is builds["arm"]["profile"]
    assert seen[0][1] is spec
    assert seen[0][2] == {"x": 1}
    assert seen[0][3] is None
