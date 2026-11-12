"""The run-layer robot realizer seam is backend-explicit."""

from __future__ import annotations

from unigym.backends.registry import backend_info
from unigym.robots import build_robot_profile
from unigym.run.build import _realize_profiles


def test_mujoco_has_no_robot_realization() -> None:
    """Mujoco consumes neutral profiles and builds no engine robot object."""
    builds = {"g1": {"profile": build_robot_profile({"name": "g1"}), "params": {}}}

    assert _realize_profiles(backend_info("mujoco"), builds, None) is None


def test_realizer_receives_neutral_profile_and_params() -> None:
    """A backend realizer gets the profile/params/asset plan only."""
    seen: list[tuple[object, object, object]] = []

    class _Info:
        robot_realizer = staticmethod(lambda: realizer)

    def realizer(profile, *, params, asset_plan):
        seen.append((profile, params, asset_plan))
        return "realized"

    builds = {"arm": {"profile": build_robot_profile({"name": "g1"}), "params": {"x": 1}}}

    result = _realize_profiles(_Info(), builds, None)

    assert result == {"arm": "realized"}
    assert seen[0][0] is builds["arm"]["profile"]
    assert seen[0][1] == {"x": 1}
    assert seen[0][2] is None
