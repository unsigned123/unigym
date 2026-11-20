"""MuJoCo law config accepts params as the source of gains."""

from __future__ import annotations

from unigym.actuators import ActuationGroup, ResolvedActuationGroup
from unigym.backends.shared.mjfamily.articulation_runtime import _law_config


def test_law_config_reads_required_gains_from_params() -> None:
    group = ActuationGroup(
        name="body",
        law="standard_pd",
        joints=("j1",),
        params={"kp": 1.0, "kd": 2.0, "damping": 0.2},
    )
    resolved = ResolvedActuationGroup(articulation="arm", group=group, joint_names=("j1",))

    cfg = _law_config(resolved, ("compiled_j1",))

    assert cfg["type"] == "standard_pd"
    assert cfg["joint_names"] == ["compiled_j1"]
    assert cfg["kp"] == 1.0
    assert cfg["kd"] == 2.0
    assert cfg["damping"] == 0.2
