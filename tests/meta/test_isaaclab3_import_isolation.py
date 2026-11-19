"""Import isolation for the IsaacLab 3 backend package.

The d0 base environment has no engine installed, but an in-process check can be
polluted by an earlier test.  Run the import in a subprocess so the assertion
is about the import graph, not about test ordering.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = """
import json
import sys

import unigym.backends.isaaclab3_newton
import unigym.backends.isaaclab3_physx
import unigym.backends.registry
import unigym.backends.shared.isaaclab3_common
import unigym.backends.shared.isaaclab3_common.asset_scene
import unigym.backends.shared.isaaclab3_common.asset_scene.artifacts
import unigym.backends.shared.isaaclab3_common.asset_scene.inspect
import unigym.backends.shared.isaaclab3_common.asset_scene.names
import unigym.backends.shared.isaaclab3_common.cfg_classes
import unigym.backends.shared.isaaclab3_common.init_pose
import unigym.backends.shared.isaaclab3_common.naming
import unigym.backends.shared.isaaclab3_common.terrains
import unigym.backends.shared.isaaclab3_common.terrains.base
import unigym.backends.shared.isaaclab3_common.terrains.custom
import unigym.backends.shared.isaaclab3_common.terrains.custom._compat
import unigym.backends.shared.isaaclab3_common.terrains.custom.gap
import unigym.backends.shared.isaaclab3_common.terrains.custom.pit
import unigym.backends.shared.isaaclab3_common.terrains.custom.slope_rough
import unigym.backends.shared.isaaclab3_common.terrains.custom.unidirectional
import unigym.backends.shared.isaaclab3_common.usd

print("P0_IMPORT_JSON:" + json.dumps(sorted({name.split(".", 1)[0] for name in sys.modules})))
"""

_ENGINE_TOP_LEVEL = {
    "isaaclab",
    "isaaclab_physx",
    "isaaclab_newton",
    "isaaclab_tasks",
    "isaaclab_assets",
    "newton",
    "mujoco",
    "mujoco_warp",
    "warp",
    "pxr",
    "omni",
    "isaacsim",
    "torch",
}


@pytest.mark.subprocess
def test_importing_the_backend_package_loads_no_engine() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    markers = [
        line[len("P0_IMPORT_JSON:") :]
        for line in proc.stdout.splitlines()
        if line.startswith("P0_IMPORT_JSON:")
    ]
    assert markers, proc.stdout
    loaded = set(json.loads(markers[-1]))
    leaked = loaded & _ENGINE_TOP_LEVEL
    assert not leaked, f"importing the backend package loaded {sorted(leaked)}"


def test_canonical_backends_include_both_identities() -> None:
    """The registry names both identities and rejects the removed aliases."""
    from unigym.backends.registry import CANONICAL_BACKENDS, build_name, known_backends

    names = known_backends()
    assert list(CANONICAL_BACKENDS) == [
        "mujoco",
        "mjwarp",
        "isaaclab2",
        "isaaclab3_physx",
        "isaaclab3_newton",
    ]
    assert names == list(CANONICAL_BACKENDS)
    assert "isaaclab3_physx" in names
    assert "isaaclab3_newton" in names
    assert "isaaclab" not in names
    assert "isaaclab3" not in names

    for alias in ("isaaclab", "isaaclab3"):
        with pytest.raises(ValueError, match=r"aliases were removed"):
            build_name(alias)
