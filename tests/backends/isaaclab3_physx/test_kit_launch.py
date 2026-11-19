"""Kit launch smoke for the IsaacLab 3 PhysX lane."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    launcher = AppLauncher(args)
    assert launcher.app is not None
    print("P0_KIT_OK", flush=True)
    launcher.app.close()
    """
)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.subprocess
def test_app_launcher_headless_smoke() -> None:
    """Launch Kit in a child process: ``app.close()`` ends the process."""
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "P0_KIT_OK" in proc.stdout
