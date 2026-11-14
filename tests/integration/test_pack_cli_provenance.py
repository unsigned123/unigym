"""Integration: ``unigym pack list --json`` is a projection of ``PackSet``.

The CLI must not own a second provenance or discovery path.  For the same
runtime, every row (name, version, api, trust, provided-item counts, root and
pack-level origin) equals the :class:`~unigym.packs.PackSet` projection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from unigym.pack_api import discover
from unigym.packs import PackSet

pytestmark = pytest.mark.subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
ROOTS = (PACKS_ROOT / "acme_content", PACKS_ROOT / "acme_code")


def _env() -> dict[str, str]:
    """The shared environment for the CLI subprocess and the in-process runtime."""
    env = os.environ.copy()
    env["UNIGYM_PACKS"] = os.pathsep.join(str(root) for root in ROOTS)
    env["UNIGYM_DEPS_ROOT"] = ""
    return env


def _run_pack_list(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the real launcher's ``pack list --json`` from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "unigym.scripts.launch", "pack", "list", "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _pack_set_projection(runtime) -> list[dict[str, object]]:
    """The same rows the CLI projects, straight from ``PackSet``."""
    rows: list[dict[str, object]] = []
    for pack in PackSet.from_runtime(runtime).packs():
        rows.append(
            {
                "name": pack.name,
                "version": pack.version,
                "api": pack.api,
                "trust": pack.trust,
                "provides": dict(pack.face_counts),
                "root": str(pack.root),
                "origin": asdict(pack.origin),
            }
        )
    return rows


def test_pack_list_json_matches_pack_set_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI rows and pack-set rows are identical for the same runtime."""
    env = _env()
    monkeypatch.setenv("UNIGYM_PACKS", env["UNIGYM_PACKS"])
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", env["UNIGYM_DEPS_ROOT"])

    result = _run_pack_list(env)

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    document = json.loads(result.stdout)
    assert document["ok"] is True

    runtime = discover(entry_points=True)
    assert document["packs"] == _pack_set_projection(runtime)
    assert [row["name"] for row in document["packs"]] == ["acme_content", "acme_code"]
    assert {row["origin"]["discovered_by"] for row in document["packs"]} == {"env:UNIGYM_PACKS"}
