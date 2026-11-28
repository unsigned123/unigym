"""First-party robocasa contact sections build through the new schema."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PACK_ROOT = Path(os.environ.get("UNIGYM_PACKS_ROOT", "/data1/code/dlx/unigym_packs"))
RUNS = PACK_ROOT / "robocasa" / "runs"

CONFIGS = (
    "TabletopTurnOffMicrowave.yaml",
    "TabletopTurnOnMicrowave.yaml",
    "PosttrainPnPNovelFromPlacematToPlateSplitA.yaml",
    "PosttrainPnPNovelFromPlateToPlateSplitA.yaml",
)


def _load(name: str) -> dict:
    path = RUNS / name
    if not path.is_file():
        pytest.skip(f"first-party pack config {path} is not available")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", CONFIGS)
def test_first_party_contact_section_builds(name):
    from tests._support.backends import contact_factory
    from unigym.contacts.manager import build_contact_manager

    cfg = _load(name)
    contacts = cfg.get("contacts")
    assert contacts, f"{name} declares no contacts section"
    for entry in contacts:
        assert "sensor_bodies" not in entry
        assert "filter_bodies" not in entry
        assert "mode" not in entry
        assert entry["sensor_refs"]

    articulations = cfg.get("articulations") or [{"name": "gr1"}]
    props = [item["name"] for item in (cfg.get("objects") or {}).get("types", [])]
    articulated = {
        item["name"]
        for item in (cfg.get("objects") or {}).get("types", [])
        if "articulation" in str(item.get("type", "")) or item.get("template")
    }
    manager = build_contact_manager(
        contacts,
        backend="mujoco",
        articulation_names=tuple(item["name"] for item in articulations),
        prop_names=props,
        articulated_props=articulated,
        contact_factory=contact_factory("mujoco"),
    )
    assert manager is not None
    assert manager.keys()
