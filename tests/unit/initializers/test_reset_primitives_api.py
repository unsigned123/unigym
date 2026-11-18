"""ResetPrimitives API surface: the stubs are gone and nothing calls them."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from unigym.initializers import ResetPrimitives

_CORE = Path(__file__).resolve().parents[3] / "unigym"


def test_removed_stub_fields_are_gone() -> None:
    """The robot-side stubs are replaced by the two explicit writes."""
    names = {field.name for field in fields(ResetPrimitives)}

    assert "joint_group" not in names
    assert "set_targets" not in names
    assert {"set_root_state", "set_joint_state"} <= names


def test_no_core_caller_of_the_removed_primitives() -> None:
    """Grep proof: no core module reaches the removed primitive names."""
    needles = ("primitives.joint_group", "primitives.set_targets")
    offenders = []
    for path in sorted(_CORE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(_CORE.parent)}: {needle}")

    assert not offenders, offenders
