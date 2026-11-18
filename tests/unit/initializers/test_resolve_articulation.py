"""ResetPrimitives articulation-name resolution is profile-name neutral."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unigym.backends.shared.mjfamily.props.reset_face import build_reset_primitives


def _env(*arts):
    return SimpleNamespace(
        mj_model=object(),
        articulations={art.name: art for art in arts},
        settle_fn=None,
        _settle_substeps=lambda substeps: None,
        _prop_extents=lambda name: (0.0, 0.0),
        prop_manager=None,
        prop_backend=None,
    )


def _art(name: str, profile: str):
    return SimpleNamespace(name=name, profile=SimpleNamespace(name=profile))


def test_resolve_articulation_uses_the_env_name_not_the_profile_name() -> None:
    primitives = build_reset_primitives(_env(_art("robot", "humanoid")))

    assert primitives.resolve_articulation("humanoid") == "robot"


def test_resolve_articulation_rejects_unknown_or_ambiguous_profiles() -> None:
    primitives = build_reset_primitives(_env(_art("left", "profile_a"), _art("right", "profile_a")))

    with pytest.raises(ValueError, match="ambiguous"):
        primitives.resolve_articulation("profile_a")
    with pytest.raises(ValueError, match="no articulation"):
        primitives.resolve_articulation("nope")
