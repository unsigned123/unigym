"""Shared IsaacLab 3 event builder (DI core) and the Newton CoM gap."""

from __future__ import annotations

import types

import pytest

from unigym.backends.shared.isaaclab3_common.dr_events import (
    DEPRECATED_EVENT_FUNCS,
    build_event_terms,
    convert_params,
    randomize_rigid_body_com_abs,
    resolve_func,
)


class _Entity:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _Term:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _mdp():
    return types.SimpleNamespace(
        reset_root_state_uniform=lambda env, env_ids, **params: None,
        nested=types.SimpleNamespace(reset=lambda env, env_ids, **params: None),
        not_a_function=42,
    )


def test_convert_params_builds_asset_cfg_and_tupleizes_ranges():
    """``asset_cfg`` becomes a SceneEntityCfg; lists become tuples recursively."""
    params = {
        "asset_cfg": {"name": "robot"},
        "pose_range": {"x": [-1.0, 1.0], "y": [0.0, 0.0]},
        "velocity_range": [0.1, 0.2],
    }

    out = convert_params(params, _Entity)

    assert isinstance(out["asset_cfg"], _Entity)
    assert out["asset_cfg"].kwargs == {"name": "robot"}
    assert out["pose_range"] == {"x": (-1.0, 1.0), "y": (0.0, 0.0)}
    assert out["velocity_range"] == (0.1, 0.2)


def test_resolve_func_accepts_dotted_paths_and_rejects_non_callables():
    """A dotted mdp path resolves; a non-callable target is a teaching error."""
    mdp = _mdp()

    assert resolve_func("reset_root_state_uniform", mdp) is mdp.reset_root_state_uniform
    assert resolve_func("nested.reset", mdp) is mdp.nested.reset
    with pytest.raises(ValueError, match="non-callable"):
        resolve_func("not_a_function", mdp)


def test_build_event_terms_builds_named_terms_with_interval():
    """A valid yaml entry becomes one named EventTermCfg-shaped term."""
    events = [
        {
            "func": "reset_root_state_uniform",
            "mode": "reset",
            "params": {"asset_cfg": {"name": "robot"}, "pose_range": {"x": [0.0, 0.1]}},
        },
        {
            "func": "nested.reset",
            "mode": "interval",
            "interval_range_s": [1.0, 2.0],
            "is_global_time": True,
        },
    ]

    terms = build_event_terms(
        events,
        mdp_module=_mdp(),
        term_cfg_cls=_Term,
        scene_entity_cfg_cls=_Entity,
    )

    assert [name for name, _term in terms] == [
        "term_00_reset_root_state_uniform",
        "term_01_nested_reset",
    ]
    first = terms[0][1].kwargs
    assert first["mode"] == "reset"
    assert first["params"]["pose_range"] == {"x": (0.0, 0.1)}
    assert isinstance(first["params"]["asset_cfg"], _Entity)
    second = terms[1][1].kwargs
    assert second["interval_range_s"] == (1.0, 2.0)
    assert second["is_global_time"] is True


def test_build_event_terms_rejects_non_list_payload():
    """The yaml events section must be a list."""
    with pytest.raises(ValueError, match="must be a list"):
        build_event_terms(
            {"func": "reset_root_state_uniform", "mode": "reset"},
            mdp_module=_mdp(),
            term_cfg_cls=_Term,
            scene_entity_cfg_cls=_Entity,
        )


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"mode": "reset"}, "needs 'func' and 'mode'"),
        ({"func": "reset_root_state_uniform"}, "needs 'func' and 'mode'"),
        ({"func": "nested.reset", "mode": "interval"}, "needs interval_range_s"),
    ],
)
def test_build_event_terms_rejects_incomplete_entries(entry, match):
    """Missing func/mode and interval mode without a range fail fast."""
    with pytest.raises(ValueError, match=match):
        build_event_terms(
            [entry],
            mdp_module=_mdp(),
            term_cfg_cls=_Term,
            scene_entity_cfg_cls=_Entity,
        )


def test_build_event_terms_redirects_the_deprecated_com_func():
    """The accumulating upstream CoM func points at the anchored replacement."""
    assert DEPRECATED_EVENT_FUNCS["randomize_rigid_body_com"] == ("randomize_rigid_body_com_abs")

    with pytest.raises(ValueError, match="deprecated func") as err:
        build_event_terms(
            [{"func": "randomize_rigid_body_com", "mode": "reset"}],
            mdp_module=_mdp(),
            term_cfg_cls=_Term,
            scene_entity_cfg_cls=_Entity,
        )

    assert "randomize_rigid_body_com_abs" in str(err.value)


class _AssetCfg:
    name = "prop"
    body_ids = slice(None)


class _Asset:
    """Engine-agnostic asset: numbers come from the fake profile."""

    num_bodies = 2


class _Scene:
    def __init__(self, items: dict, num_envs: int):
        self._items = dict(items)
        self.num_envs = num_envs

    def __getitem__(self, name: str):
        return self._items[name]


class _Env:
    def __init__(self, items: dict, *, num_envs: int = 3, engine_profile=None):
        self.scene = _Scene(items, num_envs)
        if engine_profile is not None:
            self.engine_profile = engine_profile


class _RecordingProfile:
    """Fake EngineProfile that records the CoM read/write calls."""

    def __init__(self, coms):
        self._coms = coms
        self.get_calls = []
        self.set_calls = []

    def get_body_com(self, asset):
        self.get_calls.append(asset)
        return self._coms

    def set_body_com(self, asset, coms, env_ids):
        self.set_calls.append((asset, coms, env_ids))


def test_randomize_com_abs_requires_an_engine_profile():
    """An env without an EngineProfile raises a teaching NotImplementedError."""
    env = _Env({"prop": _Asset()})

    with pytest.raises(NotImplementedError, match="EngineProfile") as err:
        randomize_rigid_body_com_abs(env, None, {"x": [0.0, 0.0]}, _AssetCfg())

    assert "get_body_com" in str(err.value)
    assert "set_body_com" in str(err.value)


def test_randomize_com_abs_reads_and_writes_through_the_profile():
    """CoM read/write go through the profile, no ``root_physx_view`` needed."""
    import torch

    coms = torch.ones(3, 2, 3)
    profile = _RecordingProfile(coms)
    asset = _Asset()
    env = _Env({"prop": asset}, num_envs=3, engine_profile=profile)
    env_ids = torch.tensor([0, 2])

    randomize_rigid_body_com_abs(env, env_ids, {"x": [0.1, 0.1]}, _AssetCfg())

    assert profile.get_calls == [asset]
    assert len(profile.set_calls) == 1
    set_asset, written, written_env_ids = profile.set_calls[0]
    assert set_asset is asset
    assert written.shape == (2, 2, 3)
    assert torch.equal(written_env_ids, env_ids)
    # default CoM (ones) + x offset on the selected envs only
    assert torch.allclose(written[0, :, 0], torch.full((2,), 1.1))
    assert torch.allclose(written[1, :, 0], torch.full((2,), 1.1))
    assert torch.allclose(written[..., 1:], torch.ones(2, 2, 2))
