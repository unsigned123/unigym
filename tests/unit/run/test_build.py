"""The run boundary builds one AssetPlan for packs and provider-only scenes."""

from __future__ import annotations

import pytest

from unigym.packs import PackError
from unigym.packs.runtime import discover
from unigym.run.build import _boundary_asset_plan


def test_scene_provider_without_a_pack_runtime_gets_an_empty_plan(tmp_path) -> None:
    """A routed scene provider still gets a boundary: absolute refs resolve."""
    plan = _boundary_asset_plan(None, {"params": {"scene_provider": "_inline_1"}})

    assert plan is not None
    absolute = str(tmp_path / "cup.usd")
    assert plan.resolve(absolute) == absolute


def test_an_empty_plan_fails_a_pack_reference_with_pack_error() -> None:
    """Without an owning pack, a ``pack:`` reference fails honestly."""
    plan = _boundary_asset_plan(None, {"params": {"scene_provider": "_inline_1"}})

    assert plan is not None
    with pytest.raises(PackError):
        plan.resolve("pack:acme/assets/cup.usd")


def test_run_without_a_pack_runtime_or_scene_provider_gets_no_plan() -> None:
    """No pack runtime and no routed scene leaves the plan absent."""
    assert _boundary_asset_plan(None, {"num_envs": 1}) is None
    assert _boundary_asset_plan(None, {"params": {}}) is None


def test_pack_runtime_owns_the_boundary_even_without_a_scene_provider() -> None:
    """The explicit pack runtime wins over the empty provider boundary."""
    runtime = discover([])

    plan = _boundary_asset_plan(runtime, {})

    assert plan is not None
    assert plan.context.runtime is runtime
