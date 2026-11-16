"""The additive ``pack_runtime=`` keyword on the run construction surface."""

from __future__ import annotations

from pathlib import Path

from unigym._typing import JsonValue
from unigym.pack_api import discover
from unigym.run.api import Run, load


def _cfg() -> dict[str, JsonValue]:
    """A small valid config that needs no runtime."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
    }


def test_constructor_accepts_the_keyword_without_changing_the_config() -> None:
    """``pack_runtime=`` is an additive constructor keyword."""
    cfg = _cfg()

    run = Run(cfg, pack_runtime=discover([]))

    assert run.cfg["run_name"] == "r"
    assert run.cfg["world_base"] == {"mjcf": "model.xml"}


def test_constructor_keeps_the_old_form_when_the_keyword_is_omitted() -> None:
    """The keyword defaults to ``None``; the old call shape is unchanged."""
    cfg = _cfg()

    run = Run(cfg)

    assert run.cfg["run_name"] == "r"
    assert run.cfg["world_base"] == {"mjcf": "model.xml"}


def test_from_config_and_load_accept_the_keyword(tmp_path: Path) -> None:
    """Both lazy loaders share the additive keyword."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: r",
                "seed: 0",
                "env: {backend: mujoco, num_envs: 1}",
                "world_base: {mjcf: model.xml}",
                "collect: {steps: 1}",
                "articulations: [{name: arm, joints: [{name: shoulder}]}]",
                "action_spec:",
                "  - {name: arm, articulation: arm, joints: [shoulder],",
                "     target: position, role: body}",
                "policies:",
                "  - {adapter: manual, groups: [[arm, arm]]}",
            ]
        ),
        encoding="utf-8",
    )
    runtime = discover([])

    from_config = Run.from_config(path, pack_runtime=runtime)
    loaded = load(path, pack_runtime=runtime)

    assert from_config.cfg["run_name"] == "r"
    assert loaded.cfg["run_name"] == "r"
