"""Metrics writer reads its step clock from the caller, never from a bundle."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests._support.fakes.step_clock import StepClockEnv
from unigym.runtime.bridge import METRICS_OFF, METRICS_SIZE, Bridge, make_run_id
from unigym.runtime.metrics import MetricsWriter, read_metrics_block


def _writer() -> tuple[MetricsWriter, bytearray]:
    buf = bytearray(METRICS_OFF + METRICS_SIZE)
    return MetricsWriter(buf), buf


def test_note_tick_derives_sim_time_and_fps_from_the_step_clock() -> None:
    """The supplied step clock drives sim time and the decimated sim fps."""
    writer, buf = _writer()
    clock = StepClockEnv(step_dt=0.02, decimation=4)

    writer.note_tick(0, clock)
    writer.note_tick(1, clock)

    block = read_metrics_block(buf)
    assert block["sim_time"] == pytest.approx(0.02)
    assert block["fps_policy"] > 0
    assert block["fps_sim"] == pytest.approx(block["fps_policy"] * 4)


def test_note_tick_without_a_step_clock_publishes_no_sim_time() -> None:
    """An absent step clock keeps the historical zero sim-time scalars."""
    writer, buf = _writer()

    writer.note_tick(3)

    block = read_metrics_block(buf)
    assert block["sim_time"] == 0.0
    assert block["fps_sim"] == 0.0


def test_note_episode_derives_steps_from_the_supplied_step_clock() -> None:
    """Episode step counts divide the episode time by the supplied period."""
    writer, _ = _writer()
    bundle = {"record": {"time_in_episode": torch.tensor([0.4])}}

    writer.note_episode(torch.tensor([0]), bundle, env=StepClockEnv(step_dt=0.02))

    assert writer.document()["episode"]["last"]["steps"] == 20


def test_note_episode_without_a_step_clock_has_no_step_count() -> None:
    """An absent step clock keeps the historical null step count."""
    writer, _ = _writer()
    bundle = {"record": {"time_in_episode": torch.tensor([0.4])}}

    writer.note_episode(torch.tensor([0]), bundle)

    assert writer.document()["episode"]["last"]["steps"] is None


def test_note_episode_ignores_a_bundle_env() -> None:
    """Only the caller-supplied clock counts, never a bundle env."""
    writer, _ = _writer()
    bundle = {
        "record": {"time_in_episode": torch.tensor([0.4])},
        "env": StepClockEnv(step_dt=1.0),
    }

    writer.note_episode(torch.tensor([0]), bundle, env=StepClockEnv(step_dt=0.02))

    assert writer.document()["episode"]["last"]["steps"] == 20


def test_bridge_metrics_read_the_construction_clock(tmp_path: Path) -> None:
    """The Bridge feeds its construction clock to ticks and episodes."""
    clock = StepClockEnv(step_dt=0.02, decimation=4)
    bundle_env = StepClockEnv(step_dt=1.0, decimation=1)
    bridge = Bridge(make_run_id("metrics"), None, state_dir=str(tmp_path), env=clock)
    try:
        bridge.on_step(0, {})
        bridge.on_step(1, {})
        bridge.on_episode_end(
            torch.tensor([0]),
            {"record": {"time_in_episode": torch.tensor([0.4])}, "env": bundle_env},
        )

        block = read_metrics_block(bridge.buf)
        assert block["sim_time"] == pytest.approx(0.02)
        assert block["fps_sim"] == pytest.approx(block["fps_policy"] * 4)
        document = block["document"]
        assert document["episode"]["last"]["steps"] == 20

        bridge.on_step(2, {})  # the status block publishes the episode count
        assert bridge.read_status()["episodes"] == 1
    finally:
        bridge.close()
