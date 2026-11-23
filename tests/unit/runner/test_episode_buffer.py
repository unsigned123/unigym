"""EpisodeBuffer owns per-env step slices, rewind marks and tick samples."""

from __future__ import annotations

import numpy as np
import torch

from unigym.runner.evaluation.buffer import EpisodeBuffer, TakenEpisode, TickSample


def _record(*, episode_ids: tuple[int, ...], x: list[list[float]]) -> dict:
    return {
        "episode_id": torch.tensor(episode_ids),
        "x": np.array(x),
    }


def test_append_step_slices_one_record_per_env():
    """Appending a vector record buffers exactly one slice per env."""
    buffer = EpisodeBuffer(2)
    buffer.append_step(_record(episode_ids=(3, 4), x=[[1.0, 2.0], [3.0, 4.0]]))

    first = buffer.take(0)
    second = buffer.take(1)

    assert len(first.steps) == 1
    assert int(first.steps[0]["episode_id"]) == 3
    assert np.array_equal(first.steps[0]["x"], [1.0, 2.0])
    assert len(second.steps) == 1
    assert int(second.steps[0]["episode_id"]) == 4
    assert np.array_equal(second.steps[0]["x"], [3.0, 4.0])


def test_take_empty_env_returns_empty_taken_episode():
    """Taking an env with no buffered step yields an empty, non-shared episode."""
    buffer = EpisodeBuffer(1)

    taken = buffer.take(0)

    assert isinstance(taken, TakenEpisode)
    assert taken.steps == []
    assert taken.rewinds == []
    assert taken.ticks == []


def test_take_clears_steps_marks_and_ticks():
    """take removes all three per-env histories and a second take is empty."""
    buffer = EpisodeBuffer(1)
    buffer.append_step(_record(episode_ids=(0,), x=[[1.0, 2.0]]))
    buffer.mark_rewound({"tick": 4})
    buffer.append_tick(
        0,
        "acme/task",
        TickSample(metrics={"m": np.array([1.0])}, fired=np.array(True)),
    )

    taken = buffer.take(0)

    assert len(taken.steps) == 1
    assert taken.rewinds == [{"tick": 4}]
    assert taken.ticks[0]["acme/task"].metrics["m"][0] == 1.0
    assert buffer.bufs[0] == []
    assert buffer.rewinds[0] == []
    assert buffer.ticks[0] == []
    assert buffer.take(0) == TakenEpisode()


def test_episode_stamp_is_first_non_empty_env_last_frame():
    """The stamp is the first non-empty env's last episode id, else None."""
    buffer = EpisodeBuffer(2)
    assert buffer.episode_stamp() is None

    buffer.append_step(_record(episode_ids=(7, 8), x=[[1.0], [2.0]]))
    assert buffer.episode_stamp() == 7

    buffer.take(0)
    assert buffer.episode_stamp() == 8


def test_snapshot_is_isolated_from_live_buffer():
    """Mutating a snapshot leaves the live steps, marks and ticks untouched."""
    buffer = EpisodeBuffer(1)
    buffer.append_step(_record(episode_ids=(0,), x=[[1.0, 2.0]]))
    buffer.mark_rewound({"tick": 4})
    buffer.append_tick(
        0,
        "acme/task",
        TickSample(metrics={"m": np.array([1.0])}, fired=np.array(True)),
    )

    state = buffer.snapshot()
    state["bufs"][0][0]["x"][0] = 99.0
    state["marks"][0][0]["tick"] = 99
    state["ticks"][0][0]["acme/task"].metrics["m"][0] = 99.0
    state["bufs"][0].append({"extra": 1})

    assert np.array_equal(buffer.bufs[0][0]["x"], [1.0, 2.0])
    assert buffer.rewinds[0] == [{"tick": 4}]
    assert buffer.ticks[0][0]["acme/task"].metrics["m"][0] == 1.0
    assert len(buffer.bufs[0]) == 1


def test_restore_round_trips_steps_marks_and_ticks_in_place():
    """Restore writes the snapshot back through the same buffer object."""
    buffer = EpisodeBuffer(1)
    buffer.append_step(_record(episode_ids=(2,), x=[[1.0, 2.0]]))
    buffer.mark_rewound({"tick": 3})
    buffer.append_tick(
        0,
        "acme/task",
        TickSample(metrics={"m": np.array([1.0])}, fired=np.array(True)),
    )
    state = buffer.snapshot()
    bufs_ref = buffer.bufs

    buffer.take(0)
    assert buffer.bufs[0] == []

    buffer.restore(state)

    assert buffer.bufs is bufs_ref
    assert buffer.episode_stamp() == 2
    assert len(buffer.bufs[0]) == 1
    assert np.array_equal(buffer.bufs[0][0]["x"], [1.0, 2.0])
    assert buffer.rewinds[0] == [{"tick": 3}]
    assert buffer.ticks[0][0]["acme/task"].metrics["m"][0] == 1.0
    assert bool(buffer.ticks[0][0]["acme/task"].fired)


def test_dump_writes_steps_and_tick_traces(tmp_path):
    """dump writes one npz with per-env step arrays and tick samples."""
    buffer = EpisodeBuffer(2)
    buffer.append_step(_record(episode_ids=(0, 1), x=[[1.0, 2.0], [3.0, 4.0]]))
    buffer.append_tick(
        0,
        "acme/task",
        TickSample(metrics={"m": np.array([1.0])}, fired=np.array(True)),
    )

    paths = buffer.dump(str(tmp_path), "attempt")

    assert paths == [str(tmp_path / "attempt.npz")]
    with np.load(paths[0]) as data:
        assert set(data.files) == {
            "env0/episode_id",
            "env0/x",
            "env1/episode_id",
            "env1/x",
            "env0/ticks/0/acme/task/m",
            "env0/ticks/0/acme/task/fired",
        }
        assert np.array_equal(data["env1/x"], [[3.0, 4.0]])
        assert data["env0/ticks/0/acme/task/m"][0] == 1.0
        assert bool(data["env0/ticks/0/acme/task/fired"])
