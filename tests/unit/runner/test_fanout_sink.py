"""FanOutSink: explicit stable child names and the unchanged unnamed form."""

from __future__ import annotations

import pytest

from unigym.runner.sinks import FanOutSink
from tests._support.fakes.external_sink import RecordingSink


def test_named_fanout_uses_the_declared_summary_keys() -> None:
    collect = RecordingSink("collect")
    bench = RecordingSink("bench")

    fanout = FanOutSink(named=(("collect", collect), ("bench", bench)))

    assert fanout.sinks == [collect, bench]
    assert fanout.on_run_end() == {
        "collect": {"sink": "RecordingSink"},
        "bench": {"sink": "RecordingSink"},
    }


def test_named_fanout_allows_two_children_of_the_same_class() -> None:
    first = RecordingSink("first")
    second = RecordingSink("second")

    fanout = FanOutSink(named=(("first", first), ("second", second)))

    assert fanout.on_run_end() == {
        "first": {"sink": "RecordingSink"},
        "second": {"sink": "RecordingSink"},
    }


@pytest.mark.parametrize(
    ("named", "match"),
    [
        ((("dup", RecordingSink()), ("dup", RecordingSink())), "duplicate child name"),
        ((("", RecordingSink()),), "non-empty strings"),
        (((1, RecordingSink()),), "non-empty strings"),
    ],
    ids=["duplicate", "empty-name", "non-string-name"],
)
def test_named_fanout_rejects_invalid_names(named, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FanOutSink(named=named)


def test_named_fanout_rejects_a_non_pair_entry() -> None:
    with pytest.raises(TypeError, match=r"\(name, sink\) pairs"):
        FanOutSink(named=(("only-a-name",),))


def test_named_and_positional_construction_are_mutually_exclusive() -> None:
    with pytest.raises(TypeError, match="either positional sinks or named"):
        FanOutSink(RecordingSink(), named=(("bench", RecordingSink()),))


def test_unnamed_fanout_keeps_the_class_name_summary_shape() -> None:
    fanout = FanOutSink(RecordingSink("a"))

    assert fanout.on_run_end() == {"RecordingSink": {"sink": "RecordingSink"}}


def test_unnamed_fanout_still_rejects_a_duplicate_class_key() -> None:
    fanout = FanOutSink(RecordingSink("a"), RecordingSink("b"))

    with pytest.raises(ValueError, match="duplicate summary key 'RecordingSink'"):
        fanout.on_run_end()


def test_named_fanout_reports_the_declared_child_in_terminal_collisions() -> None:
    first = RecordingSink("first")
    second = RecordingSink("second")
    first.drain_terminations = lambda: {0: object()}  # type: ignore[attr-defined]
    second.drain_terminations = lambda: {0: object()}  # type: ignore[attr-defined]

    fanout = FanOutSink(named=(("first", first), ("second", second)))

    with pytest.raises(ValueError, match=r"child 1 \(second\)"):
        fanout.drain_terminations()
