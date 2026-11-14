"""Neutral artifact writers: composed text and canonical describe files."""

from __future__ import annotations

import json

import pytest

from unigym.scenes.artifacts import write_composed_text, write_describe


def test_write_composed_text_writes_compose_txt_not_composed_xml(tmp_path):
    """The composed text lands in compose.txt and never aliases composed.xml."""
    text = '{"scene_name": "bench", "fields": []}\n'

    path = write_composed_text(tmp_path, text)

    assert path == str(tmp_path / "compose.txt")
    assert (tmp_path / "compose.txt").read_text(encoding="utf-8") == text
    assert not (tmp_path / "composed.xml").exists()


def test_write_composed_text_rejects_non_text_payloads(tmp_path):
    """A non-string composed payload fails fast with the actual type."""
    payload = b"not text"

    with pytest.raises(TypeError, match="composed scene text must be a string, got bytes"):
        write_composed_text(tmp_path, payload)


def test_write_describe_serializes_canonical_sorted_json(tmp_path):
    """describe.json is sorted, indented and newline-terminated."""
    describe = {"ext": {"scene": {"fields": []}}, "digest_version": "unigym.scene.asset/1"}

    path = write_describe(tmp_path, describe)

    text = (tmp_path / "describe.json").read_text(encoding="utf-8")
    assert path == str(tmp_path / "describe.json")
    assert text.endswith("\n")
    assert text.index('"digest_version"') < text.index('"ext"')
    assert json.loads(text) == describe


def test_write_describe_is_byte_identical_for_the_same_mapping(tmp_path):
    """Repeated writes of one mapping produce identical bytes."""
    describe = {"instances": [{"id": "cup", "kind": "static_rigid"}]}
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_describe(first, describe)
    write_describe(second, describe)

    assert (first / "describe.json").read_bytes() == (second / "describe.json").read_bytes()
