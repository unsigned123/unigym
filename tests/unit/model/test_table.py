"""Table indexing, lookup failure, duplicate identity and empty collections."""

from __future__ import annotations

import pytest

from unigym.model import Body, Table


def test_table_lookup_resolves_name_then_qualname_and_position():
    """A table row is addressable by name, qualname and position."""
    torso = Body(name="torso", qualname="robot/torso")
    arm = Body(name="arm", qualname="robot/arm")
    table = Table([torso, arm])

    assert table["torso"] is torso
    assert table["robot/torso"] is torso
    assert table[1] is arm
    assert "arm" in table
    assert "robot/arm" in table
    assert table.names() == ["torso", "arm"]


def test_table_missing_lookup_raises_keyerror_and_get_returns_default():
    """An absent name is a KeyError; ``get`` answers an explicit default."""
    table = Table([Body(name="torso")])

    with pytest.raises(KeyError, match="missing"):
        table["missing"]
    assert table.get("missing") is None
    assert table.get("missing", "fallback") == "fallback"


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([Body(name="joint"), Body(name="joint")], "duplicated"),
        (
            [Body(name="left", qualname="arm/elbow"), Body(name="right", qualname="arm/elbow")],
            "qualname 'arm/elbow' is duplicated",
        ),
    ],
)
def test_table_rejects_duplicate_name_or_qualname(rows, match):
    """Duplicate row identity is rejected instead of silently overwritten."""
    with pytest.raises(ValueError, match=match):
        Table(rows)


def test_empty_table_has_no_rows_and_never_resolves_a_key():
    """An empty collection is a valid table with no addressable identity."""
    table: Table[Body] = Table()

    assert len(table) == 0
    assert list(table) == []
    assert table.names() == []
    assert table.keys() == ()
    assert "anything" not in table
    assert table.get("anything") is None
