"""The backend asset-format preference list is validated at construction."""

from __future__ import annotations

import pytest

from unigym.backends.registry import BackendInfo


def _info(*formats: str) -> BackendInfo:
    return BackendInfo("fixture", (0, 0), frozenset(), asset_formats=formats)


def test_backend_info_keeps_the_declared_preference_order():
    """The backend owns the ordered tuple and exposes it unchanged."""
    assert _info("usd", "urdf").asset_formats == ("usd", "urdf")


@pytest.mark.parametrize(
    ("formats", "match"),
    [
        ((), "at least one asset format"),
        (("usd", "glb"), r"unknown asset format\(s\) \['glb'\]"),
        (("urdf", "usd", "urdf"), "repeats an asset format"),
    ],
    ids=["empty", "unknown", "duplicate"],
)
def test_backend_info_rejects_a_malformed_asset_format_list(formats, match):
    """A backend cannot declare an empty, unknown or duplicated format list."""
    with pytest.raises(ValueError, match=match):
        _info(*formats)
