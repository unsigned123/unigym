"""Camera body mounts and BASE_BODY cleanup have no implicit robot defaults."""

from __future__ import annotations

import pytest


def test_camera_body_mount_requires_an_explicit_attach():
    """There is no shared pelvis default; attach (or a profile base) is required."""
    from unigym.backends.shared.mjfamily.cameras import helpers

    assert not hasattr(helpers, "DEFAULT_BODY_MOUNT")
    assert helpers.resolve_body_mount({"attach": "pelvis"}) == "pelvis"
    assert helpers.resolve_body_mount({}, "base") == "base"
    with pytest.raises(ValueError, match="no default mount"):
        helpers.resolve_body_mount({})
