"""Public package import smoke."""
from __future__ import annotations

import re

import unigym


def test_version_is_public_release_line():
    """The public core reports its own new version line."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", unigym.__version__)
    assert unigym.__version__ == "0.106.0"


def test_public_names_resolve():
    """Every advertised public name is importable through the lazy facade."""
    for name in unigym.__all__:
        assert getattr(unigym, name) is not None
