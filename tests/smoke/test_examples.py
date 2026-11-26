"""Packaged example-resource smoke.

Proves the bundled MJCF travels with the installed ``unigym.examples`` package.
The manual-driver runtime behavior is covered by ``test_manual_driver.py``.
"""

from __future__ import annotations

from importlib.resources import files


def test_packaged_manual_driver_mjcf_exists():
    """The quickstart MJCF is a real file inside the installed package."""
    assert (files("unigym.examples") / "tiny_arm.xml").is_file()
