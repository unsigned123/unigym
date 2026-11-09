"""Compatibility shim for the isaaclab BallProp handler.

The handler lives in ``unigym.backends.isaaclab2.props.handlers.ball``;
the public name stays importable here for existing users/tests.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy compat re-export of the isaaclab BallProp handler."""
    if name in ("BallProp", "_build_sphere_spawn"):
        from importlib import import_module

        handler = import_module("unigym.backends.isaaclab2.props.handlers.ball")
        return getattr(handler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
