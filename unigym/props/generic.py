"""Compatibility shim for the isaaclab GenericProp handler.

The handler lives in ``unigym.backends.isaaclab2.props.handlers.generic``;
the public name stays importable here for existing users/tests.
"""
from __future__ import annotations


def __getattr__(name: str):
    """Lazy compat re-export of the isaaclab GenericProp handler."""
    if name in ("GenericProp", "_static_init_state"):
        from importlib import import_module

        handler = import_module(
            "unigym.backends.isaaclab2.props.handlers.generic")
        return getattr(handler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
