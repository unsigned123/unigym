"""Run construction face (v097): spec -> instance.

``RunSpec`` / ``build_spec`` are data-only and import cheaply.  The live
construction modules (``build`` / ``instance``) are loaded lazily so
importing ``unigym.run`` for a spec does not drag in backends/adapters.
"""
from __future__ import annotations

from .spec import RunSpec, build_spec

__all__ = ["RunSpec", "RunInstance", "build_spec", "build_instance"]


def __getattr__(name: str):
    if name == "RunInstance":
        from .instance import RunInstance
        return RunInstance
    if name == "build_instance":
        from .build import build_instance
        return build_instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
