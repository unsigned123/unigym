"""IsaacLab 3.x backend skeleton (EA) — trait-declared stub.

The THIRD equal backend (design §R: isaaclab2/isaaclab3 are two different
backends, not one backend with a translation shim).  This package is the EA
smoke-level placeholder: registered + selectable (the version probe routes
``backend: isaaclab`` here when isaaclab >= 3.0 is installed), with an
EMPTY trait set — every capability is declared absent until implemented
(backends/isaaclab3/README.md tracks the gap list).

Constructing a run on this backend fails fast with a stub error; the
production default remains isaaclab2.  No capability implementation, no
translation shim: il3-specific conventions (XYZW data views, write_*_index
splits, ProxyArray, renamed contact-force properties — design §2) get
absorbed by IMPLEMENTING the capability protocols here, with repo-internal
wxyz preserved through ``unigym.core.quat.from_isaaclab_data`` at the data
boundary.
"""
from __future__ import annotations

from unigym.backends.registry import ISAACLAB3_TRAITS, BackendInfo

BACKEND_INFO = BackendInfo("isaaclab3", (3, 0), ISAACLAB3_TRAITS)

__all__ = ["BACKEND_INFO", "Isaaclab3Env"]


class Isaaclab3Env:
    """EA stub: any construction attempt fails fast (nothing implemented)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "backend 'isaaclab3' is an EA skeleton — no capability is "
            "implemented yet (see unigym/backends/isaaclab3/README.md); "
            "hint: set env.backend: isaaclab2 (the production default)")
