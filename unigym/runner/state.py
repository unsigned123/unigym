"""Request-free, backend-neutral read of the current step state.

The :class:`~unigym.env.backends.base.EnvBackend` contract exposes ``extra``
only through ``step``; the reset path nevertheless needs the 17-key state
record after the spawn/reset chain.  This module owns that one best-effort
seam: it asks the backend's own state builder when it has one (mujoco /
isaaclab) and returns ``{}`` plus the bookkeeping defaults for minimal
doubles.  The ``_extra`` fallback covers in-process test doubles.

The function is the single owner of the branch order and the defaults; the
Gym facades and the experimental simulation facade call it instead of
duplicating those decisions.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

__all__ = ["read_extra"]


def read_extra(env: object) -> dict[str, object]:
    """Best-effort current 17-key extra record (private backend seam).

    Args:
        env: An env backend (or a minimal double) exposing ``device`` and
            optionally ``_build_extra`` / ``_read_state`` / ``_extra``.

    Returns:
        The builder's mapping with the ``reset_env_ids`` / ``should_reset``
        defaults filled in.

    Raises:
        TypeError: The backend's extra builder returned a non-mapping.
    """
    build = getattr(env, "_build_extra", None)
    raw: object
    if callable(build):
        read = getattr(env, "_read_state", None)
        raw = build(read()) if callable(read) else build()
    elif callable(getattr(env, "_extra", None)):
        extra_reader = getattr(env, "_extra")
        raw = extra_reader()
    else:
        raw = {}

    if not isinstance(raw, Mapping):
        raise TypeError(
            f"read_extra: the backend extra builder returned {type(raw).__name__}, "
            f"expected a mapping"
        )
    extra: dict[str, object] = dict(raw)
    device = getattr(env, "device")
    extra.setdefault("reset_env_ids", torch.zeros(0, dtype=torch.int64, device=device))
    extra.setdefault("should_reset", False)
    return extra
