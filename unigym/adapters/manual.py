"""ManualAdapter: the "user supplies the actions" Adapter.

This is the ``adapter=None`` path at the session layer: no policy, no obs
requests, no action postprocessing.  It is used by
:class:`unigym.runner.external.ExternalRunner`, whose ``step(actions)``
bypasses :meth:`Adapter.act` entirely -- calling ``act`` is a
programming error and fails loudly.

``compute_obs`` has no policy frame to assemble (a manual driver may feed
arbitrary named actions), so it returns the first tensor found in the live
``env.obs_cache``; an empty cache yields a zero-width ``(N, obs_dim)``
tensor.  The cache itself is therefore already the observation surface for
manual drivers.
"""

from __future__ import annotations

import torch

from ..runtime.actions import NamedAction
from .base import Adapter


class ManualAdapter(Adapter):
    """Pass-through adapter for user-driven action loops.

    Args:
        control_dt: policy control period in seconds (metadata only; the
            manual driver owns the clock).
        obs_dim: width of the fallback zero observation when the env's
            ``obs_cache`` is empty (default 0).
    """

    generates_command = False

    def __init__(
        self,
        control_dt: float = 0.02,
        obs_dim: int = 0,
        num_envs: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        # num_envs/device are the registry-injected kwargs (build_adapter
        # always supplies them); a manual adapter ignores them -- there is
        # no policy to bind or joint map to use.
        self._control_dt = float(control_dt)
        self._obs_dim = int(obs_dim)

    def get_obs_requests(self) -> list:
        """Manual drivers need no backend-filled raw quantities."""
        return []

    @property
    def obs_dim(self) -> int:
        """Fallback observation width (0 unless overridden)."""
        return self._obs_dim

    @property
    def control_dt(self) -> float:
        """Policy control period in seconds (metadata only)."""
        return self._control_dt

    def compute_obs(self, obs_cache: dict, cmd: torch.Tensor, env) -> torch.Tensor:
        """Return the cache's first tensor (empty cache -> ``(N, obs_dim)``).

        Documented choice: the raw ``obs_cache`` IS the observation surface
        for manual drivers, so no adapter-side assembly happens here.
        """
        for value in (obs_cache or {}).values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value
        num_envs = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        return torch.zeros(num_envs, self._obs_dim, device=device)

    def act(self, obs: torch.Tensor | None) -> torch.Tensor:
        """Never legal: ExternalRunner passes actions straight through."""
        raise RuntimeError(
            "ManualAdapter.act must never be called; drive "
            "ExternalRunner.step(actions) / SyncRunner.tick(actions_override) "
            "instead"
        )

    def postprocess_action(self, raw: torch.Tensor) -> dict[str, NamedAction]:
        """Never legal: external drivers bypass adapter postprocessing.

        The contract is explicit action dicts handed straight to
        ``ExternalRunner.step`` / ``SyncRunner.tick(actions_override=...)``;
        silently returning a tensor where the base promises named groups
        would hide a caller bug.
        """
        raise NotImplementedError(
            "ManualAdapter.postprocess_action must never be called; pass "
            "dict[str, NamedAction] to ExternalRunner.step(actions) / "
            "SyncRunner.tick(actions_override) instead"
        )

    def on_reset(self, env_ids) -> None:
        """No history to clear."""

    def bind_env(self, env) -> None:
        """No policy/env binding needed."""

    def extra_sinks(self, env, out_dir) -> list:
        """No adapter diagnostics."""
        return []
