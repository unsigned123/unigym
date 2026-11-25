"""CommandSampler interface (torch-only, unit-testable)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from unigym._typing import JsonValue
from unigym.core.episode_result import EpisodeResultView

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.env.backends.base import EnvBackend
    from unigym.values import ValueView


class CommandSampler(ABC):
    """Per-env command vector sampler with optional timed resampling.

    Holds a buffer ``self.cmd`` of shape (num_envs, dim). ``reset`` resamples
    the given envs; ``update`` is called once per policy step with the step
    dt and may resample envs whose timer expired.

    Lifecycle hooks: :meth:`bind_env` and :meth:`bind_values` run exactly
    once at the live-instance build seam (before any runner exists);
    :meth:`on_episode_end` runs at every due episode boundary, after the
    sinks are notified and before ``reset``.  All default to no-ops so
    built-in samplers stay unchanged.

    Rewind capability is explicit (never ``hasattr``): ``rewind_inert = True``
    means the sampler carries no external state (constants only, e.g.
    :class:`~unigym.commands.fixed.FixedSampler`); ``rewind_capable = True``
    means it implements :meth:`rewind_state` and :meth:`apply_rewind_state`.
    A sampler with neither is non-rewindable and the rewind gate refuses it.
    """

    rewind_inert: bool = False
    rewind_capable: bool = False

    def __init__(self, dim: int, device, num_envs: int = 1):
        self.dim = dim
        self.device = device
        self.num_envs = num_envs
        self.cmd = torch.zeros(num_envs, dim, dtype=torch.float32, device=device)

    @abstractmethod
    def reset(self, env_ids=None) -> torch.Tensor:
        """(Re)sample commands for env_ids (None = all) and return the buffer."""

    @abstractmethod
    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        """Advance timers by step_dt, resample expired envs, return the buffer."""

    def bind_env(self, env: EnvBackend) -> None:
        """Bind the live env once at the instance build seam (default: no-op).

        ``build_instance`` calls this exactly once per live run instance,
        before any runner exists; ``SyncRunner`` never binds.  The env is a
        read-only view (``env.obs`` / ``env.num_envs`` / ``env.device``); a
        sampler implements multi-stage commands itself.  It is not serialized,
        so a restored sampler must not depend on having been rebound.
        """
        return None

    def bind_values(self, values: ValueView) -> None:
        """Bind the run's live value store once (default: no-op).

        ``build_instance`` calls this exactly once per live run instance,
        immediately after :meth:`bind_env` and before any runner exists.
        A sampler that consumes values stores the read view (for example
        ``self.values = values``) and interprets names/dtypes itself; core
        performs no value-to-command mapping.  It is not replayed on restore,
        so a restored sampler must not depend on having been rebound.
        """
        return None

    def on_episode_end(
        self,
        env_ids: torch.Tensor,
        results: Mapping[int, tuple[EpisodeResultView, ...]],
    ) -> None:
        """Observe completed-episode results (default: no-op).

        ``env_ids`` are the due envs; ``results[env_id]`` is a read-only tuple
        in result declaration order.  Called after the sinks' ``on_episode_end``
        and before ``reset``, so a sampler can fold prior-episode results into
        the next episode's command.  A run whose sink publishes no results
        passes an empty mapping.
        """
        return None

    def rewind_state(self) -> dict[str, JsonValue]:
        """JSON/numpy-safe sampler state used by checkpoints.

        ``self.cmd`` is part of the sampler state.  Values must not contain
        the env, task objects or live tensors.  Only called when the sampler
        declares ``rewind_capable = True``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rewind_state(); set "
            "rewind_capable = True only when both rewind_state() and "
            "apply_rewind_state() are implemented"
        )

    def apply_rewind_state(self, state: dict[str, JsonValue]) -> None:
        """Restore a state returned by :meth:`rewind_state` (in place)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement apply_rewind_state(); "
            "set rewind_capable = True only when both rewind_state() and "
            "apply_rewind_state() are implemented"
        )

    def _resolve_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        return env_ids
