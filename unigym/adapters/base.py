"""Adapter interface: the joint-addressed observation/action bridge.

An adapter consumes explicitly requested raw observation quantities and
returns one :class:`~unigym.runtime.actions.NamedAction` per joint-addressed
batch it emits.  There is no policy-vector contract: actions are addressed by
articulation + joint name, never by a flat positional slice and never by an
action-group label (groups are authoring/policy ownership metadata).

This module must only depend on torch (no isaaclab) so adapters are
unit-testable on a plain laptop python.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import torch

from .._typing import JsonValue
from ..env.obs import ObservationRegistry, ObsSnapshot
from ..runtime.actions import NamedAction


class AdapterEnv(Protocol):
    """The env attributes an adapter may read at the observation boundary."""

    num_envs: int
    device: str | torch.device
    obs: ObservationRegistry


class Adapter(ABC):
    """Joint-addressed adapter: explicit observations -> named joint targets.

    The adapter declares the joint order it wants via :attr:`joint_order`
    (a list of simulator entity names).  The env backend gathers robot data
    by name, so no explicit permutation matrices ever leak into adapter
    code.

    The ACTION boundary is joint-addressed: :meth:`postprocess_action`
    returns a ``Sequence[NamedAction]`` with one self-describing joint batch
    per emitted entry, each carrying its articulation, explicit
    ``joint_names`` and an explicit ``kind``.  The backend resolves names to
    simulator addresses -- see :mod:`unigym.runtime.actions`.  Group
    ownership is an authoring partition, never a runtime field.
    """

    generates_command: bool = False
    """True when the adapter produces its own command vector every step
    (e.g. hierarchical policies); collect.py then skips the command sampler
    and records :attr:`last_command` instead."""

    _unigym_name: str = ""
    """Registry identity stamped by ``build_adapter`` (the policies
    partition / joint-target validation resolve ownership by this name)."""

    rewindable: bool = False
    """Rewind superpower declaration (explicit class attribute —
    never hasattr).  True means the adapter implements ``on_rewind`` and can
    resync to a restored timeline (the rewind gate fail-fasts otherwise).
    Adapters with internal streaming state should also implement
    ``snapshot_internal``/``restore_internal`` for full determinism."""

    def on_rewind(self, restored_tick: int) -> None:
        """Resync to a restored timeline (rewind face).

        Called by the rewind controller AFTER ``restore_internal`` (when
        implemented): drop in-flight work -> hold, roll back event seq,
        resync sim-time accounting, append the rewind ledger marker.
        ``restored_tick`` is the engine tick the world was restored to
        (the next executed tick is ``restored_tick + 1``)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement on_rewind "
            f"(rewindable={self.rewindable!r}); the rewind face needs the "
            f"adapter resync hook"
        )

    def snapshot_internal(self) -> dict[str, JsonValue] | None:
        """Serialize the adapter's internal streaming state (rewind face).

        Returns a numpy/JSON-safe dict carried inside every engine
        snapshot; the counterpart :meth:`restore_internal` writes it back.
        Default: None (declared absence — the rewind then degrades to the
        on_rewind re-seed path and the warmup hole is disclosed)."""
        return None

    def restore_internal(self, state: dict[str, JsonValue]) -> None:
        """Restore a :meth:`snapshot_internal` payload (rewind face)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement restore_internal")

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Single-frame adapter observation dimension."""

    @property
    @abstractmethod
    def control_dt(self) -> float:
        """Adapter control period in seconds (e.g. 0.02 for 50 Hz)."""

    @abstractmethod
    def compute_obs(self, obs: ObsSnapshot, env: AdapterEnv) -> torch.Tensor:
        """Assemble the single-frame adapter observation from typed views.

        Includes scales and concatenation order.  May also update internal
        history buffers (called exactly once per control step, after
        env.step).  Read robot/contact/object/camera/site/command values
        from *obs* (``obs.robot("g1").joint_pos``, ``obs.command.value``).
        """

    def compute_obs_collect(
        self,
        obs_frame: torch.Tensor,
        obs: ObsSnapshot,
        env: AdapterEnv,
    ) -> torch.Tensor:
        """Dataset-facing observation frame (recorded as ``rec["obs"]``).

        Default: return *obs_frame* unchanged — record exactly the frame the
        adapter consumed (bitwise-identical to the reference behavior).

        Override to assemble a different dataset frame (e.g. camera frames
        from *obs* alongside proprioception) while keeping
        *compute_obs* as the compact adapter input.  Overriding adapters
        should declare their recording semantics; ``collect.py`` flags the
        override in ``meta["obs_collect_overridden"]``.

        WARNING: do NOT call ``compute_obs()`` here — it carries history
        side effects and is invoked exactly once per control step by the
        engine.  This hook receives the already-computed frame.
        """
        return obs_frame

    @abstractmethod
    def act(self, obs: torch.Tensor | None) -> torch.Tensor:
        """Adapter forward (with internal history/latent management).

        Args:
            obs: latest single-frame observation (``None`` on the very first
                step).  History-managing adapters may ignore it.

        Returns:
            Raw adapter action (adapter action space, NOT joint targets).
        """

    @abstractmethod
    def postprocess_action(self, raw: torch.Tensor) -> Sequence[NamedAction]:
        """Raw action -> one or more self-describing joint batches.

        Every returned :class:`~unigym.runtime.actions.NamedAction` carries
        its articulation, ``(N, dim)`` target values, the joint name of every
        column, and one explicit ``kind`` (``position`` / ``velocity`` /
        ``torque`` / ``effort`` / ``current``).  One entry covers one
        articulation and one kind; it may span several physical controls of
        that articulation as long as every addressed joint expects that
        kind.  Sequence order is not a layout; backends route by
        ``(articulation, joint)`` and resolve the names to simulator
        addresses.  Group ownership is an authoring partition, not a runtime
        field.
        """

    def named_action(
        self,
        values: torch.Tensor,
        *,
        articulation: str,
        joint_names: Sequence[str] | None = None,
        kind: str = "position",
    ) -> NamedAction:
        """Convenience: wrap *values* as a joint-addressed NamedAction.

        *joint_names* defaults to :attr:`joint_order`.
        """

        order = self.joint_order if joint_names is None else joint_names
        if order is None:
            raise RuntimeError(
                f"{type(self).__name__}.named_action requires joint_names or "
                f"joint_order; declare the adapter's joint order before "
                f"wrapping values"
            )
        if isinstance(order, (str, bytes)):
            raise ValueError(
                f"{type(self).__name__}.named_action joint_names must be a "
                f"sequence of joint names, not a bare string {order!r}"
            )
        return NamedAction(
            articulation=articulation,
            values=values,
            joint_names=tuple(order),
            kind=kind,
        )

    def on_reset(self, env_ids: torch.Tensor) -> None:
        """Called on episode reset for the given env ids (clear history etc.)."""

    def bind_env(self, env: AdapterEnv) -> None:
        """Called once by the env backend after perms are ready (hook for adapters
        that need the policy->isaaclab permutation)."""

    def extra_sinks(self, env: AdapterEnv, out_dir: str) -> list[Any]:
        """Adapter-provided diagnostics sinks (default: none).

        Mounted by collect_cli AHEAD of the standard recording/scoring
        chain (FanOutSink, construction order) so keys an adapter writes
        into ``bundle["record"]`` during its own on_step reach the
        standard sinks in the same step.  The engine stays
        adapter-agnostic: it never names adapters, it only asks.
        """
        # Any: adapter-provided sink objects are opaque drain plugins; the
        # collect chain composes them structurally (no shared sink base).
        return []

    # ------------------------------------------------------------ #
    # Language states (declarative command mapping — decouples
    # collect.py from adapter-specific command semantics)
    # ------------------------------------------------------------ #
    def get_command_meanings(self) -> dict[str, int] | None:
        """Declare command-vector semantic index mapping.

        Returns a dict like ``{"vx": 0, "vy": 1, "freq": 4, ...}`` mapping
        semantic names to indices into the command vector, or ``None`` when
        the adapter does not provide a semantic mapping (language generation
        will then rely solely on ``extra_info`` quantities).
        """
        return None

    def build_language_states(
        self,
        extra_info: dict[str, torch.Tensor],
        cmd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Assemble the per-env language *states* dict from env data + command.

        The default implementation uses :meth:`get_command_meanings` to map
        command indices, then copies standard motion quantities (vx, vy, wz,
        base_z) from *extra_info*.  Subclasses may override to perform
        adapter-specific inference (e.g. gait-name from phase/offset/bound).

        Args:
            extra_info: dict with per-env tensors; expected keys (all optional):
                ``"vx"``, ``"vy"``, ``"wz"``, ``"base_z"``.
            cmd: ``(N, dim)`` command tensor.
        """
        states: dict[str, torch.Tensor] = {}
        meanings = self.get_command_meanings()
        if meanings is not None:
            for name, idx in meanings.items():
                states[name] = cmd[..., idx]
        states.update({k: v for k, v in extra_info.items() if k in ("vx", "vy", "wz", "base_z")})
        return states

    # ------------------------------------------------------------ #
    # Ordering properties
    # ------------------------------------------------------------ #
    @property
    def joint_order(self) -> list[str] | None:
        """Joint names in the adapter's (policy) order; None -> use robot profile order."""
