"""Adapter interface: the group-addressed observation/action bridge.

An adapter consumes explicitly requested raw observation quantities and
returns one :class:`~unigym.runtime.actions.NamedAction` per control group it
owns (the group names come from the run's ``controls[]`` partition).  There
is no policy-vector contract: actions are addressed by control group + joint
name, never by a flat positional slice.

This module must only depend on torch (no isaaclab) so adapters are
unit-testable on a plain laptop python.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Protocol, TypedDict, Unpack

import torch

from .._typing import JsonValue
from ..runtime.actions import NamedAction


class AdapterEnv(Protocol):
    """The env attributes an adapter may read at the observation boundary."""

    num_envs: int
    device: str | torch.device
    obs_cache: dict[str, torch.Tensor]


class _RawObsChanges(TypedDict, total=False):
    """Keyword payload accepted by :meth:`RawObsRequest._replace`."""

    key: str
    dim: int
    source: str
    joint_names: tuple[str, ...] | None
    articulation: str | None

#: robot-state keys that are scoped to ONE articulation.  Kept in sync with
#: :data:`unigym.env.obs_catalog.PER_ARTICULATION_KEYS`; the import is local
#: to keep this module torch-only at import time.
_ROBOT_STATE_KEYS = frozenset({
    "gravity_b", "joint_pos_rel", "joint_vel", "applied_torque",
    "prev_action", "command",
    "base_yaw", "base_pos_w", "base_quat_w", "base_ang_vel_b",
    "foot_contact_force", "base_lin_vel_b", "foot_contact",
    "foot_contact_force_z", "foot_pos_z",
})
#: joint-vector keys whose request MUST carry an explicit ``joint_names``.
_JOINT_VECTOR_KEYS = frozenset({"joint_pos_rel", "joint_vel",
                                "applied_torque"})


@dataclass(frozen=True)
class RawObsRequest:
    """Declare one raw quantity the adapter needs in ``obs_cache``.

    Attributes:
        key: cache key, e.g. "joint_pos_rel" / "joint_vel" /
            "base_pos_w" / "command" / "prev_action".  The env backend only
            fills requested keys.
        dim: last-dim size of the requested tensor (for validation).
        source: where it comes from -- "robot" | "contact_sensor" | "computed"
            | "adapter" | "sampler" | "prop" | "camera".
        joint_names: the joint name of every element, REQUIRED for the
            joint-vector keys ("joint_pos_rel" / "joint_vel" /
            "applied_torque").  The catalog publishes the FULL articulation
            joint set and any NON-EMPTY duplicate-free SUBSET of those names
            is legal, in ANY order — the request order IS the served order.
            ``None`` for every other key.
        articulation: the articulation name for a bare per-articulation key.
            REQUIRED for every robot-state key (there is no env-side
            controller inference).  Global keys (``cam/``,
            ``contact/``, ``object/``, ``site/``, props) keep ``None``.  A
            pre-namespaced key (``articulation/<name>/...``) needs none.
    """

    key: str
    dim: int
    source: str
    joint_names: tuple[str, ...] | None = None
    articulation: str | None = None

    def __post_init__(self):
        key = str(self.key)
        if key in _ROBOT_STATE_KEYS:
            if not self.articulation:
                raise ValueError(
                    f"RawObsRequest({key!r}) is articulation-scoped but "
                    f"declares no articulation; name it explicitly "
                    f"(e.g. articulation='g1')")
        if key in _JOINT_VECTOR_KEYS:
            names = tuple(self.joint_names or ())
            if not names:
                raise ValueError(
                    f"RawObsRequest({key!r}) addresses joints but carries "
                    f"no joint_names; joint-vector requests must be "
                    f"explicit")
            if len(set(names)) != len(names):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"RawObsRequest({key!r}) has duplicate joint names "
                    f"{dupes}")

    def _replace(self, **changes: Unpack[_RawObsChanges]) -> RawObsRequest:
        """NamedTuple-compatible update helper (canonicalization)."""
        return replace(self, **changes)


class ImageObsRequest(NamedTuple):
    """Declare one image-like quantity the adapter needs in ``obs_cache``.

    Division of labor vs :class:`RawObsRequest`: RawObsRequest covers vector
    quantities validated by LAST-DIM size (``dim`` check); ImageObsRequest
    covers camera frames validated by FULL SHAPE + dtype (``shape`` check,
    batch axis excluded).  Image keys are served by the
    ``unigym.cameras`` CameraManager (built from the yaml ``cameras:``
    section); keys follow the ``cam/<name>/<rgb|depth>`` key space, plus the
    ``depth_image`` / ``depth_image_raw`` aliases provisioned by the
    sugar shim (docs/contracts/camera_contract.md).

    Attributes:
        key: cache key, e.g. "cam/ego/rgb" or "cam/ego/depth".
        shape: per-env frame shape WITHOUT the batch axis, e.g. (3, 224, 224)
            for RGB or (1, 60, 106) for metric depth (channel axis included).
        dtype: expected torch dtype. Raw-value caching (owner ruling): uint8
            for RGB, float32 metric depth in meters; normalization stays with
            the adapter.
        source: documentation string, typically "camera".
        camera: camera (rig) name serving this key (e.g. "ego").
    """

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    source: str
    camera: str


class Adapter(ABC):
    """Group-addressed adapter: explicit observations -> named control targets.

    The adapter declares the joint/foot order it wants via
    :attr:`joint_order` / :attr:`foot_order` (lists of simulator entity
    names).  The env backend gathers robot data by name, so no explicit
    permutation matrices ever leak into adapter code.

    The ACTION boundary is group-addressed: :meth:`postprocess_action`
    returns ``dict[str, NamedAction]`` keyed by the control group names the
    run's ``controls[]`` partition assigns to this adapter, each carrying
    explicit ``joint_names`` and an explicit ``kind``.  The backend resolves
    names to simulator addresses — see :mod:`unigym.runtime.actions`.
    """

    generates_command: bool = False
    """True when the adapter produces its own command vector every step
    (e.g. hierarchical policies); collect.py then skips the command sampler
    and records :attr:`last_command` instead."""

    _unigym_name: str = ""
    """Registry identity stamped by ``build_adapter`` (the controls[]
    partition / group validation resolve ownership by this name)."""

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
            f"adapter resync hook")

    def snapshot_internal(self) -> dict[str, JsonValue] | None:
        """Serialize the adapter's internal streaming state (rewind face).

        Returns a numpy/JSON-safe dict carried inside every engine
        snapshot; the counterpart :meth:`restore_internal` writes it back.
        Default: None (declared absence — the rewind then degrades to the
        on_rewind re-seed path and the warmup hole is disclosed)."""
        return None

    def restore_internal(self, state: dict[str, JsonValue]) -> None:
        """Restore a :meth:`snapshot_internal` payload (rewind face)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement restore_internal")

    @abstractmethod
    def get_obs_requests(self) -> list[RawObsRequest]:
        """Declare raw quantities; the env backend fills obs_cache accordingly."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Single-frame adapter observation dimension."""

    @property
    @abstractmethod
    def control_dt(self) -> float:
        """Adapter control period in seconds (e.g. 0.02 for 50 Hz)."""

    @abstractmethod
    def compute_obs(self, obs_cache: dict[str, torch.Tensor],
                    cmd: torch.Tensor, env: AdapterEnv) -> torch.Tensor:
        """Assemble the single-frame adapter observation from raw quantities.

        Includes scales and concatenation order.  May also update internal
        history buffers (called exactly once per control step, after
        env.step).  ``obs_cache`` keys are the explicit request keys (robot
        state under ``articulation/<name>/...``).
        """

    def compute_obs_collect(self, obs_frame: torch.Tensor,
                            obs_cache: dict[str, torch.Tensor],
                            cmd: torch.Tensor, env: AdapterEnv) -> torch.Tensor:
        """Dataset-facing observation frame (recorded as ``rec["obs"]``).

        Default: return *obs_frame* unchanged — record exactly the frame the
        adapter consumed (bitwise-identical to the reference behavior).

        Override to assemble a different dataset frame (e.g. raw camera
        frames from *obs_cache* alongside proprioception) while keeping
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
    def postprocess_action(self, raw: torch.Tensor) -> dict[str, NamedAction]:
        """Raw action -> one NamedAction per owned control group.

        The returned dict is keyed by the control group names assigned to
        this adapter by the run's ``controls[]`` partition (e.g. ``"g1"``
        plus an extra ``"hand"`` face).  Each
        :class:`~unigym.runtime.actions.NamedAction` carries ``(N, dim)``
        target values, the joint name of every column, and its explicit
        ``kind`` (``position`` / ``velocity`` / ``torque`` / ``effort`` /
        ``current``) matching the group's configured ``target``.  Backends
        resolve the names to simulator addresses.
        """

    def named_action(self, values: torch.Tensor,
                     kind: str = "position") -> NamedAction:
        """Convenience: wrap *values* as a NamedAction over
        :attr:`joint_order` (the body group)."""

        order = self.joint_order
        if order is None:
            raise RuntimeError(
                f"{type(self).__name__}.named_action requires joint_order; "
                f"declare the adapter's joint order before wrapping values")
        return NamedAction(values=values, joint_names=list(order),
                           kind=kind)

    def on_reset(self, env_ids: torch.Tensor) -> None:
        """Called on episode reset for the given env ids (clear history etc.)."""

    def bind_env(self, env: AdapterEnv) -> None:
        """Called once by the env backend after perms are ready (hook for adapters
        that need the policy->isaaclab permutation)."""

    def extra_sinks(self, env: AdapterEnv, out_dir: str) -> list[Any]:
        """Adapter-provided diagnostics sinks (default: none).

        Mounted by collect_cli AHEAD of the standard recording/scoring
        chain (FanOutSink, construction order) so keys an adapter writes
        into ``bundle["extra"]`` during its own on_step reach the
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
        self, extra_info: dict[str, torch.Tensor], cmd: torch.Tensor,
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
        states.update(
            {k: v for k, v in extra_info.items()
             if k in ("vx", "vy", "wz", "base_z")})
        return states

    # ------------------------------------------------------------ #
    # Ordering properties
    # ------------------------------------------------------------ #
    @property
    def joint_order(self) -> list[str] | None:
        """Joint names in the adapter's (policy) order; None -> use robot profile order."""

    @property
    def foot_order(self) -> list[str] | None:
        """Foot body names in the adapter's (policy) order; None -> profile order."""
