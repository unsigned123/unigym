"""EnvBackend contract: the backend-agnostic surface consumed by collect.py.

This module is **torch-only** (no isaaclab) so the contract, the structural
validator and the capability gate are unit-testable offline.  A backend is the
per-step engine behind the data-collection rollout loop; collect.py talks to it
exactly through the surface described below — never through a concrete class.

Contract surface (consumed by collect.py / adapters)
====================================================

Methods:
``step(actions, raw_action=None, cmd=None) -> EnvStep``
    Advance one policy step.  ``actions`` is a ``Sequence[NamedAction]``
    (joint-addressed action batches routed by ``(articulation, joint)``;
    each entry's values are ``(N, dim)`` absolute joint position targets
    (rad) and ``joint_names`` names every column — backends resolve the
    names to simulator addresses, cached per name-tuple).  ``raw_action`` is
    the adapter's raw action (policy order) which must land in
    ``obs.prev_action``; ``cmd`` is the current command tensor which
    must land in ``obs.command``.
    Returns an :class:`EnvStep` carrying the typed observation snapshot,
    the record mapping (canonical artifact keys + bookkeeping) and the reset
    masks.
``reset_envs(env_ids) -> env_ids``
    Reset the given env subset through the FULL backend reset chain (scene
    teleport, DR events, manager resets, per-backend bookkeeping, obs-cache
    refresh).  Called AFTER the final step of an episode was recorded
    (record-before-reset data invariant).
``reset_all() -> env_ids``
    Batch-synchronous reset shorthand (``reset_envs(arange(num_envs))``).
``close() -> None``
    Release backend/simulation resources.

Attributes:
``obs: ObservationRegistry``
    The config-declared typed observation registry; refreshed once per step.
``controls: Mapping[tuple[str, str], ControlRuntime]``
    The ONLY per-control surface (v104.2): one runtime per explicit drive
    group, keyed by the physical ``(articulation, name)`` row label -- a
    bare group name is NOT a key (two articulations may both declare
    ``legs``).  The row label is never an action identity: runtime action
    columns are addressed by ``(articulation, joint_name)``.  Each runtime
    carries ``name`` / ``adapter`` / ``articulation`` / ``joint_names`` /
    ``q0`` / ``drive`` / ``target``.
``articulations: Mapping[str, ArticulationRuntime]``
    One runtime per declared articulation (``joint_names`` / ``q0`` /
    ``foot_bodies`` / ``body_sets`` / ``contacts`` / ``spawn`` / addresses).
    No articulation is primary.
``perm_il2pol / perm_pol2il: torch.Tensor``
    ``(J,)`` int64 gather indices (IL -> policy order and its inverse).
``to_policy_order(x_il) / to_il_order(x_pol)``
    Permutation helpers for ``(N, ...)`` joint-stacked tensors.
``policy_joint_pos_limits: (lower, upper) | None``
    Soft joint limits in policy order, each ``(N, J)``; ``None`` when the
    backend cannot provide them (termination checks degrade gracefully).
``num_envs: int``  /  ``device: str | torch.device``

Optional state-write face
=========================
State write is an **optional registry trait**, ``Trait.STATE_WRITE`` (projected
from ``BackendCapabilities.state_write``).  It is intentionally not part of the
required :class:`EnvBackend` validator: unimplemented backends stay valid
collection backends, and ``Run`` raises ``NotImplementedError`` when called on
a backend that does not declare the trait.  The three methods
(``set_root_state`` / ``set_joint_state`` / ``settle``) are ordinary backend
methods, not a second Protocol.  The contract lives in
``docs/state_write_contract.md``.

Failure semantics
=================
* A backend missing any contract member raises :class:`BackendContractError`
  from :func:`assert_backend_contract` (called by ``build_env_backend``).
* A yaml/adapter request for a capability the backend does not implement
  raises ``NotImplementedError`` from the capability gate in
  ``unigym.env.backends.build_env_backend`` — BEFORE the backend is
  constructed, with the requested feature named explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ...core import quat
from ..obs import ObsSnapshot
from ...runtime.actions import NamedAction

if TYPE_CHECKING:
    from ...backends.registry import Trait, TraitScope
    from ...runtime.controllers import ArticulationRuntime, ControlRuntime


@runtime_checkable
class _BackendInfoLike(Protocol):
    """The registry-info surface the capability gate reads."""

    @property
    def traits(self) -> frozenset[Trait]: ...

    def blocked(
        self,
        traits: Iterable[Trait],
        num_envs: int | None = None,
    ) -> Sequence[tuple[Trait, TraitScope]]: ...


def _as_backend_info(value: object) -> _BackendInfoLike | None:
    """View a registry info object through the gate's protocol."""
    return value if isinstance(value, _BackendInfoLike) else None


# --------------------------------------------------------------------------- #
# capability bits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackendCapabilities:
    """Optional features a backend may or may not implement.

    ``build_env_backend`` turns a yaml/adapter request for a capability that
    is ``False`` here into a clear ``NotImplementedError`` before the backend
    is constructed (no half-built sim state).
    """

    events: bool = False  # yaml randomization.events (per-step/reset DR)
    props: bool = False  # yaml props (interactive door/ball/generic)
    recorder: bool = False  # yaml recorder.enabled RGB recording
    depth_camera: bool = False  # selected observation cameras serve depth
    init_pose: bool = False  # yaml init_pose (randomized fallen spawn)
    cameras: bool = False  # camera obs (yaml cameras / adapter image obs;
    # docs/contracts/camera_contract.md)
    contacts: bool = False  # yaml contacts groups (ContactManager;
    # docs/contracts/contacts_contract.md)
    multi_env: bool = False  # native N>1 parallel-env runs (num_envs kwarg;
    # a single-env backend + num_envs>1 fails fast
    # at the gate, BEFORE construction)
    snapshot: bool = False  # mjSTATE-exact snapshot()/restore() (rewind,
    # Round B B1: mujoco only; isaaclab v1 stays
    # False -> the rewind gate fail-fasts)
    state_write: bool = False  # explicit world-frame state writes
    # (set_root_state/set_joint_state/settle;
    # docs/state_write_contract.md)
    native_handle: bool = False  # env.native(backend) -> NativeHandle
    # (external packs only; docs/native_handle.md)

    @classmethod
    def full(cls) -> BackendCapabilities:
        """All capabilities enabled (the isaaclab backend).

        ``snapshot`` is deliberately excluded: the rewind superpower is
        mujoco-only in Round B and ``full()``
        must keep describing exactly the isaaclab capability set.
        """
        return cls(
            events=True,
            props=True,
            recorder=True,
            depth_camera=True,
            init_pose=True,
            cameras=True,
            contacts=True,
            multi_env=True,
            state_write=True,
            native_handle=True,
        )

    def traits(self) -> frozenset[Trait]:
        """Trait view of the capability bits (unigym.backends.registry).

        The booleans are the low-bit projection of the central Trait
        enum; ``gate_capabilities`` consumes this view so the gate and the
        registry can never drift.  Each bit maps to a group of traits
        (``props`` is satisfied by any PROPS_* trait).
        """
        from ...backends.registry import Trait

        groups: dict[str, tuple[Trait, ...]] = {
            "events": (Trait.EVENTS_DR,),
            "props": (
                Trait.PROPS_RIGID,
                Trait.PROPS_ARTICULATION_URDF,
                Trait.PROPS_ARTICULATION_USD,
            ),
            "recorder": (Trait.RECORDER_VIDEO,),
            "depth_camera": (Trait.DEPTH_OBS_ALIASES,),
            "init_pose": (Trait.INIT_POSE,),
            "cameras": (Trait.CAMERAS_RGB,),
            "contacts": (Trait.CONTACTS_GROUPS,),
            "multi_env": (Trait.MULTI_ENV,),
            "snapshot": (Trait.SNAPSHOT,),
            "state_write": (Trait.STATE_WRITE,),
            "native_handle": (Trait.NATIVE_HANDLE,),
        }
        out: set[Trait] = set()
        for feat, enabled in {
            "events": self.events,
            "props": self.props,
            "recorder": self.recorder,
            "depth_camera": self.depth_camera,
            "init_pose": self.init_pose,
            "cameras": self.cameras,
            "contacts": self.contacts,
            "multi_env": self.multi_env,
            "snapshot": self.snapshot,
            "state_write": self.state_write,
            "native_handle": self.native_handle,
        }.items():
            if enabled:
                out.update(groups[feat])
        return frozenset(out)


# --------------------------------------------------------------------------- #
# per-step product
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnvStep:
    """One backend step's typed observation/record product."""

    t: int
    obs: ObsSnapshot
    record: Mapping[str, object]
    reset_env_ids: torch.Tensor
    should_reset: bool


# --------------------------------------------------------------------------- #
# the protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class EnvBackend(Protocol):
    """Structural interface every collection backend must satisfy.

    See the module docstring for the full semantic contract; the Protocol
    only fixes names/signatures (structural typing, no inheritance).
    """

    # attributes (populated by the backend implementation)
    num_envs: int
    device: str | torch.device
    obs: object
    controls: dict[tuple[str, str], ControlRuntime]
    articulations: dict[str, ArticulationRuntime]
    perm_il2pol: torch.Tensor
    perm_pol2il: torch.Tensor
    policy_joint_pos_limits: tuple[torch.Tensor, torch.Tensor] | None

    def step(
        self,
        actions: Sequence[NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> EnvStep: ...

    def record(self) -> dict[str, object]: ...

    def reset_envs(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def reset_all(self) -> torch.Tensor: ...

    def to_policy_order(self, x_il: torch.Tensor) -> torch.Tensor: ...

    def to_il_order(self, x_pol: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# state-write helpers (world frame, wxyz, name-addressed)
# --------------------------------------------------------------------------- #
# The three methods (set_root_state / set_joint_state / settle) are ordinary
# backend methods gated by Trait.STATE_WRITE; see docs/state_write_contract.md.
def normalize_state_env_ids(
    env_ids: Sequence[int] | torch.Tensor | None,
    num_envs: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Normalize a state-write env selection to a 1-D int64 tensor.

    ``None`` selects every env.  Input order is preserved (a repeated id is a
    repeated write, matching ``reset_envs``); ids outside ``[0, num_envs)``
    fail fast.  Floating/bool id tensors are a teaching error, not coerced.
    """
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.int64, device=device)
    if isinstance(env_ids, (str, bytes)):
        raise TypeError(
            f"env_ids must be a sequence of ints or a tensor, got " f"{type(env_ids).__name__}"
        )
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids))
    if ids.ndim != 1:
        raise ValueError(f"env_ids must be 1-D, got shape {tuple(ids.shape)}")
    if ids.dtype != torch.int64:
        if ids.is_floating_point() or ids.dtype == torch.bool:
            raise TypeError(f"env_ids must carry integer ids, got dtype {ids.dtype}")
        ids = ids.to(torch.int64)
    ids = ids.to(device=device)
    if ids.numel():
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= num_envs:
            raise ValueError(f"env_ids {ids.tolist()} are outside [0, num_envs={num_envs})")
    return ids


def validate_state_tensor(
    value: torch.Tensor,
    *,
    field: str,
    rows: int,
    cols: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Validate one ``(rows, cols)`` state-write tensor (shape/type/device/finite)."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor, got {type(value).__name__}")
    if tuple(value.shape) != (rows, cols):
        raise ValueError(f"{field} shape {tuple(value.shape)} != required ({rows}, {cols})")
    if not value.is_floating_point():
        raise TypeError(f"{field} must be a floating tensor, got dtype {value.dtype}")
    expected = str(torch.device(device))
    if str(value.device) != expected:
        raise ValueError(f"{field} is on device {value.device}, expected {expected}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains non-finite values")
    return value


def _require_state_tensor(value: object, *, field: str) -> torch.Tensor:
    """Narrow a dual-dtype math result to a tensor (runtime-checked)."""
    if not isinstance(value, torch.Tensor):  # pragma: no cover - core invariant
        raise TypeError(f"{field} did not produce a torch.Tensor")
    return value


def normalize_state_quat(
    quat_wxyz: torch.Tensor,
    *,
    rows: int,
    device: str | torch.device,
    field: str = "quat_wxyz",
) -> torch.Tensor:
    """Validate + normalize a ``(rows, 4)`` wxyz quaternion (core-owned math)."""
    value = validate_state_tensor(quat_wxyz, field=field, rows=rows, cols=4, device=device)
    norms = value.square().sum(dim=-1).sqrt()
    if bool((norms <= 1e-12).any()):
        raise ValueError(f"{field} contains a zero-norm quaternion")
    return _require_state_tensor(quat.normalize(value), field=field)


class BackendContractError(AssertionError):
    """Raised when a backend violates the structural EnvBackend contract."""


# --------------------------------------------------------------------------- #
# structural validator (pure torch, offline-runnable)
# --------------------------------------------------------------------------- #
def _validate_control_runtimes(env: EnvBackend) -> list[ControlRuntime]:
    """Validate ``env.controls`` / ``env.articulations``; return body controls."""
    controls = env.controls
    articulations = env.articulations
    if not isinstance(controls, dict) or not controls:
        raise BackendContractError(
            "controls must be a non-empty {(articulation, name): ControlRuntime} mapping"
        )
    if not isinstance(articulations, dict) or not articulations:
        raise BackendContractError(
            "articulations must be a non-empty {name: ArticulationRuntime} " "mapping"
        )
    for key, rt in controls.items():
        for member in (
            "name",
            "adapter",
            "articulation",
            "joint_names",
            "q0",
            "drive",
            "target",
        ):
            if not hasattr(rt, member):
                raise BackendContractError(
                    f"control {key!r} lacks the ControlRuntime member " f"{member!r}"
                )
        expected = (str(rt.articulation), str(rt.name))
        if key != expected:
            raise BackendContractError(
                f"controls key {key!r} != runtime identity {expected!r}; the "
                f"authoritative key is (articulation, name), never a bare name"
            )
        if not getattr(rt, "prop", False) and rt.articulation not in articulations:
            raise BackendContractError(
                f"control {key!r} names unknown articulation " f"{rt.articulation!r}"
            )
        q0 = rt.q0
        if (
            not isinstance(q0, torch.Tensor)
            or q0.dtype != torch.float32
            or tuple(q0.shape) != (len(rt.joint_names),)
        ):
            raise BackendContractError(
                f"control {key!r} q0 must be a float32 tensor of shape " f"({len(rt.joint_names)},)"
            )
    bodies = [c for c in controls.values() if c.role == "body"]
    if not bodies:
        raise BackendContractError("controls declares no explicit body group (role == 'body')")
    return bodies


def _validate_flat_permutations(
    env: EnvBackend,
    p12: torch.Tensor,
    p21: torch.Tensor,
    il: list[str],
    j: int,
    n: int,
) -> None:
    """The single-policy flat permutation/round-trip face."""
    if tuple(p12.shape) != (j,):
        raise BackendContractError(f"perm_il2pol must have shape ({j},)")
    if not torch.equal(p21, torch.argsort(p12)):
        raise BackendContractError("perm_pol2il must be the inverse of perm_il2pol")
    if int(p12.min()) < 0 or int(p12.max()) >= len(il):
        raise BackendContractError("perm_il2pol out of range for the asset joint names")
    x = torch.zeros(n, len(il), device=env.device)
    if tuple(env.to_policy_order(x).shape) != (n, j):
        raise BackendContractError("to_policy_order must map (N, len(il)) -> (N, J)")
    if tuple(env.to_il_order(torch.zeros(n, j, device=env.device)).shape) != (n, j):
        raise BackendContractError("to_il_order must map (N, J) -> (N, J)")


def assert_backend_contract(
    env: EnvBackend,
    exercise_step: bool = False,
) -> None:
    """Structurally validate *env* against the EnvBackend contract.

    Pure-torch and offline-runnable.  With ``exercise_step=True`` the
    validator additionally calls ``env.step`` once with zero targets and
    validates the returned observation snapshot/record against the schema —
    safe for in-process fakes, never call it on a live simulation backend.
    """
    # 1) member presence (explicit: a None/placeholder attribute must fail)
    methods = ("step", "reset_envs", "reset_all", "close", "to_policy_order", "to_il_order")
    attrs = (
        "num_envs",
        "device",
        "obs",
        "controls",
        "articulations",
        "perm_il2pol",
        "perm_pol2il",
        "policy_joint_pos_limits",
    )
    missing = [m for m in methods if not callable(getattr(env, m, None))]
    missing += [a for a in attrs if not hasattr(env, a)]
    if missing:
        raise BackendContractError(
            f"backend {type(env).__name__} does not satisfy the EnvBackend "
            f"protocol; missing/invalid members: {missing}"
        )

    n = env.num_envs
    if not isinstance(n, int) or n <= 0:
        raise BackendContractError(f"num_envs must be a positive int, got {n!r}")
    try:
        probe = torch.zeros(1, device=env.device)
    except Exception as e:  # pragma: no cover; wide on purpose: any device
        # failure is a contract violation -> typed raise below
        raise BackendContractError(f"invalid device {env.device!r}: {e}") from e
    del probe

    # 2) the per-control / per-articulation runtimes
    body_controls = _validate_control_runtimes(env)
    runtime = body_controls[0]
    pol = list(runtime.joint_names)
    art = env.articulations[runtime.articulation]
    il = list(art.joint_names)
    j = len(pol)
    p12, p21 = env.perm_il2pol, env.perm_pol2il
    for name, t in (("perm_il2pol", p12), ("perm_pol2il", p21)):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or t.ndim != 1:
            raise BackendContractError(f"{name} must be an int64 1-D tensor")
    if len(body_controls) == 1 and len(env.articulations) == 1:
        _validate_flat_permutations(env, p12, p21, il, j, n)

    # 3) limits / managers
    lim = env.policy_joint_pos_limits
    if lim is not None:
        if (
            not isinstance(lim, tuple)
            or len(lim) != 2
            or any(not isinstance(t, torch.Tensor) or tuple(t.shape) != (n, j) for t in lim)
        ):
            raise BackendContractError("policy_joint_pos_limits must be None or ((N, J), (N, J))")
    schema = getattr(env.obs, "schema", None)
    if schema is None:
        raise BackendContractError("obs registry must expose an ObservationSchema")

    # 4) optional live exercise (fakes / offline tests only)
    if exercise_step:
        actions = tuple(
            NamedAction.zeros(
                n,
                rt.articulation,
                list(rt.joint_names),
                rt.target,
                device=env.device,
            )
            for rt in body_controls
        )
        step = env.step(actions)
        if not isinstance(step.obs, ObsSnapshot):
            raise BackendContractError("step().obs must be an ObsSnapshot")
        for key, value in step.obs.values.items():
            if not isinstance(value, torch.Tensor):
                raise BackendContractError(f"obs[{key!r}] must be a torch tensor")
            if value.shape[0] != n:
                raise BackendContractError(
                    f"obs[{key!r}].shape[0] {value.shape[0]} != num_envs {n}"
                )
            if str(value.device) != str(torch.device(env.device)):
                raise BackendContractError(f"obs[{key!r}] on {value.device}, expected {env.device}")
        record = env.record()
        for key in env.obs.schema.record_keys():
            if key not in record:
                raise BackendContractError(f"record missing schema key {key!r}")
        for key in ("time_in_episode", "episode_id"):
            if key not in record:
                raise BackendContractError(f"record missing bookkeeping key {key!r}")


# --------------------------------------------------------------------------- #
# capability gate (used by build_env_backend; needs no backend instance)
# --------------------------------------------------------------------------- #
def requested_features(kwargs: Mapping[str, object]) -> dict[str, bool]:
    """Map capability name -> requested? from ``build_env_backend`` kwargs.

    Capability inference is config/trait driven: the observation declaration
    names the selected cameras and sensors, never an adapter request list.
    """
    observations = kwargs.get("observations")
    camera_names = tuple(getattr(observations, "camera", ()) or ())
    recorder = kwargs.get("recorder")
    raw_n = kwargs.get("num_envs")
    try:
        if not isinstance(raw_n, (int, float, str)):
            raise TypeError("num_envs is not numeric")
        n_envs = int(raw_n)
    except (TypeError, ValueError):
        n_envs = 1
    camera_manager = kwargs.get("camera_manager")
    camera_keys = set(camera_manager.obs_keys()) if camera_manager is not None else set()
    depth_requested = any(
        key.startswith(f"cam/{name}/") and key.endswith("/depth")
        for name in camera_names
        for key in camera_keys
    )
    return {
        "events": kwargs.get("events_cfg") is not None,
        "props": kwargs.get("prop_manager") is not None,
        "recorder": (recorder is not None and bool(getattr(recorder, "enabled", False))),
        "depth_camera": depth_requested,
        "init_pose": kwargs.get("init_pose") is not None,
        "cameras": camera_manager is not None,
        "contacts": kwargs.get("contact_manager") is not None,
        "multi_env": n_envs > 1,
        # director is not a capability bit: the registry trait table is its
        # truth (see the gate's director branch below).
        "director": kwargs.get("director_cfg") is not None,
    }


def gate_capabilities(
    backend_label: str,
    caps: BackendCapabilities | frozenset[Trait],
    kwargs: Mapping[str, object],
    num_envs: int | None = None,
) -> None:
    """Raise ``NotImplementedError`` when a request exceeds the backend.

    ``caps`` is a :class:`BackendCapabilities` (or anything exposing a
    ``traits()`` frozenset view / being a plain trait frozenset).  The
    check runs against the central trait registry; the missing-trait
    message format is unchanged (contract tests lock it).  *num_envs*
    enables the v096 ``TraitScope`` check (``None`` = no scope check, so
    fake/unit callers keep working).
    """
    from . import known_backends  # local: avoids a package-import cycle

    trait_view: frozenset[Trait]
    if isinstance(caps, BackendCapabilities):
        trait_view = caps.traits()
    elif hasattr(caps, "traits"):
        trait_view = frozenset(getattr(caps, "traits")())
    else:
        trait_view = frozenset(caps)
    from ...backends.registry import BackendSelectionError, Trait, backend_info

    info_view: _BackendInfoLike | None = None
    if num_envs is not None:
        try:
            info_view = _as_backend_info(backend_info(backend_label))
        except BackendSelectionError:  # unregistered fake names: no scopes
            info_view = None

    for feat, wanted in requested_features(kwargs).items():
        if not wanted:
            continue
        groups: dict[str, tuple[Trait, ...]] = {
            "events": (Trait.EVENTS_DR,),
            "props": (
                Trait.PROPS_RIGID,
                Trait.PROPS_ARTICULATION_URDF,
                Trait.PROPS_ARTICULATION_USD,
            ),
            "recorder": (Trait.RECORDER_VIDEO,),
            "depth_camera": (Trait.DEPTH_OBS_ALIASES,),
            "init_pose": (Trait.INIT_POSE,),
            "cameras": (Trait.CAMERAS_RGB,),
            "contacts": (Trait.CONTACTS_GROUPS,),
            "multi_env": (Trait.MULTI_ENV,),
            # snapshot is never yaml-requested (the rewind gate checks the
            # trait at command time); mapped here so the bit->trait view
            # cannot drift from the gate
            "snapshot": (Trait.SNAPSHOT,),
            "director": (
                Trait.DIRECTOR_TELEPORT,
                Trait.DIRECTOR_SETVEL,
                Trait.DIRECTOR_ATTACH_WELD,
                Trait.DIRECTOR_GRIPPER,
            ),
        }
        view: set[Trait] | frozenset[Trait] = trait_view
        if feat == "director" and info_view is not None:
            # no director capability bit exists: the registry declares it
            view = set(trait_view) | set(info_view.traits)
        present = set(groups[feat]) & set(view)
        if not present:
            # multi_env gets the actionable spawn hint (the num_envs knob is
            # a top-level run request, not a yaml capability section)
            hint = (
                " | hint: use -b isaaclab (multi-env native) or " "num_envs=1"
                if feat == "multi_env"
                else ""
            )
            raise NotImplementedError(
                f"backend '{backend_label}' does not implement capability "
                f"'{feat}' but the yaml/adapter requests it "
                f"(capabilities: {caps}); available backends: "
                f"{known_backends()}{hint}"
            )
        if info_view is None:
            continue
        blocked = list(info_view.blocked(present, num_envs))
        # only fail when EVERY present trait of the group is scoped out;
        # an unscoped sibling still serves the request.
        if blocked and {t for t, _ in blocked} == present:
            trait, scope = blocked[0]
            raise NotImplementedError(
                f"backend '{backend_label}' does not support capability "
                f"'{feat}' at num_envs={num_envs}: trait {trait.name} is "
                f"scoped to max_envs={scope.max_envs} (requested "
                f"num_envs={num_envs}) | hint: "
                f"{scope.note or 'this capability is N=1 only'}; run with "
                f"num_envs <= {scope.max_envs} or use a backend that "
                f"declares this trait for N>1 (available backends: "
                f"{known_backends()})"
            )
