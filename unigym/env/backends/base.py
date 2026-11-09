"""EnvBackend contract: the backend-agnostic surface consumed by collect.py.

This module is **torch-only** (no isaaclab) so the contract, the structural
validator and the capability gate are unit-testable offline.  A backend is the
per-step engine behind the data-collection rollout loop; collect.py talks to it
exactly through the surface described below — never through a concrete class.

Contract surface (consumed by collect.py / adapters)
====================================================

Methods:
``step(actions, raw_action=None, cmd=None) -> (obs_cache, extra)``
    Advance one policy step.  ``actions`` is a ``dict[str, NamedAction]``
    (name-addressed action faces; each entry's values are ``(N,
    dim)`` absolute joint position targets (rad) and ``joint_names`` names
    every column — backends resolve the names to simulator addresses, cached
    per name-tuple).  ``raw_action`` is the adapter's raw action (policy
    order) which must land in ``obs_cache["prev_action"]``; ``cmd`` is the
    current command tensor which must land in ``obs_cache["command"]``.
    Returns ``(obs_cache, extra)``: the (copied) observation cache dict and
    the per-step extra-info dict holding the 15 record keys + 2 control keys
    (see ``EXTRA_INFO_KEYS``).
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
``obs_cache: dict``
    Raw quantities the adapter requested via ``adapter.get_obs_requests()``;
    values are ``(N, dim)`` tensors on ``device``, refreshed every step.
``controls: Mapping[str, ControlRuntime]``
    The ONLY per-control surface (v104.2): one runtime per explicit drive
    face (``name`` / ``action_face`` / ``adapter`` / ``articulation`` /
    ``joint_names`` / ``q0`` / ``drive`` / ``target``).
``articulations: Mapping[str, ArticulationRuntime]``
    One runtime per declared articulation (``joint_names`` / ``q0`` /
    ``feet`` / ``body_parts`` / ``spawn`` / addresses).  No articulation is
    primary.
``perm_il2pol / perm_pol2il: torch.Tensor``
    ``(J,)`` int64 gather indices (IL -> policy order and its inverse).
``to_policy_order(x_il) / to_il_order(x_pol)``
    Permutation helpers for ``(N, ...)`` joint-stacked tensors.
``policy_joint_pos_limits: (lower, upper) | None``
    Soft joint limits in policy order, each ``(N, J)``; ``None`` when the
    backend cannot provide them (termination checks degrade gracefully).
``num_envs: int``  /  ``device: str | torch.device``

extra-info key set (EXTRA_INFO_KEYS)
====================================
Shapes are ``(N, ...)`` with ``N`` = num_envs, ``J`` = the policy
body control's joint count, ``F`` = its foot count; ``Ki`` is variable-length.  All tensors live on
``device``.  The 15 record keys are written to disk by collect.py; the 2
control keys drive the record-before-reset loop and are excluded from records.

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
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

import torch

from ..._typing import JsonValue
from ...runtime.actions import NamedAction

if TYPE_CHECKING:
    from ...backends.registry import Trait, TraitScope
    from ...runtime.controllers import ArticulationRuntime, ControlRuntime


class _ObsRequestLike(Protocol):
    """One adapter observation request (only the cache key is read)."""

    key: str


@runtime_checkable
class _BackendInfoLike(Protocol):
    """The registry-info surface the capability gate reads."""

    @property
    def traits(self) -> frozenset[Trait]: ...

    def blocked(
        self, traits: Iterable[Trait], num_envs: int | None = None,
    ) -> Sequence[tuple[Trait, TraitScope]]: ...


def _as_backend_info(value: object) -> _BackendInfoLike | None:
    """View a registry info object through the gate's protocol."""
    return value if isinstance(value, _BackendInfoLike) else None


def _is_obs_request_sequence(
    value: object,
) -> TypeGuard[Sequence[_ObsRequestLike]]:
    """True for an iterable of observation requests (not a str/mapping)."""
    return (hasattr(value, "__iter__")
            and not isinstance(value, (str, bytes))
            and not isinstance(value, Mapping))


def _is_json_mapping(value: object) -> TypeGuard[Mapping[str, JsonValue]]:
    """True for a mapping (the JSON config shape)."""
    return isinstance(value, Mapping)


# --------------------------------------------------------------------------- #
# extra-info key specification (15 record keys + 2 control keys)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtraKeySpec:
    """Shape/dtype contract for one ``extra`` key.

    ``dims`` entries are ints or the template letters ``"J"`` (policy joints)
    / ``"F"`` (policy feet) / ``"K"`` (variable length; also dtype-wildcard
    axis via ``-1``).  ``batch=False`` marks keys without the leading ``N``
    env dim (``reset_env_ids`` is a plain id vector).  ``dtype`` is a
    ``torch.dtype``; ``should_reset`` uses the pseudo-marker ``"py_bool"``
    for a plain python bool.
    """

    dims: tuple[int | str, ...]
    dtype: torch.dtype | str
    batch: bool = True


EXTRA_INFO_KEYS: dict[str, ExtraKeySpec] = {
    # ---- per-joint recorded quantities (policy order) ----
    "joint_pos": ExtraKeySpec(dims=("J",), dtype=torch.float32),
    "joint_vel": ExtraKeySpec(dims=("J",), dtype=torch.float32),
    "applied_torque": ExtraKeySpec(dims=("J",), dtype=torch.float32),
    # ---- base frame (world ``*_w`` / base frame ``*_b``) ----
    "gravity_b": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    "base_pos_w": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    "base_quat_w": ExtraKeySpec(dims=(4,), dtype=torch.float32),  # (w,x,y,z)
    "base_lin_vel_w": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    "base_ang_vel_w": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    "base_lin_vel_b": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    "base_ang_vel_b": ExtraKeySpec(dims=(3,), dtype=torch.float32),
    # ---- feet (policy foot order) ----
    "foot_contact": ExtraKeySpec(dims=("F",), dtype=torch.bool),
    "foot_contact_force_z": ExtraKeySpec(dims=("F",), dtype=torch.float32),
    "foot_pos_z": ExtraKeySpec(dims=("F",), dtype=torch.float32),
    # ---- episode bookkeeping ----
    "time_in_episode": ExtraKeySpec(dims=(), dtype=torch.float32),
    "episode_id": ExtraKeySpec(dims=(), dtype=torch.int64),
    # ---- control keys (never recorded) ----
    "reset_env_ids": ExtraKeySpec(dims=("K",), dtype=torch.int64, batch=False),
    "should_reset": ExtraKeySpec(dims=(), dtype="py_bool"),
}

RECORD_KEYS = tuple(k for k, s in EXTRA_INFO_KEYS.items()
                    if k not in ("reset_env_ids", "should_reset"))
CONTROL_KEYS = ("reset_env_ids", "should_reset")


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

    events: bool = False          # yaml randomization.events (per-step/reset DR)
    props: bool = False           # yaml props (interactive door/ball/generic)
    recorder: bool = False        # yaml recorder.enabled RGB recording
    depth_camera: bool = False    # yaml camera_depth (depth-encoder policy input)
    init_pose: bool = False       # yaml init_pose (randomized fallen spawn)
    cameras: bool = False           # camera obs (yaml cameras / adapter image obs;
                                    # docs/contracts/camera_contract.md)
    contacts: bool = False          # yaml contacts groups (ContactManager;
                                    # docs/contracts/contacts_contract.md)
    multi_env: bool = False         # native N>1 parallel-env runs (num_envs kwarg;
                                    # a single-env backend + num_envs>1 fails fast
                                    # at the gate, BEFORE construction)
    snapshot: bool = False          # mjSTATE-exact snapshot()/restore() (rewind,
                                    # Round B B1: mujoco only; isaaclab v1 stays
                                    # False -> the rewind gate fail-fasts)
    actuator_seam: bool = False     # pluggable ctrl laws (env.params.actuators,
                                    # unigym/actuators; mujoco only
                                    # for now, isaaclab maps ActuatorBase later)

    @classmethod
    def full(cls) -> BackendCapabilities:
        """All capabilities enabled (the isaaclab backend).

        ``snapshot`` is deliberately excluded: the rewind superpower is
        mujoco-only in Round B and ``full()``
        must keep describing exactly the isaaclab capability set.
        """
        return cls(events=True, props=True, recorder=True,
                   depth_camera=True, init_pose=True, cameras=True,
                   contacts=True, multi_env=True)

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
            "props": (Trait.PROPS_RIGID, Trait.PROPS_ARTICULATION_URDF,
                      Trait.PROPS_ARTICULATION_USD),
            "recorder": (Trait.RECORDER_VIDEO,),
            "depth_camera": (Trait.DEPTH_OBS_ALIASES,),
            "init_pose": (Trait.INIT_POSE,),
            "cameras": (Trait.CAMERAS_RGB,),
            "contacts": (Trait.CONTACTS_GROUPS,),
            "multi_env": (Trait.MULTI_ENV,),
            "snapshot": (Trait.SNAPSHOT,),
            "actuator_seam": (Trait.ACTUATORS_PLUGGABLE,),
        }
        out: set[Trait] = set()
        for feat, enabled in {
            "events": self.events,
            "props": self.props, "recorder": self.recorder,
            "depth_camera": self.depth_camera, "init_pose": self.init_pose,
            "cameras": self.cameras, "contacts": self.contacts,
            "multi_env": self.multi_env, "snapshot": self.snapshot,
            "actuator_seam": self.actuator_seam,
        }.items():
            if enabled:
                out.update(groups[feat])
        return frozenset(out)


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
    obs_cache: dict[str, torch.Tensor]
    controls: dict[str, ControlRuntime]
    articulations: dict[str, ArticulationRuntime]
    perm_il2pol: torch.Tensor
    perm_pol2il: torch.Tensor
    policy_joint_pos_limits: tuple[torch.Tensor, torch.Tensor] | None

    def step(
        self, actions: dict[str, NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor | bool],
    ]: ...

    def reset_envs(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def reset_all(self) -> torch.Tensor: ...

    def to_policy_order(self, x_il: torch.Tensor) -> torch.Tensor: ...

    def to_il_order(self, x_pol: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# ObsCache contract (comment-only spec; enforced structurally below)
# --------------------------------------------------------------------------- #
# obs_cache is a plain dict, refreshed by the backend every step:
#   * keys  == exactly the requests from adapter.get_obs_requests()
#     (Request.key strings such as "gravity_b", "joint_pos_rel", "joint_vel",
#     "command", "prev_action", "depth_image", ... plus prop-manager keys).
#   * values are (N, request.dim) tensors on env.device; image-style entries
#     are (N, C, H, W) with shape[0] == N.
#   * "command"/"prev_action" are passthrough: when the caller passes None
#     the backend keeps the previous entry (or zero-fills on the first step).
# Adapters own scaling/stacking; the backend never interprets values.


class BackendContractError(AssertionError):
    """Raised when a backend violates the structural EnvBackend contract."""


# --------------------------------------------------------------------------- #
# structural validator (pure torch, offline-runnable)
# --------------------------------------------------------------------------- #
def _resolve_dims(
    spec: ExtraKeySpec, n_joints: int, n_feet: int,
) -> tuple[int, ...]:
    table: dict[str, int] = {
        "J": n_joints, "F": n_feet, "K": -1}  # -1 == wildcard axis
    out: list[int] = []
    for d in spec.dims:
        if isinstance(d, str):
            if d not in table:
                raise BackendContractError(f"unknown dim template {d!r}")
            out.append(table[d])
        else:
            out.append(int(d))
    return tuple(out)


def _shape_matches(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    """Axis-wise shape match; a -1 axis in *expected* is a wildcard."""
    return (len(actual) == len(expected)
            and all(e == -1 or a == e for a, e in zip(actual, expected)))


_EXTRA_BOOKKEEPING = frozenset({"time_in_episode", "episode_id",
                                "reset_env_ids", "should_reset"})


def _extra_candidates(
    extra: dict[str, torch.Tensor | bool], key: str,
) -> list[str]:
    """Record keys serving *key*: bare (legacy) or namespaced (v103.5).

    Per-articulation state is published as ``articulation/<name>/<key>``;
    the validator accepts every namespace (one for a single-policy run),
    never inventing a primary.
    """
    if key in _EXTRA_BOOKKEEPING:
        return [key] if key in extra else []
    suffix = "/" + key
    return [k for k in extra
            if isinstance(k, str) and k.startswith("articulation/")
            and k.endswith(suffix)]


def check_extra_info(
    extra: dict[str, torch.Tensor | bool], num_envs: int, n_joints: int,
    n_feet: int, device: str | torch.device = "cpu",
) -> None:
    """Validate one ``step()`` extra dict against ``EXTRA_INFO_KEYS``.

    Checks key completeness (15 record keys + 2 control keys), shapes and
    dtypes.  Per-articulation record keys are resolved under
    ``articulation/<name>/...`` (v103.5).  Raises
    :class:`BackendContractError` on the first violation.
    """
    resolved = {k: _extra_candidates(extra, k) for k in EXTRA_INFO_KEYS}
    missing = [k for k, c in resolved.items() if not c]
    if missing:
        raise BackendContractError(f"extra missing keys {missing}; "
                                   f"required: {sorted(EXTRA_INFO_KEYS)}")
    for key, spec in EXTRA_INFO_KEYS.items():
        for actual_key in resolved[key]:
            val = extra[actual_key]
            dims = _resolve_dims(spec, n_joints, n_feet)  # "K" -> -1 wildcard
            expected_shape = ((num_envs,) if spec.batch else ()) + dims
            if spec.dtype == "py_bool":
                if not isinstance(val, (bool, torch.Tensor)) or (
                        isinstance(val, torch.Tensor)
                        and val.dtype != torch.bool):
                    raise BackendContractError(
                        f"extra['{actual_key}'] must be a python bool or "
                        f"bool tensor, got {type(val).__name__}")
                continue
            if not isinstance(val, torch.Tensor):
                raise BackendContractError(
                    f"extra['{actual_key}'] must be a torch tensor, got "
                    f"{type(val).__name__}")
            if not _shape_matches(tuple(val.shape), expected_shape):
                raise BackendContractError(
                    f"extra['{actual_key}'].shape {tuple(val.shape)} != "
                    f"{expected_shape}")
            if val.dtype != spec.dtype:
                raise BackendContractError(
                    f"extra['{actual_key}'].dtype {val.dtype} != {spec.dtype}")
            if str(val.device) != str(torch.device(device)):
                raise BackendContractError(
                    f"extra['{actual_key}'] on device {val.device}, "
                    f"expected {device}")


def check_obs_cache(
    obs: dict[str, torch.Tensor], num_envs: int,
    device: str | torch.device = "cpu",
) -> None:
    """Validate obs-cache entries: (N, ...) tensors on ``device``."""
    if not isinstance(obs, dict):
        raise BackendContractError(
            f"obs_cache must be a dict, got {type(obs).__name__}")
    for key, val in obs.items():
        if not isinstance(val, torch.Tensor):
            raise BackendContractError(
                f"obs_cache['{key}'] must be a torch tensor, "
                f"got {type(val).__name__}")
        if val.shape[0] != num_envs:
            raise BackendContractError(
                f"obs_cache['{key}'].shape[0] {val.shape[0]} != num_envs "
                f"{num_envs}")
        if str(val.device) != str(torch.device(device)):
            raise BackendContractError(
                f"obs_cache['{key}'] on device {val.device}, expected {device}")


def _validate_control_runtimes(env: EnvBackend) -> list[ControlRuntime]:
    """Validate ``env.controls`` / ``env.articulations``; return body controls."""
    controls = env.controls
    articulations = env.articulations
    if not isinstance(controls, dict) or not controls:
        raise BackendContractError(
            "controls must be a non-empty {name: ControlRuntime} mapping")
    if not isinstance(articulations, dict) or not articulations:
        raise BackendContractError(
            "articulations must be a non-empty {name: ArticulationRuntime} "
            "mapping")
    for key, rt in controls.items():
        for member in ("name", "action_face", "adapter", "articulation",
                       "joint_names", "q0", "drive", "target"):
            if not hasattr(rt, member):
                raise BackendContractError(
                    f"control {key!r} lacks the ControlRuntime member "
                    f"{member!r}")
        if key != rt.name:
            raise BackendContractError(
                f"controls key {key!r} != runtime name {rt.name!r}")
        if not getattr(rt, "prop", False) \
                and rt.articulation not in articulations:
            raise BackendContractError(
                f"control {key!r} names unknown articulation "
                f"{rt.articulation!r}")
        q0 = rt.q0
        if (not isinstance(q0, torch.Tensor) or q0.dtype != torch.float32
                or tuple(q0.shape) != (len(rt.joint_names),)):
            raise BackendContractError(
                f"control {key!r} q0 must be a float32 tensor of shape "
                f"({len(rt.joint_names)},)")
    bodies = [c for c in controls.values()
              if c.action_face == c.articulation]
    if not bodies:
        raise BackendContractError("controls declares no body/articulation face")
    return bodies


def _validate_flat_permutations(
    env: EnvBackend, p12: torch.Tensor, p21: torch.Tensor, il: list[str],
    j: int, n: int,
) -> None:
    """The single-policy flat permutation/round-trip face."""
    if tuple(p12.shape) != (j,):
        raise BackendContractError(f"perm_il2pol must have shape ({j},)")
    if not torch.equal(p21, torch.argsort(p12)):
        raise BackendContractError(
            "perm_pol2il must be the inverse of perm_il2pol")
    if int(p12.min()) < 0 or int(p12.max()) >= len(il):
        raise BackendContractError(
            "perm_il2pol out of range for the asset joint names")
    x = torch.zeros(n, len(il), device=env.device)
    if tuple(env.to_policy_order(x).shape) != (n, j):
        raise BackendContractError(
            "to_policy_order must map (N, len(il)) -> (N, J)")
    if tuple(env.to_il_order(torch.zeros(n, j, device=env.device)).shape) \
            != (n, j):
        raise BackendContractError("to_il_order must map (N, J) -> (N, J)")


def assert_backend_contract(
    env: EnvBackend, exercise_step: bool = False,
) -> None:
    """Structurally validate *env* against the EnvBackend contract.

    Pure-torch and offline-runnable.  With ``exercise_step=True`` the
    validator additionally calls ``env.step`` once with zero targets and
    validates the returned extra/obs_cache against the specifications —
    safe for in-process fakes, never call it on a live simulation backend.
    """
    # 1) member presence (explicit: a None/placeholder attribute must fail)
    methods = ("step", "reset_envs", "reset_all", "close",
               "to_policy_order", "to_il_order")
    attrs = ("num_envs", "device", "obs_cache", "controls",
             "articulations", "perm_il2pol", "perm_pol2il",
             "policy_joint_pos_limits")
    missing = [m for m in methods if not callable(getattr(env, m, None))]
    missing += [a for a in attrs if not hasattr(env, a)]
    if missing:
        raise BackendContractError(
            f"backend {type(env).__name__} does not satisfy the EnvBackend "
            f"protocol; missing/invalid members: {missing}")

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
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.int64
                or t.ndim != 1):
            raise BackendContractError(f"{name} must be an int64 1-D tensor")
    if len(body_controls) == 1 and len(env.articulations) == 1:
        _validate_flat_permutations(env, p12, p21, il, j, n)

    # 3) limits / managers
    lim = env.policy_joint_pos_limits
    if lim is not None:
        if (not isinstance(lim, tuple) or len(lim) != 2
                or any(not isinstance(t, torch.Tensor)
                       or tuple(t.shape) != (n, j) for t in lim)):
            raise BackendContractError(
                "policy_joint_pos_limits must be None or ((N, J), (N, J))")
    if not isinstance(env.obs_cache, dict):
        raise BackendContractError("obs_cache must be a dict")

    # 4) optional live exercise (fakes / offline tests only)
    if exercise_step:
        actions = {rt.action_face: NamedAction.zeros(
            n, list(rt.joint_names), rt.target, device=env.device)
            for rt in body_controls}
        obs, extra = env.step(actions)
        check_extra_info(extra, n, j, len(runtime.feet),
                         device=env.device)
        check_obs_cache(obs, n, device=env.device)
        check_obs_cache(env.obs_cache, n, device=env.device)

# --------------------------------------------------------------------------- #
# capability gate (used by build_env_backend; needs no backend instance)
# --------------------------------------------------------------------------- #
def _requests_pluggable_actuators(
    backend_params: Mapping[str, JsonValue] | None,
) -> bool:
    """Does ``env.params.actuators`` declare a non-default ctrl law?

    Capability-gate inference: software laws beyond standard_pd
    (computed_torque_pd / native_servo / custom registrations) are
    mujoco-only for now; a standard-only declaration rides the shared
    default face and never gates.  Missing ``type`` counts as standard
    here (the backend's own config validation fails fast on it).
    """
    act = backend_params.get("actuators") \
        if backend_params is not None else None
    if not isinstance(act, dict):
        return False
    for group in act.values():
        if isinstance(group, dict) \
                and str(group.get("type", "standard_pd")) != "standard_pd":
            return True
    return False


def requested_features(kwargs: Mapping[str, object]) -> dict[str, bool]:
    """Map capability name -> requested? from ``build_env_backend`` kwargs."""
    adapter = kwargs.get("adapter")
    req_keys: set[str]
    try:
        if adapter is None:
            raise TypeError("no adapter: no obs requests to infer")
        get_requests = getattr(adapter, "get_obs_requests", None)
        if not callable(get_requests):
            raise TypeError("adapter exposes no get_obs_requests")
        raw_requests = get_requests()
        if not _is_obs_request_sequence(raw_requests):
            raise TypeError("adapter obs requests expose no string key")
        req_keys = {r.key for r in raw_requests}
    except Exception:  # best effort: capability inference only -> no requests
        req_keys = set()
    recorder = kwargs.get("recorder")
    raw_n = kwargs.get("num_envs")
    try:
        if not isinstance(raw_n, (int, float, str)):
            raise TypeError("num_envs is not numeric")
        n_envs = int(raw_n)
    except (TypeError, ValueError):
        n_envs = 1
    raw_params = kwargs.get("backend_params")
    backend_params: Mapping[str, JsonValue] | None = None
    if _is_json_mapping(raw_params):
        backend_params = raw_params
    return {
        "events": kwargs.get("events_cfg") is not None,
        "props": kwargs.get("prop_manager") is not None,
        "recorder": (recorder is not None
                     and bool(getattr(recorder, "enabled", False))),
        "depth_camera": bool({"depth_image", "depth_image_raw"} & req_keys),
        "init_pose": kwargs.get("init_pose") is not None,
        "cameras": kwargs.get("camera_manager") is not None,
        "contacts": kwargs.get("contact_manager") is not None,
        "multi_env": n_envs > 1,
        "actuator_seam": _requests_pluggable_actuators(backend_params),
        # director is not a capability bit: the registry trait table is its
        # truth (see the gate's director branch below).
        "director": kwargs.get("director_cfg") is not None,
    }


def gate_capabilities(
    backend_label: str, caps: BackendCapabilities | frozenset[Trait],
    kwargs: Mapping[str, object], num_envs: int | None = None,
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
            "props": (Trait.PROPS_RIGID, Trait.PROPS_ARTICULATION_URDF,
                      Trait.PROPS_ARTICULATION_USD),
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
            "actuator_seam": (Trait.ACTUATORS_PLUGGABLE,),
            "director": (Trait.DIRECTOR_TELEPORT, Trait.DIRECTOR_SETVEL,
                         Trait.DIRECTOR_ATTACH_WELD, Trait.DIRECTOR_GRIPPER),
        }
        view: set[Trait] | frozenset[Trait] = trait_view
        if feat == "director" and info_view is not None:
            # no director capability bit exists: the registry declares it
            view = set(trait_view) | set(info_view.traits)
        present = set(groups[feat]) & set(view)
        if not present:
            # multi_env gets the actionable spawn hint (the num_envs knob is
            # a top-level run request, not a yaml capability section)
            hint = (" | hint: use -b isaaclab (multi-env native) or "
                    "num_envs=1" if feat == "multi_env" else "")
            if feat == "actuator_seam":
                hint = (" | hint: env.params.actuators non-standard ctrl "
                        "laws (computed_torque_pd / native_servo) are "
                        "mujoco-only for now (unigym/actuators, v063 D2) "
                        "-- run this config with env.backend: mujoco; the "
                        "isaaclab ActuatorBase mapping is a later task")
            raise NotImplementedError(
                f"backend '{backend_label}' does not implement capability "
                f"'{feat}' but the yaml/adapter requests it "
                f"(capabilities: {caps}); available backends: "
                f"{known_backends()}{hint}")
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
                f"{known_backends()})")
