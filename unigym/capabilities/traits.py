"""Pure backend capability vocabulary (traits, scopes, model-edit fields).

Single source of truth for "what can a backend do".  The trait table is the
mechanized touch matrix (docs/designs/backend-traits.md §1.1 → §3.1): every
optional feature a yaml/adapter/director-op can request maps to a trait, and a
backend missing that trait fails fast at build time (``{error, hint}``) —
never a silent degradation.

Backends are *equal citizens*: ``mujoco`` / ``mjwarp`` / ``isaaclab2`` /
``isaaclab3_physx`` / ``isaaclab3_newton``.  The two IsaacLab 3 identities share
one implementation package but differ in engine, runtime stack and capability
set.  Selection is canonical-only: there are no ``isaaclab`` / ``isaaclab3``
aliases and no version-based family auto-selection.

This module is **stdlib-only** and imports no backend module: it owns the pure
trait/scope tables and the validations that do not need backend factories.
Backend construction faces live in the composition layer.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CANONICAL_BACKENDS",
    "ISAACLAB2_MODEL_EDIT_FIELDS",
    "ISAACLAB2_TRAITS",
    "ISAACLAB3_NEWTON_TRAITS",
    "ISAACLAB3_PHYSX_TRAITS",
    "MJWARP_BACKEND",
    "MJWARP_SCOPES",
    "MJWARP_TRAITS",
    "MUJOCO_MODEL_EDIT_FIELDS",
    "MUJOCO_SCOPES",
    "MUJOCO_TRAITS",
    "TRAIT_RULES",
    "ModelEditField",
    "Trait",
    "TraitRule",
    "TraitScope",
    "build_name",
    "known_backends",
    "validate_trait_relations",
    "validate_trait_scopes",
]


class Trait(Enum):
    """One addressable backend capability.

    Env-core traits (the first three) are mandatory: a backend that cannot
    deliver them is not registrable.  All other traits are optional; a
    request for a missing one fails fast with the trait name, the request
    source and the backend's trait set.
    """

    # --- env core (registration prerequisite) ---
    ENV_STEP_RESET = "ENV_STEP_RESET"  # step/reset_envs/reset_all + extra 17-key contract
    ACTION_FACES_NAMED = "ACTION_FACES_NAMED"  # consumes Sequence[NamedAction]
    JOINT_PERMUTATION = "JOINT_PERMUTATION"  # policy/il orders + inverse perm
    # --- env shape (optional: single-env backends omit it) ---
    MULTI_ENV = "MULTI_ENV"  # native N>1 parallel-env runs
    # --- state writes (optional: collection-only backends omit it) ---
    STATE_WRITE = "STATE_WRITE"  # explicit world-frame simulator state writes
    # --- env layout (exactly one ENV_LAYOUT_* per registered backend) ---
    ENV_LAYOUT_TILED = "ENV_LAYOUT_TILED"  # env_spacing; one view spans envs
    ENV_LAYOUT_COINCIDENT = "ENV_LAYOUT_COINCIDENT"  # same-coordinate worlds
    # --- observation / recording ---
    CAMERAS_RGB = "CAMERAS_RGB"  # rig-level RGB frames
    CAMERAS_GLOBAL = "CAMERAS_GLOBAL"  # scene-wide single camera/image
    CAMERAS_PER_ENV = "CAMERAS_PER_ENV"  # one camera instance per env (N>=1)
    CAMERAS_PER_ENV_STATE = "CAMERAS_PER_ENV_STATE"  # (camera, env) own pose state
    CAMERAS_BATCH_RENDER = "CAMERAS_BATCH_RENDER"  # one render covers all instances
    CAMERAS_PER_ENV_RESOLUTION = "CAMERAS_PER_ENV_RESOLUTION"  # reserved
    CAMERAS_PER_ENV_INTRINSICS = "CAMERAS_PER_ENV_INTRINSICS"  # reserved
    CAMERAS_DEPTH_RIG = "CAMERAS_DEPTH_RIG"  # rig-level metric z-depth
    DEPTH_OBS_ALIASES = "DEPTH_OBS_ALIASES"  # depth obs alias keys (isaaclab)
    RECORDER_VIDEO = "RECORDER_VIDEO"  # recorder.enabled mp4
    # --- contacts ---
    CONTACTS_GROUPS = "CONTACTS_GROUPS"  # declarative groups + read_raw
    CONTACTS_OVERRIDE = "CONTACTS_OVERRIDE"  # director contact op
    # --- props ---
    PROPS_RIGID = "PROPS_RIGID"  # sphere/box-family/usd tiers
    PROPS_ARTICULATION_URDF = "PROPS_ARTICULATION_URDF"
    PROPS_ARTICULATION_USD = "PROPS_ARTICULATION_USD"
    PROPS_LATCH = "PROPS_LATCH"  # runtime drive swap
    PROPS_ACTUATED_ACTIONS = "PROPS_ACTUATED_ACTIONS"  # actuated joints -> NamedAction
    # --- director ---
    DIRECTOR_TELEPORT = "DIRECTOR_TELEPORT"
    DIRECTOR_SETVEL = "DIRECTOR_SETVEL"
    DIRECTOR_ATTACH_WELD = "DIRECTOR_ATTACH_WELD"  # frozen relative pose
    DIRECTOR_ATTACH_SLAVE = "DIRECTOR_ATTACH_SLAVE"  # per-tick kinematic follow
    DIRECTOR_GRIPPER = "DIRECTOR_GRIPPER"
    # --- scene / randomization ---
    SCENE_COMPOSE = "SCENE_COMPOSE"  # neutral Scene -> composed world face
    TERRAIN = "TERRAIN"
    EVENTS_DR = "EVENTS_DR"
    INIT_POSE = "INIT_POSE"
    SPAWN_XY_YAW = "SPAWN_XY_YAW"  # both backends (parity)
    MOCAP = "MOCAP"  # mocap body carrier (mujoco)
    # --- rewind (Round B B1) ---
    SNAPSHOT = "SNAPSHOT"  # mjSTATE-exact snapshot()/restore()
    # --- model introspection (no second capability enum) ---
    MODEL_ADDRESSES = "MODEL_ADDRESSES"  # qpos/dof/geom/actuator ids
    MODEL_MASS_INERTIA = "MODEL_MASS_INERTIA"  # compiled mass/inertia
    MODEL_MOCAP = "MODEL_MOCAP"  # mocap body carriers
    MODEL_EQ_SLOTS = "MODEL_EQ_SLOTS"  # equality slots
    MODEL_USD_PRIMS = "MODEL_USD_PRIMS"  # USD prim hierarchy
    # --- model edits (global; no per-env face) ---
    MODEL_EDIT = "MODEL_EDIT"  # run.model_edit() transaction
    # --- on-demand runtime query (not an observation channel) ---
    RUNTIME_QUERY = "RUNTIME_QUERY"  # run.query -> Query facade
    # --- native escape hatch (external packs only; unstable) ---
    NATIVE_HANDLE = "NATIVE_HANDLE"  # env.native(backend) -> NativeHandle


class ModelEditField(Enum):
    """One addressable model parameter inside a ``MODEL_EDIT`` transaction.

    The enum is the single source of truth for the field capability matrix;
    backends declare the subset they can write through their trait metadata.
    """

    BODY_MASS = "body.mass"
    BODY_COM = "body.com"
    BODY_INERTIA = "body.inertia"
    BODY_INERTIA_QUAT = "body.inertia_quat"
    GEOM_FRICTION = "geom.friction"
    GEOM_RESTITUTION = "geom.restitution"


BODY_MODEL_EDIT_FIELDS = frozenset(
    {
        ModelEditField.BODY_MASS,
        ModelEditField.BODY_COM,
        ModelEditField.BODY_INERTIA,
        ModelEditField.BODY_INERTIA_QUAT,
    }
)
GEOM_MODEL_EDIT_FIELDS = frozenset(
    {
        ModelEditField.GEOM_FRICTION,
        ModelEditField.GEOM_RESTITUTION,
    }
)
MUJOCO_MODEL_EDIT_FIELDS = BODY_MODEL_EDIT_FIELDS | GEOM_MODEL_EDIT_FIELDS
ISAACLAB2_MODEL_EDIT_FIELDS = BODY_MODEL_EDIT_FIELDS


# --------------------------------------------------------------------------- #
# TraitScope: per-backend availability predicate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraitScope:
    """One (backend, trait) availability predicate; PRESENCE stays the enum.

    A scope only *narrows* a declared trait (``has()`` stays presence-only):
    the backend has the trait, but a run with more than ``max_envs``
    environments cannot use it.  ``max_envs=None`` means "any num_envs".
    """

    max_envs: int | None = None
    note: str = ""


# --------------------------------------------------------------------------- #
# trait rules (the single source of truth for cross-trait relations)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraitRule:
    """Declarative relations of one trait to the rest of the set."""

    requires: tuple[Trait, ...] = ()
    exclusive_with: tuple[Trait, ...] = ()


TRAIT_RULES: dict[Trait, TraitRule] = {
    Trait.ENV_LAYOUT_TILED: TraitRule(exclusive_with=(Trait.ENV_LAYOUT_COINCIDENT,)),
    Trait.ENV_LAYOUT_COINCIDENT: TraitRule(exclusive_with=(Trait.ENV_LAYOUT_TILED,)),
    Trait.CAMERAS_GLOBAL: TraitRule(requires=(Trait.ENV_LAYOUT_TILED,)),
    Trait.CAMERAS_PER_ENV_STATE: TraitRule(requires=(Trait.CAMERAS_PER_ENV,)),
    Trait.CAMERAS_BATCH_RENDER: TraitRule(requires=(Trait.CAMERAS_PER_ENV, Trait.MULTI_ENV)),
    Trait.CAMERAS_PER_ENV_RESOLUTION: TraitRule(requires=(Trait.CAMERAS_PER_ENV,)),
    Trait.CAMERAS_PER_ENV_INTRINSICS: TraitRule(requires=(Trait.CAMERAS_PER_ENV_STATE,)),
}

_LAYOUT_TRAITS = (Trait.ENV_LAYOUT_TILED, Trait.ENV_LAYOUT_COINCIDENT)


# --------------------------------------------------------------------------- #
# per-backend trait tables (declaration = registration)
# --------------------------------------------------------------------------- #
_CORE = frozenset(
    {
        Trait.ENV_STEP_RESET,
        Trait.ACTION_FACES_NAMED,
        Trait.JOINT_PERMUTATION,
    }
)

MUJOCO_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N independent MjData
        Trait.ENV_LAYOUT_COINCIDENT,  # one MjModel / one world per run
        Trait.SCENE_COMPOSE,
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,
        Trait.RECORDER_VIDEO,
        Trait.CAMERAS_PER_ENV,  # N=1 degenerate instance
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,
        Trait.PROPS_ACTUATED_ACTIONS,
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,
        Trait.DIRECTOR_ATTACH_WELD,
        Trait.DIRECTOR_GRIPPER,
        Trait.SPAWN_XY_YAW,
        Trait.MOCAP,
        Trait.SNAPSHOT,
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        # Model introspection: the compiled MjModel carries them all
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
        Trait.MODEL_EDIT,  # global body/geom parameter transaction
        Trait.RUNTIME_QUERY,  # on-demand query face (not an obs channel)
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

# The independent MuJoCo Warp backend.  N>1 is ONE batched warp
# Data (nworld=N) with N CPU mirrors, plus scene compose through the
# shared MujocoSceneEngine, box-family props, contact groups, RGB rigs,
# planar spawn, and the mirror-faithful snapshot/restore face.
MJWARP_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,
        Trait.ENV_LAYOUT_COINCIDENT,  # N same-coordinate worlds
        Trait.SCENE_COMPOSE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,  # free-joint articulation N>1
        Trait.PROPS_ACTUATED_ACTIONS,  # prop-name NamedAction -> servo
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,  # manager-level record override
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,  # metric z from rc.depth_data
        Trait.CAMERAS_PER_ENV,  # CPU mirror per world
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CAMERAS_BATCH_RENDER,  # native warp renderer
        Trait.RECORDER_VIDEO,  # manager cache -> mp4
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,  # N=1 mirror + push
        Trait.DIRECTOR_ATTACH_WELD,
        Trait.DIRECTOR_GRIPPER,
        Trait.MOCAP,
        Trait.SPAWN_XY_YAW,
        Trait.SNAPSHOT,
        Trait.STATE_WRITE,  # CPU mirror write + per-world device push
        # Same compiled MjModel as mujoco
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
        Trait.MODEL_EDIT,  # global body/geom parameter transaction
        Trait.RUNTIME_QUERY,  # on-demand query face over the CPU mirror
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

ISAACLAB2_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N scene cloning
        Trait.ENV_LAYOUT_TILED,  # env_spacing clone grid
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,
        Trait.DEPTH_OBS_ALIASES,
        # CAMERAS_GLOBAL is the 1-prim mechanism (num_instances = parent prim
        # matches = 1); upstream docs do not name it -- container-verify.
        Trait.CAMERAS_GLOBAL,
        Trait.CAMERAS_PER_ENV,
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CAMERAS_BATCH_RENDER,
        Trait.RECORDER_VIDEO,
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,
        Trait.PROPS_ARTICULATION_USD,
        Trait.PROPS_LATCH,
        Trait.PROPS_ACTUATED_ACTIONS,
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,
        Trait.DIRECTOR_ATTACH_SLAVE,
        Trait.DIRECTOR_GRIPPER,
        Trait.SCENE_COMPOSE,
        Trait.TERRAIN,
        Trait.EVENTS_DR,
        Trait.INIT_POSE,
        Trait.SPAWN_XY_YAW,
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        # The USD prim hierarchy is the isaaclab model face
        Trait.MODEL_USD_PRIMS,
        Trait.MODEL_EDIT,  # body mass/COM/inertia across every env row
        Trait.RUNTIME_QUERY,  # on-demand model + live-pose query face
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

# IsaacLab 3 identities: PhysX and Newton share one implementation package
# but own separate runtime stacks and capability sets.  C2-1 lights up the
# verified shared core conservatively; prop traits stay undeclared until
# their backend tests land (P2).  The rigid-prop path is lane-tested, but the
# shared props bit projects all three PROPS_* traits, so its declaration is
# deferred until the capability model can express it.  The PhysX director,
# camera and recorder paths are lane-proven, so those traits are declared;
# weld stays absent until an engine-hosted weld test exists.
_IL3_CORE = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N scene cloning
        Trait.ENV_LAYOUT_TILED,  # env_spacing clone grid
        Trait.SCENE_COMPOSE,  # neutral Scene -> composed asset scene
        Trait.TERRAIN,
        Trait.INIT_POSE,
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        Trait.MODEL_USD_PRIMS,  # USD prim hierarchy is the isaaclab model face
        Trait.RUNTIME_QUERY,  # on-demand model + live-pose query face
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

ISAACLAB3_PHYSX_TRAITS = _IL3_CORE | frozenset(
    {
        Trait.EVENTS_DR,  # PhysX profile owns the CoM DR write
        Trait.CONTACTS_GROUPS,  # contacts: all world-frame groups
        Trait.DIRECTOR_TELEPORT,  # root-pose writes on a rigid prop
        Trait.DIRECTOR_SETVEL,  # root-velocity writes on a rigid prop
        Trait.DIRECTOR_ATTACH_SLAVE,  # per-tick kinematic follow
        Trait.DIRECTOR_GRIPPER,  # kinematic gripper pose face
        Trait.CAMERAS_RGB,  # RTX CameraCfg rigs
        Trait.CAMERAS_DEPTH_RIG,  # distance_to_camera annotator
        Trait.CAMERAS_GLOBAL,  # global request-axis world rig
        Trait.CAMERAS_PER_ENV,  # cloned sensor per env
        Trait.CAMERAS_PER_ENV_STATE,  # per-(camera, env) pose state
        Trait.RECORDER_VIDEO,  # camera_ref capture -> MP4
    }
)

ISAACLAB3_NEWTON_TRAITS = _IL3_CORE

# mujoco's model-global writes are one-MjModel only (no per-env
# face); they stay declared traits but are unavailable at N>1.  mjwarp
# declares the same N=1 scope (director writes the CPU mirror + pushes the
# device; one MjModel / one mocap namespace).  isaaclab2 supports these at
# N>1 (no scope); the isaaclab3 neutral Director writes env 0 only.
_MODEL_GLOBAL = "model-global writes (one MjModel): no per-env face"
_MJWARP_MODEL_GLOBAL = "model-global writes (one MjModel); mirror-only at N>1"
MUJOCO_SCOPES: dict[Trait, TraitScope] = {
    Trait.DIRECTOR_TELEPORT: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_SETVEL: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_ATTACH_WELD: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_GRIPPER: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.MOCAP: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
}
MJWARP_SCOPES: dict[Trait, TraitScope] = {
    Trait.DIRECTOR_TELEPORT: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_SETVEL: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_ATTACH_WELD: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_GRIPPER: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.MOCAP: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
}


# --------------------------------------------------------------------------- #
# canonical names
# --------------------------------------------------------------------------- #
CANONICAL_BACKENDS: tuple[str, ...] = (
    "mujoco",
    "mjwarp",
    "isaaclab2",
    "isaaclab3_physx",
    "isaaclab3_newton",
)

#: canonical name of the MuJoCo-Warp backend (the single truth callers key
#: config/CLI binding tables on; never hand-write the literal elsewhere).
MJWARP_BACKEND: str = "mjwarp"


def known_backends() -> list[str]:
    """Canonical backend names (registry order, no probing)."""
    return list(CANONICAL_BACKENDS)


def build_name(name: str) -> str:
    """Validate a canonical build spelling (no probing, no aliases).

    Only :data:`CANONICAL_BACKENDS` is accepted.  ``None``, ``""``,
    ``"isaaclab"``, ``"isaaclab3"`` and unknown spellings raise a teaching
    :class:`ValueError` listing the five canonical names.
    """
    raw = "" if name is None else str(name).strip().lower()
    if raw in CANONICAL_BACKENDS:
        return raw
    raise ValueError(
        f"unknown env backend {name!r}; available backends: "
        f"{', '.join(CANONICAL_BACKENDS)} "
        f"(canonical names only; 'isaaclab'/'isaaclab3' aliases were removed)"
    )


# --------------------------------------------------------------------------- #
# pure trait/scope validation (no backend factories)
# --------------------------------------------------------------------------- #
def validate_trait_scopes(
    name: str, traits: Collection[Trait], scopes: Mapping[Trait, TraitScope]
) -> None:
    """Validate that *scopes* only narrow traits already declared in *traits*."""
    declared = set(traits)
    for trait, scope in scopes.items():
        if trait not in declared:
            raise RuntimeError(
                f"backend {name!r}: scope declared for undeclared trait "
                f"{trait.name} (scopes only narrow presence)"
            )
        if scope.max_envs is not None and scope.max_envs < 1:
            raise RuntimeError(
                f"backend {name!r}: trait {trait.name} scope max_envs "
                f"must be >= 1, got {scope.max_envs}"
            )


def validate_trait_relations(name: str, traits: Collection[Trait]) -> None:
    """Validate layout exclusivity and ``TRAIT_RULES`` for *traits*.

    Empty trait sets are skipped by the caller: the IsaacLab 3 skeletons
    declare no layout because they implement nothing yet.
    """
    declared = set(traits)
    layouts = [trait for trait in _LAYOUT_TRAITS if trait in declared]
    if len(layouts) != 1:
        raise RuntimeError(
            f"backend {name!r}: exactly one ENV_LAYOUT_* trait is "
            f"required, got {[trait.name for trait in layouts]}"
        )
    for trait in sorted(declared, key=lambda item: item.value):
        rule = TRAIT_RULES.get(trait)
        if rule is None:
            continue
        for other in rule.exclusive_with:
            if other in declared:
                raise RuntimeError(
                    f"backend {name!r}: trait {trait.name} is "
                    f"exclusive with {other.name} but both are declared"
                )
        for required in rule.requires:
            if required not in declared:
                raise RuntimeError(
                    f"backend {name!r}: trait {trait.name} requires "
                    f"{required.name}, which is not declared"
                )
