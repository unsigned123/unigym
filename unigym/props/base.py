"""PropManager: interactive prop (door / ball / generic) scene + state manager.

Loads interactive props that expose per-step state to ``obs_cache`` for adapters
to consume. Each prop *type* is spawned once per environment (cloned by
GridCloner); at
reset the manager re-samples a random placement ``(x, y, z, yaw)`` from the
prop's :class:`PropPlacement` and parks unassigned instances far below ground.

Prop kinds (see articulation.py / ball.py / generic.py):
  - ``articulation``  : anchored articulated prop (articulation.py). The
    asset (URDF/MJCF fragment) defines the joints;
    the root body is FIXED at the sampled placement (no freejoint).
    Per-joint ``mode: passive|actuated`` is declared in the spec
    (``joints:``); actuated joints join the NamedAction surface
    automatically. State exposes the anchor root pose (object/* record
    keys) and ``object/<name>/joints/<jname>`` per declared joint.
  - ``free_articulation`` : free-floating articulated prop (T-B3
    GAP-A, articulation.py FreeArticulationProp). The asset's ROOT
    body KEEPS its freejoint (a pushable table object); internal
    passive joints are declared like ``articulation`` (``init`` reset
    face included); ``placement`` is the freejoint qpos0/reset pose.
    Two asset forms: a fragment file carrying the freejoint (the
    laptop family), or a generated payload whose
    freejoint the backend INJECTS -- the jointless robocasa data
    objects. Both mujoco-only (the isaaclab face refuses with an
    error+hint; generated payloads refuse at setup_scene).
  - ``rigid_object``   : URDF with no joints -> :class:`RigidObjectCfg`.
  - ``sphere``         : no URDF; :class:`SphereCfg` spawn -> :class:`RigidObjectCfg`.
    State exposes ball_pos_b / ball_vel_b (ball.py).
  - ``usd``            : native USD asset (generic.py) -- isaaclab
    spawns it via ``UsdFileCfg`` (materials/textures render as authored);
    the mujoco backend substitutes geometry per the declared representation
    (mjcf_file / box_texture / box -- see backends/shared/mjfamily/props/props.py).

The manager is domain-neutral : it knows nothing about hands,
grasps or held semantics.  Any hand-contact criterion is a DECLARED contact
group (yaml ``contacts:`` with ``{body_parts: ...}`` / ``{prop: ...}``
references -- contacts/base.py) consumed by the ContactManager; the
``object/<name>/{pos,quat}`` record keys (props/object_state.py) are the
only object-key family the manager emits.

The manager is isaaclab-free at import time (cfg classes resolved lazily via
``_resolve``); unit tests inject fake classes through ``classes=``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    import torch

    from ..adapters.base import RawObsRequest
    from unigym.assets.plan import AssetPlan

from .asset_ir import AssetDefinition, iter_asset_bodies, joint_names


# --------------------------------------------------------------------------- #
# PropSpec
# --------------------------------------------------------------------------- #
@dataclass
class PropSpec:
    """Configuration for a single interactive prop."""

    name: str
    """Prop identifier, e.g. ``"door"`` / ``"ball"``."""
    type: str
    """``"articulation"`` / ``"rigid_object"`` / ``"sphere"`` / ``"usd"``
    / ``"free_articulation"``."""
    asset: str | None = None
    """Articulation asset path (``type: articulation``): a URDF or
    MJCF fragment whose joints are defined INSIDE the asset.  The root body
    is FIXED to the world at the sampled placement (anchored semantics --
    a fragment carrying a freejoint fails fast), so there is no freejoint
    and no ``spawn.joint_params`` any more: per-joint behaviour is declared
    in :attr:`joints`."""
    joints: dict = field(default_factory=dict)
    """Articulation joint declarations (``type: articulation``):
    ``{jname: {mode: passive|actuated, damping?, stiffness?, range?, init?}}``.
    fail-fast: ``actuated`` requires ``stiffness``; a ``passive`` damping
    is OPTIONAL (an asset-driven joint -- e.g. the
    composite chains' springdamper -- keeps the asset's own dynamics when
    the declaration is silent); every asset joint must be declared
    (checked at backend bind time, when the asset joint names are known).
    Actuated joints automatically join the action surface as the named
    entry ``{prop_name: NamedAction}``; passive joints get damping
    injection only (no action face).  ``range`` (optional, [lo, hi]) feeds
    the mujoco position-servo ctrlrange; the isaaclab side keeps the asset
    limits. both modes also accept optional ``effort_limit``
    (ImplicitActuatorCfg.effort_limit_sim) and a ``passive`` joint may keep
    a declared ``stiffness`` (USD-authored drive semantics without joining
    the action surface -- the HA door handle).  ``init`` (T-B1, optional,
    default 0.0) is the declared RESET joint position, fail-fast outside a
    declared range (the backend build re-checks against the asset range
    otherwise); both backends write it on every reset -- the official
    fixture semantics (robocasa ``set_door_state``) start fixtures at a
    declared open state while the passive joint stays freely movable."""
    latch: dict = field(default_factory=dict)
    """Runtime drive-switch declaration (``type: articulation`` /
    ``usd_articulation``)::
        latch:
          lock_joint: <declared joint>     # drive gets switched
          sense_joint: <declared joint>    # threshold-gating angle (deg)
          unlock_threshold_deg: -20.0      # negative: unlock when <= it
          locked: {stiffness: 1e5, damping: 5e4, friction: 1e4}
          unlocked: {...}                  # optional (defaults = declared)
    Normalized + validated by props.articulation.validate_latch (fail-fast
    on undeclared joints / an empty ``locked`` block / a non-finite
    threshold).  isaaclab-only : the mujoco articulation handler
    rejects a latch declaration at build time."""
    urdf_path: str | None = None
    """URDF path (required for rigid_object; unused for sphere; ``None`` for
    the single-box ``box`` kind, which spawns a native ``CuboidCfg`` on
    isaaclab; container/table keep a generated URDF)."""
    spawn: dict = field(default_factory=dict)
    """Spawn parameters.

    sphere: ``{radius, mass, color, friction, restitution}``.
    rigid_object: ``{fix_base}``.
    """
    placement: dict = field(default_factory=dict)
    """``{mode, x_range, y_range, z_range, yaw_range(deg), height_offset}``."""
    state_keys: list[str] = field(default_factory=list)
    """obs_cache keys this prop exposes (e.g. ``["door_pos", "door_angle"]``)."""
    init_roll: float = 0.0
    """Root roll (rad) baked into the spawn init quaternion; 0 for z-up URDFs."""
    size: list | None = None
    """Box-family kinds (container/table/box): outer envelope ``[sx, sy, sz]``
    (m); the prop origin is the outer bottom-face center.  Required for those
    kinds; ignored otherwise.  For the dynamic primitive ``box`` kind the same
    envelope is used with a centre origin (see :mod:`unigym.props.primitive`)."""
    radius: float | None = None
    """Primitive kinds (``sphere`` / ``cylinder`` / ``capsule``): radius in
    metres.  ``sphere`` falls back to ``spawn.radius`` when omitted (the
    long-standing ball declaration keeps working unchanged)."""
    half_height: float | None = None
    """Primitive ``cylinder`` / ``capsule``: half-length of the axial
    section in metres (local z, matching the scene primitives).  A capsule's
    resting centre height is ``radius + half_height``."""
    material: str | None = None
    """Logical material name carried by a dynamic primitive declaration.

    Props have no scene-material table of their own: the name is retained on
    the specification for declaration identity / round-trip fidelity and is
    not translated into an engine material (use ``spawn.rgba`` /
    ``spawn.texture`` for cosmetic control)."""
    aabb: dict = field(default_factory=dict)
    """Box-family kinds: declared axis-aligned box ``{center: [x,y,z],
    half_extents: [hx,hy,hz]}`` relative to the prop origin (container: the
    INNER cavity).  Exposed via :meth:`PropManager.aabbs` for FormulaTask
    ``point_in_box_fraction`` parameters.  Required for container/table."""
    usd_path: str | None = None
    """USD path (``type: usd``): a native USD asset spawned
    directly via ``UsdFileCfg`` on the isaaclab backend -- materials/textures
    embedded in (or relatively referenced by) the USD render as authored
    (prep the asset sim-ready with ``scripts/usd_prep_prop.py``: SIMPLE
    USDs carry their rigid APIs on inner prims only).  The mujoco backend
    has no USD loader: the ``mujoco`` field declares how the prop is
    represented there.  Resolution follows the shared path rules (absolute
    as-is / relative to the declaring yaml's directory)."""
    mujoco: dict = field(default_factory=dict)
    """mujoco representation of a ``type: usd`` prop
    (backends/shared/mjfamily/props/props.py) -- three declared tiers.  A
    ``rigid_object`` prop may declare the same block on the mujoco
    backend (T2.4: URDF-less mesh props -- the fragment carries the mass
    /inertia; isaaclab keeps its URDF semantics and needs ``urdf_path``
    for that kind)::

        mujoco:
          representation: mjcf_file   # mjcf_file | box_texture | box
          mjcf_file: <path>           # required iff mjcf_file (converted
                                      # MJCF fragment; scripts/usd_to_mjcf.py)
          texture: <png>              # required iff box_texture (2d albedo)
          mass: 0.2                   # box tiers' total mass
          friction: 1.0               # optional (box tiers)

    ``mjcf_file`` = high-fidelity converted fragment (its worldbody must
    hold exactly ONE body named after the prop, carrying one freejoint);
    ``box_texture`` = declared ``size`` envelope box + MJCF texture/
    material; ``box`` = plain envelope box.  A usd prop WITHOUT a mujoco
    section fails fast on the mujoco backend listing the three tiers
    (the declaration is mandatory -- silent geometry substitution would
    hide the domain gap).  Ignored by the isaaclab backend."""


def _prop_shape(spec) -> str:
    """Coarse shape family for the curated scene tree (honest bucket).

    Declared-size box family -> box; usd asset -> usd; sphere -> sphere;
    urdf/mjcf asset -> mesh.  Pure ``PropSpec`` read: both
    backend scene faces import this one neutral definition.
    """
    if getattr(spec, "usd_path", None):
        return "usd"
    if getattr(spec, "size", None):
        return "box"
    if spec.type == "sphere":
        return "sphere"
    return "mesh"


# --------------------------------------------------------------------------- #
# backward-compatible lazy spawner names
# --------------------------------------------------------------------------- #
# The isaaclab builders moved to ``backends/isaaclab2/props/spawn.py``.  The
# old names stay importable from here for existing callers/tests; the import is
# resolved lazily (and dynamically) so this neutral module keeps no static
# backend edge and imports no engine at module load.
def _spawn_module():
    """Load the isaaclab spawner module on first use."""
    import importlib

    return importlib.import_module("unigym.backends.isaaclab2.props.spawn")


def _resolve(classes):
    """Compat: the isaaclab cfg-class table (lazy import when classes is None)."""
    return _spawn_module()._resolve(classes)


def _build_urdf_spawn(spec: PropSpec, c: dict):
    """Compat: build the isaaclab UrdfFileCfg spawn."""
    return _spawn_module()._build_urdf_spawn(spec, c)


def _build_usd_spawn(spec: PropSpec, c: dict):
    """Compat: build the isaaclab UsdFileCfg spawn."""
    return _spawn_module()._build_usd_spawn(spec, c)


def _build_actuators(spec: PropSpec, c: dict) -> dict:
    """Compat: build implicit actuator cfgs from spawn.joint_params."""
    return _spawn_module()._build_actuators(spec, c)


def _prop_backend(backend: str | None = None) -> PropBackendCap:
    """Resolve a backend's prop face through the registry.

    ``backend=None`` resolves the historical isaaclab build face (the
    direct-construction default); the registry factory keeps every engine
    import inside the backend module.
    """
    from ..backends.registry import backend_info, build_name

    info = backend_info(build_name(backend))
    factory = info.prop_backend
    if factory is None:
        raise ValueError(
            f"backend {build_name(backend)!r} has no prop backend face "
            f"(hint: use mujoco/mjwarp/isaaclab)"
        )
    return cast("PropBackendCap", factory())


def _compat_class(name: str) -> type:
    """Resolve a legacy public handler class name."""
    return _prop_backend(None).compat(name)


def _resolve_urdf_path(path: str, base_dir: str | None) -> str:
    """Resolve a (possibly relative) urdf path to an absolute path.

    Try ``base_dir`` then its parent so ``props/door.urdf`` resolves from
    ``config/``.
    """
    if os.path.isabs(path) or "://" in path:
        return path
    if base_dir is None:
        base_dir = os.getcwd()
    candidates = [
        os.path.join(base_dir, path),
        os.path.join(os.path.dirname(base_dir), path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return os.path.abspath(candidates[0])


# --------------------------------------------------------------------------- #
# quaternion helpers (wxyz, world-frame composition: q = q_yaw (x) q_roll)
# DELEGATED to unigym.core.quat (M1 convergence; bodies live there now)
# --------------------------------------------------------------------------- #
def _roll_quat(roll: float) -> torch.Tensor:
    import torch  # noqa: F401  (lazy: import unigym.props stays torch-free)
    from ..core import quat

    return quat.about_x(roll)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    import torch  # noqa: F401  (lazy: import unigym.props stays torch-free)
    from ..core import quat

    return cast(torch.Tensor, quat.about_z(yaw))


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a (x) b (both (N,4) wxyz)."""
    import torch  # noqa: F401  (lazy: import unigym.props stays torch-free)
    from ..core import quat

    return cast(torch.Tensor, quat.mul(a, b))


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q) . v for q=(w,x,y,z) (N,4), v=(N,3). Body -> world."""
    import torch  # noqa: F401  (lazy: import unigym.props stays torch-free)
    from ..core import quat

    return quat.rotate(q, v)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q)^T . v for q=(w,x,y,z) (N,4), v=(N,3). World -> body.

    Matches IsaacGym/Isaac Lab ``quat_rotate_inverse``.
    """
    import torch  # noqa: F401  (lazy: import unigym.props stays torch-free)
    from ..core import quat

    return quat.rotate_inv(q, v)


# --------------------------------------------------------------------------- #
# PropPlacement: per-prop (x, y, z, yaw) sampler
# --------------------------------------------------------------------------- #
class PropPlacement:
    """Sample prop placements ``(x, y, z, yaw_rad)`` per environment.

    This supports an explicit ``z_range`` (e.g. a ball spawned at its radius
    height) and
    optional ``yaw_range`` (a round ball has no yaw, so it may be omitted).
    ``yaw_range`` is configured in degrees for human-friendly yaml.
    """

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        self.mode = str(config.get("mode", "random"))
        if self.mode not in ("random", "fixed"):
            raise ValueError(f"placement mode must be 'random' or 'fixed', got '{self.mode}'")
        self.x_range = tuple(config.get("x_range", [0.0, 0.0]))
        self.y_range = tuple(config.get("y_range", [0.0, 0.0]))
        self.z_range = tuple(config.get("z_range", [0.0, 0.0]))
        yaw_deg = config.get("yaw_range")
        if yaw_deg is None:
            self.yaw_range = (0.0, 0.0)
        else:
            yaw_deg = tuple(yaw_deg)
            self.yaw_range = (math.radians(yaw_deg[0]), math.radians(yaw_deg[1]))
        # height_offset is an additive z shift (e.g. spawn above ground);
        # applied on top of the sampled z_range midpoint/sample.
        self.height_offset = float(config.get("height_offset", 0.0))
        # touch ranges so bad shapes raise early
        _ = (self.x_range, self.y_range, self.z_range)

    def sample(self, num_positions: int, device: str = "cpu") -> torch.Tensor:
        """Return a ``(num_positions, 4)`` float tensor ``(x, y, z, yaw_rad)``."""
        import torch  # lazy: import unigym.props stays torch-free

        if num_positions <= 0:
            return torch.zeros((0, 4), device=device)

        def _draw(rng):
            # uniform_(a, a) raises on some torch versions for equal bounds;
            # degenerate (single-value) ranges return a constant via full().
            lo, hi = rng
            if lo == hi:
                return torch.full((num_positions,), lo, device=device)
            return torch.empty(num_positions, device=device).uniform_(lo, hi)

        if self.mode == "random":
            x = _draw(self.x_range)
            y = _draw(self.y_range)
            z = _draw(self.z_range)
            yaw = _draw(self.yaw_range)
        else:  # fixed: midpoints
            x = torch.full(
                (num_positions,), 0.5 * (self.x_range[0] + self.x_range[1]), device=device
            )
            y = torch.full(
                (num_positions,), 0.5 * (self.y_range[0] + self.y_range[1]), device=device
            )
            z = torch.full(
                (num_positions,), 0.5 * (self.z_range[0] + self.z_range[1]), device=device
            )
            yaw = torch.full(
                (num_positions,), 0.5 * (self.yaw_range[0] + self.yaw_range[1]), device=device
            )
        z = z + self.height_offset
        return torch.stack([x, y, z, yaw], dim=-1)


# --------------------------------------------------------------------------- #
# PropHandler protocol
# --------------------------------------------------------------------------- #
class PropHandler:
    """Per-type prop handler (door / ball / generic).

    Each handler owns one :class:`PropSpec`, builds its scene cfg, binds the
    spawned asset after scene init, re-places it on reset, and exposes per-step
    state tensors. Subclasses must set ``self.spec`` and implement the four
    runtime hooks.
    """

    spec: PropSpec
    num_envs: int
    device: str

    def __init__(
        self, spec: PropSpec, num_envs: int, device: str, base_dir: str | None = None
    ) -> None:
        """Store the handler's declaration context (subclass hooks extend)."""
        self.spec = spec
        self.num_envs = int(num_envs)
        self.device = device

    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return ``(scene_field_name, cfg)`` to merge into the scene config."""
        raise NotImplementedError

    def bind_env(self, env) -> None:
        """Cache the spawned asset handle + per-spec roll quat (after init)."""
        raise NotImplementedError

    def get_world_pose(self, env):
        """``(pos (N,3), quat (N,4) wxyz)`` world root pose, or None.

        Consumed by the object/* record-key layer (props/object_state);
        handlers without a root pose (door) return None.
        """
        return None

    def on_reset(self, env_ids, env) -> None:
        """Re-sample placement + reset prop state for the given env subset."""
        raise NotImplementedError

    def get_state(self, env) -> dict[str, torch.Tensor]:
        """Return ``{state_key: (N, D) tensor}`` for this prop."""
        raise NotImplementedError

    @property
    def state_requests(self) -> list[RawObsRequest]:
        """Declare obs_cache keys (RawObsRequest format) for this prop."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# PropBackendCap protocol (backend-traits M2: the dual-track convergence)
# --------------------------------------------------------------------------- #
@runtime_checkable
class PropBackendCap(Protocol):
    """The backend-agnostic props capability surface (design §3.2).

    The two backends historically used two orchestration models (the root
    cause of interface drift):

    * isaaclab — **scene-cfg-first**: before scene cloning the backend
      produces ``{scene_field: cfg}`` entries merged into the scene config;
    * mujoco — **spec-first**: before compilation the backend mutates the
      ``MjSpec`` in place (bodies/actuators/weld slots), addressing tables
      resolve against the compiled model afterwards.

    ``PropBackendCap`` makes both forms explicit and first-class instead of
    implicit: a capability object exposes BOTH entry points, exactly one of
    which does work per backend (the other is a documented no-op).  The
    post-build lifecycle is single-form (``bind_env`` / ``on_reset`` /
    ``get_state``).  :class:`PropManager` implements this surface; the
    per-variant ``PropHandler`` chain (isaaclab) and the mujoco
    representation tiers hang off it.

    Orchestration unification beyond this (one shared model) stays a
    separate project (design §7.5) — this Protocol is the typed seam.
    """

    def scene_fields(self, manager, *, classes=None) -> dict:
        """Isaaclab form: ``{scene_field: cfg}`` to merge before cloning.
        (mujoco: ``{}`` — no scene elements.)

        The manager is passed explicitly because the backend face formats
        the manager's resolved handlers.
        """
        ...

    def inject_spec(self, manager, mjspec) -> None:
        """Mujoco form: in-place MjSpec injection before compilation.
        (isaaclab: no-op.)"""
        ...

    #: whether this face implements the MjSpec injection form
    injects_mjspec: bool

    def compat(self, name: str) -> type:
        """Resolve a legacy public handler class name (isaaclab compat)."""
        ...

    def build_handler(
        self, spec: PropSpec, *, num_envs: int, device: str, base_dir: str | None = None
    ) -> PropHandler:
        """Instantiate the declared handler for one spec."""
        ...

    def handler_for(self, spec: PropSpec) -> type[PropHandler]:
        """Return the handler class declared for ``spec.type`` (fail fast)."""
        ...

    def bind_env(self, env) -> None:
        """Post-build: resolve asset handles / address tables."""
        ...

    def on_reset(self, env_ids, env) -> None:
        """Re-sample placement + reset prop state for the env subset."""
        ...

    def get_state(self, env) -> dict:
        """``{state_key: (N, D) tensor}`` per-prop state surface."""
        ...


# --------------------------------------------------------------------------- #
# PropManager: orchestrates a list of PropHandlers
# --------------------------------------------------------------------------- #
def _require_path(
    path: str | None, kind: str, spec: PropSpec, base_dir: str | None, noun: str = "asset"
) -> str:
    """Resolve + existence-check one declared asset path (fail-fast)."""
    if not path:
        raise ValueError(f"{kind} prop {spec.name!r}: {noun} is required")
    resolved = _resolve_urdf_path(path, base_dir)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{kind} prop {spec.name!r}: asset not found: {resolved}")
    return resolved


def _resolve_spec_paths(
    spec: PropSpec, base_dir: str | None, asset: AssetDefinition | None = None
) -> None:
    """Per-kind declaration validation + path resolution (fail-fast).

    Extracted from ``PropManager.__init__`` (the CC-cap split; same
    checks, one kind per block).  The articulation family (anchored +
    free, T-B3 GAP-A) shares the asset face: the asset (urdf/mjcf
    fragment) is required and resolved through the shared path rules;
    per-joint behaviour validation lives in the handler.  The free
    kind's freejoint itself lives INSIDE the asset (the fragment gate
    requires it on the root body) and additionally requires at least
    one declared internal joint.

    A prop carrying a generated payload (``asset`` is not None)
    takes the generated branch instead: no asset path exists to
    resolve, and the joint-COVERAGE check runs HERE -- the IR joint
    names are known at resolve time, vs bind time for static assets.
    """
    if asset is not None:
        _resolve_generated_spec(spec, asset)
        return
    if spec.type == "articulation":
        spec.asset = _require_path(spec.asset, "articulation", spec, base_dir)
    elif spec.type == "free_articulation":
        if not spec.joints:
            raise ValueError(
                f"free_articulation prop {spec.name!r}: joints must "
                f"declare at least one (internal) joint"
            )
        spec.asset = _require_path(spec.asset, "free_articulation", spec, base_dir)
    else:
        if spec.type == "rigid_object" and spec.urdf_path:
            spec.urdf_path = _resolve_urdf_path(spec.urdf_path, base_dir)
        if spec.type == "rigid_object" and (spec.mujoco or {}).get("representation"):
            # URDF-less mesh declaration (T2.4): the mujoco backend
            # honors the same three representation tiers as the usd
            # kind ( mjcf_file = attachable fragment ); isaaclab keeps
            # its URDF semantics and needs urdf_path for this kind
            mj = spec.mujoco
            rep = mj.get("representation")
            if rep not in ("mjcf_file", "box_texture", "box"):
                raise ValueError(
                    f"rigid_object prop {spec.name!r}: unknown mujoco "
                    f"representation {rep!r} (allowed: mjcf_file | "
                    f"box_texture | box)"
                )
            if rep == "mjcf_file":
                if not mj.get("mjcf_file"):
                    raise ValueError(
                        f"rigid_object prop {spec.name!r}: "
                        f"representation mjcf_file requires "
                        f"mujoco.mjcf_file"
                    )
                p = _resolve_urdf_path(mj["mjcf_file"], base_dir)
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"rigid_object prop {spec.name!r}: mjcf_file " f"not found: {p}"
                    )
                spec.mujoco = dict(mj, mjcf_file=p)
        if spec.type == "usd_articulation":
            # native USD articulation (isaaclab-only kind --
            # the mujoco backend has no USD articulation loader; the
            # backend itself also fails fast at build time)
            spec.usd_path = _require_path(
                spec.usd_path, "usd_articulation", spec, base_dir, noun="usd_path"
            )
            if not spec.joints:
                raise ValueError(
                    f"usd_articulation prop {spec.name!r}: joints must "
                    f"declare at least one joint"
                )
        if spec.type == "usd":
            spec.usd_path = _require_path(spec.usd_path, "usd", spec, base_dir, noun="usd_path")
            mj = spec.mujoco or {}
            rep = mj.get("representation")
            if rep not in (None, "mjcf_file", "box_texture", "box"):
                raise ValueError(
                    f"usd prop {spec.name!r}: unknown mujoco "
                    f"representation {rep!r} (allowed: mjcf_file | "
                    f"box_texture | box)"
                )
            if rep == "mjcf_file":
                if not mj.get("mjcf_file"):
                    raise ValueError(
                        f"usd prop {spec.name!r}: representation "
                        f"mjcf_file requires mujoco.mjcf_file "
                        f"(scripts/usd_to_mjcf.py)"
                    )
                p = _resolve_urdf_path(mj["mjcf_file"], base_dir)
                if not os.path.exists(p):
                    raise FileNotFoundError(f"usd prop {spec.name!r}: mjcf_file not found: " f"{p}")
                spec.mujoco = dict(mj, mjcf_file=p)


#: the internal-joint vocabulary a generated free payload may carry
#: (v074: the zoo laptop lid hinge; passive damping-only dofs beside
#: the injected freejoint).
_PASSIVE_IR_JOINTS = frozenset({"hinge", "slide"})


def _resolve_generated_spec(spec: PropSpec, asset: AssetDefinition) -> None:
    """Generated-payload branch.

    The asset lives in the IR: there is no path to resolve.  Two shapes
    are supported: the anchored ``articulation`` kind (S2.0; every IR
    joint must be declared and every declaration must hit an IR joint
    -- the fragment bind-time coverage gate moved HERE, to resolve
    time), and the ``free_articulation`` data-object form: the renderer
    INJECTS the root freejoint (the prop's live
    dof, never declarable).  v074 relaxes the v1 jointless ruling: the
    payload may carry PASSIVE hinge/slide internal joints (the zoo
    laptop's ``lid_hinge``) with exact declared coverage.
    """
    if spec.type not in ("articulation", "free_articulation"):
        raise ValueError(
            f"prop {spec.name!r}: generated asset payloads support the "
            f"anchored 'articulation' kind or the jointless "
            f"'free_articulation' data-object form only, got {spec.type!r}"
        )
    if spec.asset is not None:
        raise ValueError(
            f"prop {spec.name!r}: a generated payload and an asset path "
            f"are mutually exclusive (got asset {spec.asset!r})"
        )
    declared = set(spec.joints or {})
    ir_joints = set(joint_names(asset))
    if spec.type == "free_articulation":
        # v074: the zoo laptop face -- a generated payload may carry
        # PASSIVE hinge/slide internal joints (the lid_hinge), declared
        # with exact coverage like the anchored kind; the injected
        # freejoint stays the prop's live root dof and is never
        # declarable.  Anything beyond hinge/slide is still refused.
        bad_types = sorted(
            j.name
            for b in iter_asset_bodies(asset)
            for j in b.joints
            if j.type not in _PASSIVE_IR_JOINTS
        )
        if bad_types:
            raise ValueError(
                f"free_articulation prop {spec.name!r}: a generated "
                f"payload carries passive hinge/slide internal joints "
                f"only (the injected freejoint is the prop's live dof); "
                f"got {bad_types}"
            )
        if declared != ir_joints:
            raise ValueError(
                f"free_articulation prop {spec.name!r}: the declared "
                f"joints must cover the payload's internal joints exactly "
                f"(declared {sorted(declared)} vs IR {sorted(ir_joints)})"
            )
        return
    missing = sorted(declared - ir_joints)
    undeclared = sorted(ir_joints - declared)
    if missing:
        raise ValueError(
            f"articulation prop {spec.name!r}: declared joints {missing} "
            f"not in the generated asset joints {sorted(ir_joints)}"
        )
    if undeclared:
        raise ValueError(
            f"articulation prop {spec.name!r}: generated asset joints "
            f"{undeclared} are not declared in spec.joints (every asset "
            f"joint must be declared)"
        )


class PropManager:
    """Interactive-prop scene + state manager."""

    def __init__(
        self,
        config: dict,
        num_envs: int,
        env_spacing: float,
        device: str = "cpu",
        base_dir: str | None = None,
        backend: str | None = None,
        asset_plan: AssetPlan | None = None,
    ):
        self.backend = backend
        self._prop_backend: PropBackendCap = _prop_backend(backend)
        self.num_envs = int(num_envs)
        self.env_spacing = float(env_spacing)
        self.device = device
        self.base_dir = base_dir
        self.specs: list[PropSpec] = []
        self.asset_definitions: dict[str, AssetDefinition] = {}
        """Generated payloads by prop name (empty = all-static)."""
        from .templates import resolve_prop_entries

        for entry, asset in resolve_prop_entries(config, asset_plan):
            spec = PropSpec(**entry)
            _resolve_spec_paths(spec, base_dir, asset=asset)
            self.specs.append(spec)
            if asset is not None:
                self.asset_definitions[spec.name] = asset
        # build handlers through the backend face (no neutral type branching)
        self._handlers: list[PropHandler] = [
            self._prop_backend.build_handler(
                spec, num_envs=self.num_envs, device=device, base_dir=base_dir
            )
            for spec in self.specs
        ]
        self._bound = False

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> dict:
        """Return ``{SceneCfg field: cfg}`` to merge into the scene config.

        A prop carrying a generated payload fails fast here (the isaaclab
        scene form): generated payloads are
        mujoco-only -- the free_articulation refusal precedent (an honest
        refusal beats a silently substituted asset).
        """
        if self.asset_definitions:
            raise ValueError(
                f"props {sorted(self.asset_definitions)}: generated asset "
                f"payloads are mujoco-only (the isaaclab scene spawner has "
                f"no generated-asset face) -- run this pack on backend: "
                f"mujoco"
            )
        return self._prop_backend.scene_fields(self, classes=classes)

    def scene_fields(self, *, classes=None) -> dict:
        """PropBackendCap isaaclab form (alias of :meth:`setup_scene`;
        mujoco-side callers never invoke it — see :meth:`inject_spec`).
        ``classes`` passes through the injectable cfg-class table (unit
        tests run this offline without isaaclab)."""
        return self.setup_scene(classes=classes)

    def inject_spec(self, mjspec) -> None:
        """PropBackendCap mujoco form: append the declared prop bodies to
        the MjSpec before compilation.

        The declaration handlers are mujoco-compatible even when the manager
        was built through the isaaclab face (the historical dual-track seam:
        direct ``build_prop_manager`` callers then hand the manager to a
        mujoco env), so a non-mujoco manager routes this form through the
        registry's mujoco prop face.
        """
        if not self._handlers:
            return
        face = self._prop_backend
        if not getattr(face, "injects_mjspec", False):
            face = _prop_backend("mujoco")
        face.inject_spec(self, mjspec)

    # ------------------------------------------------------------------ #
    # runtime
    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        for h in self._handlers:
            h.bind_env(env)
        self._bound = True

    def on_reset(self, env_ids, env) -> None:
        if not self._handlers or not self._bound:
            return
        for h in self._handlers:
            h.on_reset(env_ids, env)

    def on_step(self, env) -> None:
        """Per-control-step hook : latch polls live here.

        Handlers without an ``on_step`` are skipped;
        the env backend calls this once per step before the record extras are
        built, so a latch unlock from this step's poll is visible to the
        same step's ``object/<name>/joints/*`` keys.
        """
        if not self._handlers or not self._bound:
            return
        for h in self._handlers:
            hook = getattr(h, "on_step", None)
            if hook is not None:
                hook(env)

    def get_state(self, env) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for h in self._handlers:
            out.update(h.get_state(env))
        return out

    def get_world_poses(self, env) -> dict[str, tuple]:
        """``{name: (pos (N,3), quat (N,4) wxyz)}`` per prop (object layer)."""
        out = {}
        for h in self._handlers:
            pose = h.get_world_pose(env)
            if pose is not None:
                out[h.spec.name] = pose
        return out

    def get_articulation_joint_states(self, env) -> dict:
        """``{name: {jname: (N,1)}}`` for wired articulation handlers
        (state surface; consumed by object_state + the record
        extras).  Empty when no articulation prop is wired."""
        out: dict = {}
        for h in self._handlers:
            getter = getattr(h, "get_joint_positions", None)
            if getter is None:
                continue
            states = getter(env)
            if states:
                out[h.spec.name] = states
        return out

    def aabbs(self) -> dict[str, dict]:
        """Declared aabb metadata ``{name: {center, half_extents}}``.

        Only props that DECLARE an aabb (box-family kinds) appear; the values
        are plain lists (yaml-embeddable FormulaTask parameters, relative to
        the prop origin).
        """
        out: dict[str, dict] = {}
        for h in self._handlers:
            center = getattr(h, "aabb_center", None)
            half = getattr(h, "aabb_half", None)
            if center is None or half is None:
                continue
            out[h.spec.name] = {
                "center": [float(v) for v in center],
                "half_extents": [float(v) for v in half],
            }
        return out

    @property
    def has_usd_props(self) -> bool:
        """True when any declared prop is a USD-family kind (``usd`` /
        ``usd_articulation``) (isaaclab default lighting trigger:
        a textured asset without scene lights renders as a black
        silhouette -- collect_cli enables dome+distant by default unless
        ``env.lighting`` is declared explicitly)."""
        return any(s.type in ("usd", "usd_articulation") for s in self.specs)

    @property
    def state_requests(self) -> list[RawObsRequest]:
        out: list[RawObsRequest] = []
        for h in self._handlers:
            out.extend(h.state_requests)
        return out
