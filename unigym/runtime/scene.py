"""Runtime scene face: the ``scene`` typed query (cmd 16).

One curated scene document per query in two tiers:

* ``tree``  -- STATIC structure, computed lazily ONCE (props/contact
  groups/cameras do not change short of a scene rebuild, which does not
  exist anywhere in the engine API -- so one computation serves the run's life);
* ``state`` -- LIVE dynamics, sampled fresh per query (the frame-telemetry
  discipline: observation happens in the query path).

Zero-engine boundary: serializers read the existing model/data state
objects only; no engine writes anywhere on this path (gate 6).  The
curated depth aligns with the shard visible face and the director
vocabulary (world -> robot / props / contact groups / cameras); robot
body-level detail deliberately stays out.

Schema (contract: docs/contracts/daemon_api_contract.md §1.23)::

    {"tree":   {"robot":    {"name", "base", "n_joints",
                              "joint_groups": {legs|waist|arms|hands: [...]},
                              "joints": [{name, group, range, limited}]},
                "props":    {name: {"type", "joints": [{name, range,
                                                        limited}] | [],
                                    "shape"}},
                "contacts": {name: {"members": [...]}},
                "cameras":  [names]},
     "state":  {"robot":    {"base_pos_w": [3], "base_quat_w": [4] wxyz,
                              "joints": {name: {"pos", "vel"}}},
                "props":    {name: {"pos", "quat", "joints": {n:{pos,vel}}}},
                "contacts": {name: {"on": bool, "force_z": float|null}}}}

``?full=1`` escape hatch swaps the curated face for the backend's raw
full tree (mujoco body recursion / isaaclab data-level inventory); the
UI never consumes it.  Backend mechanics live in the backend packages
(``backends/mujoco/scene.py`` / ``backends/isaaclab2/scene.py``) behind
the :class:`SceneOps` seam -- the ``director/ops.py`` pattern; this
module stays engine-free (numpy only inside the shared contact sampler).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from unigym._typing import JsonValue

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from numpy.typing import NDArray

    from unigym.contacts.manager import ContactManager
    from unigym.runtime.bridge import Bridge

__all__ = [
    "SceneOps",
    "SceneFace",
    "build_scene_ops",
    "attach",
    "robot_joint_groups",
    "joint_limit_rows",
    "contact_groups_state",
]


def require_robot_name(robot_name: object) -> str:
    """Validate the explicit, caller-owned SceneOps robot name.

    There is no backend fallback or environment inference: the caller (the run
    build) owns the articulation name, and an empty/missing name is a boundary
    error.
    """
    if (
        not isinstance(robot_name, str)
        or not robot_name.strip()
        or robot_name != robot_name.strip()
    ):
        raise ValueError(
            "scene face: robot_name must be an explicit non-empty string "
            f"without surrounding whitespace, got {robot_name!r}"
        )
    return robot_name


@runtime_checkable
class SceneOps(Protocol):
    """Per-backend scene mechanics (the director-ops seam).

    Implementations: ``backends/mujoco/scene.MujocoSceneOps`` /
    ``backends/isaaclab2/scene.IsaacLabSceneOps``.  All methods are
    read-only over the env's existing state objects.
    """

    backend: str

    def tree(self) -> dict[str, JsonValue]:
        """Curated static structure (cached by the face, not here)."""
        ...

    def state(self) -> dict[str, JsonValue]:
        """Live dynamics, sampled fresh per call."""
        ...

    def full_tree(self) -> dict[str, JsonValue]:
        """``?full=1`` raw face.

        Unbounded-ish; the 8KB ack cap is the honest failure mode --
        never clipped silently.
        """
        ...


def build_scene_ops(
    env: object,
    backend: str,
    robot_name: str,
    camera_names: Iterable[str] | None = None,
) -> SceneOps:
    """Factory: the ops object for *env* (single dispatch point).

    ``robot_name`` is the caller-owned articulation name; it is required and
    validated here before any backend import.  The implementations live in the
    backend packages -- lazy function-body imports keep this module engine-free.
    """
    from ..backends.registry import BackendSelectionError, backend_info

    name = require_robot_name(robot_name)
    try:
        info = backend_info(backend)
    except BackendSelectionError as e:
        raise ValueError(f"scene face: unknown backend {backend!r} " "(mujoco | isaaclab)") from e
    if info.scene_ops is None:
        raise ValueError(f"scene face: backend {backend!r} declares no " "scene_ops face")
    # The registry entry is a no-arg factory returning the backend's SceneOps
    # class; the runtime check establishes the callable invariant the type
    # system cannot express across the registry's opaque factory boundary.
    loaded = info.scene_ops()
    if not callable(loaded):
        raise TypeError(f"scene face: backend {backend!r} scene_ops factory is not callable")
    scene_ops_factory = cast(Callable[..., SceneOps], loaded)
    return scene_ops_factory(env, robot_name=name, camera_names=camera_names)


# --------------------------------------------------------------------------- #
# shared curated helpers (backend-free; both ops implementations reuse these)
# --------------------------------------------------------------------------- #
def robot_joint_groups(
    joint_names: Iterable[str],
    joint_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Curated joint groups folded into the robot tree.

    The group vocabulary IS the profile's ``joint_sets``: every named set is
    intersected with the run's actual joint set (run order preserved,
    unknown names dropped) and empty groups are omitted -- an asset keeps
    the face honest instead of listing joints it does not have.
    """
    have = list(joint_names)
    groups: dict[str, list[str]] = {}
    for raw_name, declared in (joint_sets or {}).items():
        names = [joint for joint in have if joint in declared]
        if names:
            groups[str(raw_name)] = names
    return groups


def joint_limit_rows(
    names: Sequence[str],
    limits: Mapping[str, tuple[float, float] | None],
    joint_sets: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, JsonValue]]:
    """``tree.robot.joints`` / prop-articulation limit rows.

    ``names``: ordered joint names -- ALL ACTUATED robot joints for the
    robot face (the full articulation table, hands
    included -- the 29-joint policy table is a subset the action face
    carries; contract §1.23).
    ``limits``: ``{name: (lo, hi) | None}`` from the backend's model
    face; ``None`` = the engine does not limit the joint (or the
    backend has no honest limit source) -> ``limited: False`` with a
    null range: no declared limit, no gate (nothing is guessed).
    ``group`` folds the curated robot groups (null outside them).
    """
    groups = robot_joint_groups(names, joint_sets)
    gname = {j: g for g, js in groups.items() for j in js}
    rows: list[dict[str, JsonValue]] = []
    for n in names:
        lim = limits.get(n)
        rows.append(
            {
                "name": n,
                "group": gname.get(n),
                "range": ([float(lim[0]), float(lim[1])] if lim is not None else None),
                "limited": lim is not None,
            }
        )
    return rows


def _to_numpy(x: object) -> NDArray[Any] | None:
    """torch/numpy -> numpy (cpu); None passthrough."""
    if x is None:
        return None
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu().numpy()
    import numpy as np

    return np.asarray(x)


def _any_truthy(val: object) -> bool:
    """``any()`` over a group value in its native face.

    torch -> cpu numpy -> object-with-any -> numpy; a missing value is
    False.
    """
    if val is None:
        return False
    detach = getattr(val, "detach", None)
    if detach is not None:  # torch tensor (isaaclab face)
        return bool(detach().cpu().numpy().any())
    anyfn = getattr(val, "any", None)
    if anyfn is not None:
        return bool(anyfn())
    import numpy as np

    return bool(np.asarray(val).any())


def contact_groups_state(cm: ContactManager | None) -> dict[str, JsonValue]:
    """Curated contact-group state from a bound ContactManager.

    ``on`` comes from the manager's FORMATTED value (respects each
    group's declared format/threshold -- the same number the record
    face serves); ``force_z`` is the peak world-frame |z| across the
    group's RAW force array (``backend.read_raw``, a read-only scan of
    the engine's existing contact data).  A group without raw support
    keeps ``force_z`` honestly null.
    """
    if cm is None:
        return {}
    vals: Mapping[str, object] = cm.read()
    out: dict[str, JsonValue] = {}
    read_raw = getattr(cm.backend, "read_raw", None)
    for spec in cm.specs:
        spec_name = getattr(spec, "name")
        on = _any_truthy(vals.get(f"contact/{spec_name}"))
        fz = None
        try:
            if read_raw is None:
                raise NotImplementedError
            raw = _to_numpy(read_raw(spec))
            if raw is not None and raw.size:
                fz = float(abs(raw[..., 2]).max())
        except (AttributeError, NotImplementedError, RuntimeError):
            fz = None  # honest null (contract §1.23)
        out[spec_name] = {"on": on, "force_z": fz}
    return out


# --------------------------------------------------------------------------- #
# SceneFace: the cmd-16 handler (curated face + lazy tree cache)
# --------------------------------------------------------------------------- #
class SceneFace:
    """Worker-side scene face: curated doc + ``?full=1`` escape hatch.

    The tree is computed on the FIRST query and cached (props/contact
    groups/cameras are start-time constants); state is sampled per
    query.  Attach via :func:`attach` (collect_cli), which registers the
    ``scene`` bridge command -- the console/watchers/director pattern.
    """

    def __init__(
        self,
        env: object,
        backend: str,
        robot_name: str,
        camera_names: Iterable[str] | None = None,
    ):
        self._env = env
        self._backend = str(backend)
        self._robot_name = require_robot_name(robot_name)
        self._camera_names = list(camera_names or [])
        self._ops: SceneOps | None = None
        self._tree: dict[str, JsonValue] | None = None
        self.n_queries = 0

    @property
    def ops(self) -> SceneOps:
        """The backend SceneOps (built lazily on first access)."""
        if self._ops is None:
            self._ops = build_scene_ops(
                self._env,
                self._backend,
                robot_name=self._robot_name,
                camera_names=self._camera_names,
            )
        return self._ops

    def tree_face(self) -> dict[str, JsonValue]:
        """Lazy cached tree (computed at most once per run).

        The engine has no scene-rebuild operation, so there is nothing
        to invalidate.
        """
        if self._tree is None:
            self._tree = self.ops.tree()
        return self._tree

    def handle(
        self, rid: int, params: Mapping[str, JsonValue] | None = None
    ) -> dict[str, JsonValue]:
        """Cmd-16 handler: curated doc, or ``full`` raw tree on demand."""
        self.n_queries += 1
        if params and params.get("full"):
            return {"scene": {"backend": self._backend, "full": self.ops.full_tree()}}
        return {"scene": {"tree": self.tree_face(), "state": self.ops.state()}}


def attach(bridge: Bridge, env: object, backend: str, robot_name: str) -> SceneFace:
    """Wire the scene face onto the bridge (collect_cli face).

    ``robot_name`` is the explicit articulation name (no fallback).  Registers
    the cmd-16 handler via the bridge's extension surface (``register_command``
    -- the console/watchers/director pattern); camera names ride from the
    bridge's own camera manager view.
    """
    face = SceneFace(
        env, backend, robot_name=robot_name, camera_names=getattr(bridge, "camera_names", None)
    )
    bridge.register_command("scene", face.handle)
    return face
