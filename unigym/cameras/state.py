"""Per-instance camera state (design §4).

Engine-free helpers shared by the camera manager and both backends.  The
``(camera_key, env_id | None)`` row table itself lives here; ``body`` rigs
never store poses (their pose is the env robot's FK ∘ mount offset).

Keeping these tiny helpers here (rather than inside ``manager.py`` /
``backends/mujoco/cameras.py``) keeps those already-large files above the
complexity gate's per-file MI floor.
"""
from __future__ import annotations

__all__ = ["CameraStateRow", "CameraStateTable", "apply_declared_state",
           "check_single_env", "make_state_table", "read_env_ids",
           "state_is_shared"]


class CameraStateRow:
    """One ``(camera, env)`` pose row (world convention, wxyz quat)."""

    __slots__ = ("pos", "quat")

    def __init__(self, pos=None, quat=None):
        self.pos = pos
        self.quat = quat

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CameraStateRow(pos={self.pos!r}, quat={self.quat!r})"


class CameraStateTable:
    """Per-instance camera pose state keyed by ``(camera_key, env_id|None)``.

    ``state='shared'`` cameras hold ONE row keyed ``(camera, None)`` -- the
    table ``broadcast``s it (the rig pushes that single row to the
    addressed sensor instances).  ``state='per_instance'`` cameras hold one
    row per ``(camera, env_id)``.  ``body`` rigs never store poses (their
    pose is the env robot's FK ∘ mount offset, derived every sync); the
    table is simply not written for them.

    The table is engine-free (torch/numpy values are opaque here) so the
    resolution rules are unit-testable without isaaclab/mujoco.
    """

    def __init__(self):
        self._rows: dict[tuple[str, int | None], CameraStateRow] = {}
        self._modes: dict[str, str] = {}

    # -- declaration ------------------------------------------------------ #
    def declare(self, camera: str, state: str) -> None:
        """Register a camera's state mode (``shared|per_instance``)."""
        if state not in ("shared", "per_instance"):
            raise ValueError(
                f"camera {camera!r}: state must be shared|per_instance, "
                f"got {state!r}")
        self._modes[str(camera)] = state

    def mode(self, camera: str) -> str:
        """Registered mode (unregistered cameras default ``shared``)."""
        return self._modes.get(str(camera), "shared")

    # -- addressing ------------------------------------------------------- #
    def _key(self, camera: str, env_id: int | None) -> tuple[str, int | None]:
        if self.mode(camera) == "shared":
            return (str(camera), None)
        return (str(camera), None if env_id is None else int(env_id))

    # -- writes ----------------------------------------------------------- #
    def set(self, camera: str, env_id: int | None, pos, quat) -> None:
        """Store one pose row (shared cameras always hit the ``None`` row)."""
        self._rows[self._key(camera, env_id)] = CameraStateRow(pos, quat)

    def broadcast(self, camera: str) -> int:
        """Copy the single shared row onto every existing per-env row.

        Returns the number of per-instance rows overwritten.  Raises
        ``KeyError`` when no shared row exists (a per-instance camera has
        nothing to broadcast -- its rows are independent).
        """
        key = (str(camera), None)
        row = self._rows.get(key)
        if row is None:
            raise KeyError(
                f"camera {camera!r}: no shared row to broadcast "
                f"(mode={self.mode(camera)!r})")
        n = 0
        for k in list(self._rows):
            if k[0] == str(camera) and k[1] is not None:
                self._rows[k] = CameraStateRow(row.pos, row.quat)
                n += 1
        return n

    # -- reads ------------------------------------------------------------ #
    def get(self, camera: str, env_id: int | None = None):
        """The row for ``(camera, env)`` (shared ignores ``env_id``)."""
        return self._rows.get(self._key(camera, env_id))

    def has(self, camera: str, env_id: int | None = None) -> bool:
        """Whether a row exists for ``(camera, env)`` (see :meth:`get`)."""
        return self._key(camera, env_id) in self._rows

    def rows(self, camera: str) -> dict:
        """``{env_id|None: row}`` of one camera (insertion order)."""
        return {env: row for (cam, env), row in self._rows.items()
                if cam == str(camera)}

    def __len__(self) -> int:
        return len(self._rows)


# --------------------------------------------------------------------------- #
# request-side helpers (duck-typed specs; no base.py import -> no cycle)
# --------------------------------------------------------------------------- #
def state_is_shared(spec) -> bool:
    """True when the camera's request axis is ``state='shared'``."""
    return getattr(spec, "state", None) == "shared"


def read_env_ids(spec, rig):
    """Instance scope of one sensor refresh (the tensor shape never changes).

    shared -> the single meaningful ``target_env``; per_instance -> ``None``
    (= every env id).
    """
    if state_is_shared(spec):
        return [int(getattr(rig, "target_env", 0))]
    return None


def check_single_env(name: str, env_ids) -> None:
    """Single-world gate: ``env_ids`` must be None or ``[0]``."""
    if env_ids is None:
        return
    if [int(i) for i in env_ids] != [0]:
        raise ValueError(
            f"camera '{name}': the backend is single-env; env_ids must be "
            f"None or [0], got {env_ids}")


def make_state_table(name: str, state: str) -> CameraStateTable:
    """A declared state table (``shared``/``per_instance``)."""
    table = CameraStateTable()
    table.declare(name, state)
    return table


def apply_declared_state(spec, rig, num_envs: int) -> None:
    """Bind hook: validate ``env_poses`` then apply the declared rows."""
    spec.validate_env_poses(num_envs)
    apply_declared = getattr(rig, "apply_declared_poses", None)
    if callable(apply_declared):
        apply_declared(getattr(spec, "env_poses", None))
