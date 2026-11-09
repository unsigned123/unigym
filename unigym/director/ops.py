"""DirectorPropOps: the per-backend director mechanics seam (M2).

The ``Director`` is the backend-agnostic op face (command parsing,
validation, bookkeeping, directed marks, logging — director_contract.md).
Every touch that actually moves simulation state used to be an ``if
backend-name branch inside ``Director`` (10 sites, the
largest branch cluster in the repo).  This module is the convergence
point: a :class:`DirectorPropOps` object owns ALL backend mechanics, the
Director dispatches to it, and the weld/kinematic semantic split becomes
a registry trait check
(:attr:`unigym.backends.registry.Trait.DIRECTOR_ATTACH_WELD`)
instead of a string comparison.

director-console: the kinematic primitives (pin/follow/release)
are the UNIVERSAL grasp face (kin is the default attach mode on every
backend; weld is the mujoco opt-in reality mode), plus the robot-root
``push`` op.  The per-tick follower application stays in the Director's
sink hook (``Director.on_step``) via ``write_slave_pose`` + ``halt``.

Implementations: ``backends/mujoco/director_ops.MujocoDirectorOps`` /
``backends/isaaclab2/director_ops.IsaacLabDirectorOps`` (the M2.5 physical
move is done).  Both engines
stay lazy (function-body imports) — ``director/`` remains agnostic at
module level.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataclasses import dataclass

import numpy as np

from ..backends.registry import Trait, backend_info

# kinematic-console: the ``hand`` host alias resolves to the FIRST
# resolvable candidate (G1 finger-link naming, right hand primary -- the
# scripted-pick fixture hand).  Explicit body/link names stay available.
HAND_ALIAS_CANDIDATES = ("right_hand_index_0_link", "left_hand_index_0_link")

# accepted ``prop:``-prefixed host spellings (webui dropdown face)
_PROP_PREFIX = "prop:"


def strip_prop_prefix(host: str) -> str:
    """``prop:bench`` -> ``bench`` (plain names pass through)."""
    host = str(host)
    return host[len(_PROP_PREFIX) :] if host.startswith(_PROP_PREFIX) else host


@dataclass(frozen=True)
class WeldHost:
    """Opaque backend host handle for a weld/pin attachment.

    :meth:`DirectorPropOps.resolve_weld_host` returns one and the neutral
    Director forwards it to :meth:`DirectorPropOps.attach_weld` untouched;
    only the producing backend reads ``payload``.  This keeps engine
    addressing (body/link ids) out of the neutral protocol.
    """

    payload: object


# --------------------------------------------------------------------------- #
# SE(3) pose helpers (wxyz quats; numpy only -- agnostic zone)
# --------------------------------------------------------------------------- #
def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a*b (wxyz)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[1:4] *= -1.0
    return out


def quat_rot(q, v) -> np.ndarray:
    """Rotate vector v by quat q."""
    t = quat_mul(quat_mul(q, [0.0, v[0], v[1], v[2]]), quat_conj(q))
    return t[1:4]


def pose_rel(obj_pos, obj_quat, host_pos, host_quat) -> tuple[np.ndarray, np.ndarray]:
    """host<-obj relative pose ``(rel_pos, rel_quat)``: obj = host * rel."""
    dq = quat_conj(host_quat)
    rel_pos = quat_rot(
        dq, np.asarray(obj_pos, dtype=np.float64) - np.asarray(host_pos, dtype=np.float64)
    )
    return rel_pos, quat_mul(dq, obj_quat)


def pose_compose(host_pos, host_quat, rel_pos, rel_quat) -> tuple[np.ndarray, np.ndarray]:
    """obj = host * rel (the per-tick follow write)."""
    pos = np.asarray(host_pos, dtype=np.float64) + quat_rot(host_quat, rel_pos)
    return pos, quat_mul(host_quat, rel_quat)


@runtime_checkable
class DirectorPropOps(Protocol):
    """Backend mechanics for the ``direct.*`` op family.

    Semantics (design §1.1 #10): the two attach modes are different
    behaviors, not missing features — ``weld`` freezes the object's
    relative pose into an equality constraint (mujoco eq slot);
    ``slave`` re-writes the object pose every tick (kinematic follow,
    isaaclab write_root_pose).  The trait tables declare which backend
    speaks which mode; :meth:`has_weld` is the runtime query.
    """

    backend: str

    def bind(self, gripper_cfg: dict | None) -> object | None:
        """Resolve prop tables + construct the virtual gripper.
        Returns the gripper (or None when undeclared)."""
        ...

    def prop_names(self) -> list[str]:
        """Declared prop names (error-hint face)."""
        ...

    def resolve_obj(self, name: str):
        """Validate + resolve a prop reference (mujoco: the name itself;
        isaaclab: the asset handle)."""
        ...

    def resolve_weld_host(self, host: str) -> WeldHost:
        """Weld host -> opaque backend handle (contract §2 order).

        The returned value is never inspected by the neutral Director; it
        is passed through to :meth:`attach_weld` unchanged.
        """
        ...

    def teleport(self, name, pos, quat, reset_vel, joints) -> dict: ...

    def setvel(self, name, lin, ang) -> dict: ...

    def attach_weld(self, obj: str, host: WeldHost) -> dict:
        """Activate the weld to an opaque host handle; returns the record."""
        ...

    def detach(self, att: dict) -> None:
        """Deactivate one attachment's backend side (weld eq slot)."""
        ...

    def host_pose(self, host: str) -> tuple:
        """``(pos (3,), quat (4,) wxyz)`` of a follow/weld host:
        gripper | ``hand`` alias | ``prop:name`` | prop name | robot body
        link (contract §2)."""
        ...

    def write_slave_pose(self, obj: str, pos, quat) -> None: ...

    def obj_pose(self, obj: str) -> tuple:
        """``(pos (3,), quat (4,) wxyz)`` of a prop ROOT (pin/follow
        capture face; rigid free props only -- articulation roots are
        anchored and fail-fast with the honest hint)."""
        ...

    def halt(self, obj: str) -> None:
        """Zero one prop root's velocity (the ghost discipline:
        a pinned/followed object carries no momentum, so release
        resumes dynamics at rest)."""
        ...

    def push_root(self, lin, ang, mode: str) -> dict:
        """Robot root velocity write: ``mode`` set|add (the push op)."""
        ...

    def write_joints(self, joints: dict[str, float], zero_vel: bool) -> dict:
        """Robot joint-angle write by name (the joints op).

        Pose-teleport semantics (contract §3.11): a one-shot position
        write -- PAUSED it IS the final pose (scene staging); RUNNING the
        position controller pulls back toward the action targets.  No
        follower state is created; ``zero_vel`` additionally zeroes the
        written joints' velocity.  Unknown names fail-fast listing the
        robot joint table (scene face tree.robot; ``Prop_*`` injected
        joints are not addressable)."""
        ...

    def default_attach_mode(self) -> str: ...

    def has_weld(self) -> bool: ...

    def after_gripper_move(self) -> None:
        """Post-move engine kick (mujoco: mj_forward; isaaclab: no-op)."""
        ...


class _OpsBase:
    """Shared helpers for the two ops implementations."""

    backend = ""

    def has_weld(self) -> bool:
        return backend_info(self.backend).has(Trait.DIRECTOR_ATTACH_WELD)

    def after_gripper_move(self) -> None:
        pass


def build_ops(env, director_ops) -> DirectorPropOps:
    """Factory: instantiate the INJECTED per-backend ops face.

    ``director_ops`` is the zero-argument ``BackendInfo.director_ops``
    factory resolved by the run layer.  This helper never
    inspects ``env`` to guess a backend.
    """
    if director_ops is None:
        raise TypeError(
            "build_ops requires an explicit director_ops face injected by "
            "the run layer (no env discovery)"
        )
    return director_ops()(env)
