"""MuJoCo implementation of the runtime-query face.

The shared MuJoCo-family math lives in
``unigym.backends.shared.mjfamily.query``; this module binds it to
``MujocoEnv`` and the MuJoCo capability declaration.  The query runs on the
engine thread at a step boundary; it is not thread-safe with ``step``/``reset``
and the backend does not add a lock.  Every result is a detached CPU torch
copy.  Scratch reads use a per-env ``MjData`` copy (``mj_copyData`` +
``mj_forward``) and never mutate the live ``MjData``.
"""

from __future__ import annotations

from unigym.backends.mujoco.env import MujocoEnv
from unigym.backends.shared.mjfamily.query import MjFamilyQuery
from unigym.env.backends.base import EnvBackend
from unigym.query.capabilities import (
    ActuatorCaps,
    ContactCaps,
    Feature,
    GeometryCaps,
    JacobianCaps,
    ModelCaps,
    QueryCapabilities,
    ScratchCaps,
    StateCaps,
)

__all__ = ["MUJOCO_QUERY_CAPABILITIES", "MujocoQuery"]

_NATIVE = Feature("native")
_CONTACT_JACOBIAN = (
    "contact constraint rows from dense data.efc_J at data.contact.efc_address; "
    "effective row count follows model.opt.cone; columns follow the "
    "articulation DOF layout; sparse efc_J is not mapped"
)
_CONTACT_WRENCH = (
    "rows.T @ data.efc_force for the same contact rows, in the articulation "
    "DOF layout; MuJoCo efc force sign/frame; dense efc_J only"
)
_ACTUATOR_FACTS = (
    "compiled ModelIR actuator row: compiled slot, target/kind, ctrlrange, "
    "gear, gainprm, biasprm; slot_kind='mujoco_actuator'"
)
_ACTUATOR_STATE = (
    "live data.ctrl / data.actuator_force / data.actuator_length / "
    "data.actuator_velocity at the compiled actuator slot; detached CPU copies"
)
_CONTACT_MANIFOLD = (
    "group-based active contacts from data.contact; force acts on the geom2 body "
    "in the MuJoCo contact frame, rotated to world"
)
_ROOT_STATE = (
    "free-base articulation qpos/qvel at the compiled free-joint addresses; "
    "linear qvel is world, angular qvel is converted from body-local with "
    "R(quat); body-frame origin pose and world-frame twist"
)
_COM_STATE = (
    "body COM from xipos plus mj_jacBodyCom @ qvel; subtree COM from "
    "subtree_com plus mj_jacSubtreeCom @ qvel; world linear velocity only, "
    "no COM angular velocity"
)
_OBJECT_STATE = (
    "rigid free prop qpos/qvel at the resolved free-joint addresses; linear "
    "qvel is world, angular qvel is converted from body-local with R(quat); "
    "body-frame origin pose and world-frame twist; no raw handles"
)
_OBJECT_COM_STATE = (
    "single rigid free prop body COM from xipos plus mj_jacBodyCom @ qvel; "
    "world linear velocity only, no COM angular velocity; no raw handles"
)
_OBJECT_JACOBIAN = (
    "six free-joint columns of mj_jacBody (reference='body_origin') or "
    "mj_jacBodyCom (reference='com') at the resolved prop body; world rows, "
    "angular columns in the free-joint body-local qvel basis; no subtree"
)
_ENGINE_THREAD = "engine thread only; no concurrent step/reset"
_COPIES = "detached CPU torch copies; no obs publication or record writes"
_SCRATCH = "one exclusive scratch per env; mj_copyData + mj_forward"

MUJOCO_QUERY_CAPABILITIES = QueryCapabilities(
    api_version=1,
    backend="mujoco",
    model=ModelCaps(
        ir=_NATIVE,
        body=_NATIVE,
        geom=_NATIVE,
        site=_NATIVE,
        joint=_NATIVE,
        actuator=_NATIVE,
    ),
    state=StateCaps(
        body_pose=_NATIVE,
        site_pose=_NATIVE,
        com_pose=_NATIVE,
        root_pose=_NATIVE,
        joint_position=_NATIVE,
        joint_velocity=_NATIVE,
        root_state=Feature("native", _ROOT_STATE),
        com_state=Feature("native", _COM_STATE),
        object_state=Feature("native", _OBJECT_STATE),
        object_com_state=Feature("native", _OBJECT_COM_STATE),
    ),
    scratch=ScratchCaps(
        copy=_NATIVE,
        live_untouched=_NATIVE,
        set_root_state=_NATIVE,
        set_joint_state=_NATIVE,
        forward=_NATIVE,
        body_pose=_NATIVE,
        site_pose=_NATIVE,
        com_pose=_NATIVE,
        geometry=_NATIVE,
        jacobian=_NATIVE,
    ),
    jacobian=JacobianCaps(
        body=_NATIVE,
        site=_NATIVE,
        com=Feature(
            "partial",
            "body COM has translational+rotational; subtree COM is translational only",
        ),
        point=_NATIVE,
        object=Feature("native", _OBJECT_JACOBIAN),
    ),
    geometry=GeometryCaps(distance=_NATIVE, aabb=_NATIVE),
    contacts=ContactCaps(
        manifold=Feature("native", _CONTACT_MANIFOLD),
        jacobian=Feature("native", _CONTACT_JACOBIAN),
        wrench_to_joint=Feature("native", _CONTACT_WRENCH),
    ),
    actuators=ActuatorCaps(
        facts=Feature("native", _ACTUATOR_FACTS),
        state=Feature("native", _ACTUATOR_STATE),
    ),
    notes={
        "engine_thread": _ENGINE_THREAD,
        "returns": _COPIES,
        "scratch": _SCRATCH,
    },
)


class MujocoQuery(MjFamilyQuery):
    """The MuJoCo runtime-query facade over the live CPU ``MjData``."""

    backend = "mujoco"
    capabilities = MUJOCO_QUERY_CAPABILITIES

    def __init__(self, env: EnvBackend | MujocoEnv) -> None:
        if not isinstance(env, MujocoEnv):
            raise TypeError(f"MujocoQuery requires a MujocoEnv, got {type(env).__name__}")
        super().__init__(env)
