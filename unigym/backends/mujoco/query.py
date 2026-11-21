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
