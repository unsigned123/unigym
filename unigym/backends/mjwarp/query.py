"""MuJoCo Warp implementation of the runtime-query face.

mjWarp keeps one CPU ``MjData`` mirror per world as its read face; the shared
MuJoCo-family query math runs against that mirror, never against the warp
device buffers.  Model facts come from the compiled ``ModelIR`` / ``model_set``
and live state is a mirror read.  Scratch is a private ``mj_copyData`` of the
mirror plus ``mj_forward``: it is a CPU copy, not a device clone, and it never
mutates the live mirror or the ``mujoco_warp`` ``Data``.

The caller must query on the engine thread at a step boundary, after the
backend has synchronized the mirrors from the device.  There is no implicit
sync and no lock; a mid-step query reads stale mirror state.
"""

from __future__ import annotations

from unigym.backends.mjwarp.env import MjwarpEnv
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

__all__ = ["MJWARP_QUERY_CAPABILITIES", "MjwarpQuery"]

_NATIVE = Feature("native", "CPU mirror of the compiled model / live mirror read")
_P1A_DETAIL = "contact Jacobian and wrench-to-joint are added with the P1A-2 plan"
_P1_DETAIL = "actuator queries are added in P1B"
_CONTACT_MIRROR = (
    "CPU contact mirror math; group-based active contacts regrouped from warp's "
    "contact solve, not a warp device read"
)
_MIRROR = "CPU mirror copy (mj_copyData + mj_forward); not a warp device clone"
_CPU_MATH = "CPU MuJoCo math on the mirror copy"
_ENGINE_THREAD = (
    "engine thread only; query at a step boundary after the mirror sync, "
    "no concurrent step/reset"
)
_COPIES = "detached CPU torch copies; no obs publication or record writes"
_SCRATCH = "one exclusive scratch per env; CPU mirror copy + mj_forward"

MJWARP_QUERY_CAPABILITIES = QueryCapabilities(
    api_version=1,
    backend="mjwarp",
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
        copy=Feature("synthetic", _MIRROR),
        live_untouched=Feature("synthetic", _MIRROR),
        set_root_state=Feature("synthetic", _MIRROR),
        set_joint_state=Feature("synthetic", _MIRROR),
        forward=Feature("synthetic", _MIRROR),
        body_pose=Feature("synthetic", _MIRROR),
        site_pose=Feature("synthetic", _MIRROR),
        com_pose=Feature("synthetic", _MIRROR),
        geometry=Feature("synthetic", _MIRROR),
        jacobian=Feature("synthetic", _MIRROR),
    ),
    jacobian=JacobianCaps(
        body=Feature("synthetic", _CPU_MATH),
        site=Feature("synthetic", _CPU_MATH),
        com=Feature(
            "partial",
            "CPU mirror math; body COM has translational+rotational, "
            "subtree COM is translational only",
        ),
        point=Feature("synthetic", _CPU_MATH),
    ),
    geometry=GeometryCaps(
        distance=Feature("synthetic", _CPU_MATH),
        aabb=Feature("synthetic", _CPU_MATH),
    ),
    contacts=ContactCaps(
        manifold=Feature("synthetic", _CONTACT_MIRROR),
        jacobian=Feature("unsupported", _P1A_DETAIL),
        wrench_to_joint=Feature("unsupported", _P1A_DETAIL),
    ),
    actuators=ActuatorCaps(
        facts=Feature("unsupported", _P1_DETAIL),
        state=Feature("unsupported", _P1_DETAIL),
    ),
    notes={
        "engine_thread": _ENGINE_THREAD,
        "returns": _COPIES,
        "scratch": _SCRATCH,
        "mirror": "live state reads the CPU MjData mirror, not warp device buffers",
    },
)


class MjwarpQuery(MjFamilyQuery):
    """The mjWarp runtime-query facade over the CPU ``MjData`` mirrors."""

    backend = "mjwarp"
    capabilities = MJWARP_QUERY_CAPABILITIES

    def __init__(self, env: EnvBackend | MjwarpEnv) -> None:
        if not isinstance(env, MjwarpEnv):
            raise TypeError(f"MjwarpQuery requires a MjwarpEnv, got {type(env).__name__}")
        super().__init__(env)
