"""Neutral actuation IR -> Isaac Lab actuator config plan.

Only ``standard_pd`` maps to a native Isaac Lab actuator
(:class:`ImplicitActuatorCfg`).  Every other neutral law has no native
equivalent and fails with a capability teaching error -- never a silent
approximation of a different physics model.

The D5 G1 migration supplies the adapter
``actuator_plan(profile, spec) -> build_actuator_plan(resolved_groups)``;
this module defines and unit-tests the plan itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

from unigym.actuators.ir import ResolvedActuationGroup

__all__ = ["build_actuator_plan"]


def _implicit_actuator_cfg(classes: Mapping[str, object] | None) -> Callable[..., object]:
    if classes is not None:
        cls = classes.get("ImplicitActuatorCfg")
        if cls is None:
            raise ValueError(
                "actuator plan classes must provide 'ImplicitActuatorCfg', "
                f"got {sorted(classes)}"
            )
        return cast(Callable[..., object], cls)
    from isaaclab.actuators import ImplicitActuatorCfg

    return ImplicitActuatorCfg


def _standard_pd_kwargs(resolved: ResolvedActuationGroup) -> dict[str, object]:
    group = resolved.group
    if group.law != "standard_pd":
        raise ValueError(
            f"actuation group {group.name!r} law {group.law!r} has no native "
            f"Isaac Lab actuator mapping; the Isaac Lab backends implement "
            f"standard_pd only"
        )
    if group.parts:
        raise ValueError(
            f"actuation group {group.name!r} law 'standard_pd' declares "
            f"parts {list(group.parts)}; parts are only valid for "
            f"computed_torque_pd"
        )
    if group.command:
        raise ValueError(
            f"actuation group {group.name!r} declares a non-empty `command`; "
            f"no current law accepts command parameters"
        )
    unknown = sorted(set(group.gains) - {"kp", "kd"})
    if unknown:
        raise ValueError(
            f"actuation group {group.name!r} law 'standard_pd' declares "
            f"unknown gains {unknown}; allowed gains are ['kd', 'kp']"
        )
    missing = [key for key in ("kp", "kd") if key not in group.gains]
    if missing:
        raise ValueError(
            f"actuation group {group.name!r} law 'standard_pd' is missing "
            f"gains {missing}; declare both kp and kd explicitly"
        )
    kwargs: dict[str, object] = {
        "joint_names_expr": list(resolved.joint_names),
        "stiffness": group.gains["kp"],
        "damping": group.gains["kd"],
    }
    if group.effort_limit is not None:
        kwargs["effort_limit_sim"] = group.effort_limit
    if group.velocity_limit is not None:
        kwargs["velocity_limit_sim"] = group.velocity_limit
    return kwargs


def build_actuator_plan(
    groups: Sequence[ResolvedActuationGroup],
    *,
    classes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Translate resolved actuation groups into Isaac Lab actuator configs.

    Args:
        groups: the resolved per-articulation actuation plan, in declaration
            order.
        classes: optional ``{"ImplicitActuatorCfg": <class>}`` injection table
            for unit tests.  ``None`` lazily imports the Isaac Lab class.

    Returns:
        ``{group name: ImplicitActuatorCfg}``.
    """
    cfg_cls = _implicit_actuator_cfg(classes)
    plan: dict[str, object] = {}
    for resolved in groups:
        group = resolved.group
        if group.name in plan:
            raise ValueError(f"duplicate actuation group name {group.name!r} in the plan")
        plan[group.name] = cfg_cls(**_standard_pd_kwargs(resolved))
    return plan
