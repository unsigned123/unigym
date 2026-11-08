"""Pluggable actuator factory: yaml ``actuator`` field -> Isaac Lab actuator config.

Supported types:
  * ``implicit_pd``:  ImplicitActuatorCfg(stiffness, damping) -- ideal PD
  * ``dc_motor``:     DCMotorCfg (saturation_effort torque clipping +
                       effort/velocity limits; default rating 23.7 N·m /
                       30 rad/s, overridable via ``params``)
  * ``actuator_net``: ActuatorNetMLPCfg (learned actuator network, inputs a
                       pos_err/vel history, outputs torque).  Optional
                       ``class`` field pointing at a custom ActuatorNetMLPCfg
                       subclass
  * ``custom``:       yaml gives ``class: "path.to:Cls"``, reflection-loaded,
                       params passed verbatim to the constructor

The factory is written in dependency-injection form (the ``classes`` argument
admits fake implementations); the core logic is unit-testable locally without
isaaclab.
"""
from __future__ import annotations

import importlib


def _load_class(path: str):
    """Load ``"module.path:ClassName"`` via reflection."""
    if ":" not in path:
        raise ValueError(f"actuator custom class must be 'module:Class', got {path!r}")
    module_name, cls_name = path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"cannot import actuator class module '{module_name}': {e}") from e
    try:
        cls = getattr(module, cls_name)
    except AttributeError as e:
        raise ValueError(f"module '{module_name}' has no class '{cls_name}'") from e
    return cls


def build_actuator(cfg: dict, classes: dict | None = None):
    """Build an Isaac Lab actuator config from a yaml ``actuator`` section.

    Args:
        cfg: ``{"type": ..., "params": {...}}`` (``custom`` also carries a
             ``"class"`` field).
        classes: optional {type: cfg-class} injection table (for local unit
             tests); defaults to the isaaclab classes.
    """
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if "type" not in cfg:
        raise ValueError(f"actuator section needs a 'type', got {cfg!r}")
    atype = cfg["type"]
    params = dict(cfg.get("params", {}))

    if classes is None and atype in ("implicit_pd", "dc_motor"):
        # the isaaclab dependency is imported only when a builtin type needs it
        # (the custom branch stays isaaclab-free)
        from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
        classes = {"implicit_pd": ImplicitActuatorCfg, "dc_motor": DCMotorCfg}

    if atype == "implicit_pd":
        cls = classes["implicit_pd"]
        kwargs = {"joint_names_expr": params.pop("joint_names", [".*"])}
        for k in ("stiffness", "damping", "effort_limit", "velocity_limit",
                  "armature", "friction"):
            if k in params:
                kwargs[k] = params.pop(k)
        _no_leftover(params, atype)
        return cls(**kwargs)

    if atype == "dc_motor":
        cls = classes["dc_motor"]
        kwargs = {"joint_names_expr": params.pop("joint_names", [".*"])}
        for k in ("stiffness", "damping", "effort_limit", "velocity_limit",
                  "saturation_effort", "armature", "friction"):
            if k in params:
                kwargs[k] = params.pop(k)
        # Default motor ratings: peak torque / velocity limits
        kwargs.setdefault("saturation_effort", 23.7)
        kwargs.setdefault("effort_limit", 23.7)
        kwargs.setdefault("velocity_limit", 30.0)
        _no_leftover(params, atype)
        return cls(**kwargs)

    if atype == "actuator_net":
        # Optional class override (e.g. a custom ActuatorNetMLPCfg variant with
        # motor_strength / motor_offset DR buffers + flat effort clip).
        # Priority: explicit 'class' > injected classes > isaaclab default.
        if "class" in cfg:
            cls = _load_class(cfg["class"])
        elif classes is not None and "actuator_net" in classes:
            cls = classes["actuator_net"]
        else:
            from isaaclab.actuators import ActuatorNetMLPCfg
            cls = ActuatorNetMLPCfg
        kwargs = {"joint_names_expr": params.pop("joint_names_expr",
                      params.pop("joint_names", [".*"]))}
        for k in ("network_file", "pos_scale", "vel_scale", "torque_scale",
                  "input_order", "input_idx", "effort_limit",
                  "saturation_effort", "velocity_limit"):
            if k in params:
                kwargs[k] = params.pop(k)
        _no_leftover(params, atype)
        return cls(**kwargs)

    if atype == "custom":
        if "class" not in cfg:
            raise ValueError("actuator type 'custom' requires a 'class: module:Cls' field")
        cls = _load_class(cfg["class"])
        return cls(**params)

    raise ValueError(f"unknown actuator type '{atype}' "
                     "(implicit_pd | dc_motor | actuator_net | custom)")


def _no_leftover(params: dict, atype: str) -> None:
    if params:
        raise ValueError(f"actuator type '{atype}' got unsupported params: {sorted(params)}")
