"""Pluggable Actuator seam: the ctrl-write protocol + registry.

One first-class package (sibling of adapters/benchmarks, owner ruling) --
the "absolute position target -> per-substep torque/servo ctrl write" law
retires out of backend hardcode into pluggable actuators
(docs/designs/v063-gr00t-adapter.md D2/D9).  The action-face semantics are
untouched (NamedAction = absolute position targets; routing stays with the
backend); an actuator owns ONLY the ctrl slots assigned at bind time, and
out-of-bounds/duplicate coverage fails fast.

Protocol shape (the three built-in laws subclass this base; third-party
laws subclass + ``@register``):
* ``bind(env, joint_group)`` -- resolve joints + actuator slots from the
  group's regex/joint_names (fail-fast: unknown joint / no actuator /
  slot out of bounds / duplicate coverage is checked by the backend seam);
* ``set_targets(qpos_targets)`` -- the backend feeds the joint-ordered
  target vector once per control step (held constant across the substeps,
  the audited GR1 no-interpolation semantics);
* ``step(model, data)`` -- the backend calls once per sim substep; the
  law writes only its own ctrl slots;
* ``snapshot_internal()/restore_internal()`` -- stateful-law rewind face;
  the built-ins are stateless and the defaults are no-ops (the snapshot's
  mjSTATE ctrl bits already carry the whole write face).

Import face stays numpy-only (mujoco imports lazily inside ``bind``), so
the registry and config validation unit-test on hosts without mujoco.
"""
from __future__ import annotations

import re
from typing import ClassVar

import numpy as np

__all__ = [
    "ACTUATOR_REGISTRY",
    "ActuatorBase",
    "build_actuator",
    "register",
    "resolve_gain_vector",
]

#: type name -> actuator class (populated by ``@register``; the built-in
#: modules self-register on import and ``unigym/actuators/__init__.py``
#: imports them all).
ACTUATOR_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: register the actuator law under *name* (fail on dup)."""

    def deco(cls):
        if name in ACTUATOR_REGISTRY:
            raise ValueError(
                f"actuator type {name!r} already registered "
                f"({ACTUATOR_REGISTRY[name].__name__})")
        cls.type_name = name
        ACTUATOR_REGISTRY[name] = cls
        return cls

    return deco


def build_actuator(cfg: dict) -> ActuatorBase:
    """Instantiate the law named by ``cfg["type"]`` (unbound).

    Fail-fast with the registered-type list when the name is unknown (the
    teaching hint names the config face so the yaml author self-corrects).
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"actuator group must be a mapping {{type, regex, ...}}, got "
            f"{type(cfg).__name__} | hint: env.params.actuators entries "
            f"look like {{body: {{type: computed_torque_pd, regex: "
            f"robot0_.*, kp: 1000, kd: 200}}}}")
    atype = cfg.get("type")
    if atype not in ACTUATOR_REGISTRY:
        raise ValueError(
            f"unknown actuator type {atype!r}; registered types: "
            f"{sorted(ACTUATOR_REGISTRY)} | hint: set "
            f"env.params.actuators.<group>.type to one of the registered "
            f"names (standard_pd is the default law; omitting the whole "
            f"'actuators' key keeps the legacy backend path)")
    return ACTUATOR_REGISTRY[atype]()


def _gain_override_hit(name: str, table: dict):
    """First hit for *name* in a gain dict.

    Exact key first, then regex fullmatch in dict order (the legacy
    env.params kp/kd semantics).
    """
    if name in table:
        return float(table[name])
    for expr, val in table.items():
        if re.fullmatch(str(expr), name):
            return float(val)
    return None


def resolve_gain_vector(spec, which: str, joint_names: list) -> np.ndarray:
    """Resolve scalar | joint-name/regex dict | full vector -> per-joint float64.

    A joint with no dict entry fails fast (no silent zeros in a ctrl law).
    """
    n = len(joint_names)
    if isinstance(spec, bool):
        raise ValueError(
            f"actuator gain {which!r} must be numeric, got a bool")
    if isinstance(spec, (int, float)):
        return np.full(n, float(spec), dtype=np.float64)
    if isinstance(spec, dict):
        out = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(joint_names):
            val = _gain_override_hit(name, spec)
            if val is None:
                raise ValueError(
                    f"actuator gain {which!r} dict has no entry for joint "
                    f"{name!r} | hint: dict keys are joint names or "
                    f"regexes fullmatching them")
            out[i] = val
        return out
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n:
        return arr.copy()
    raise ValueError(
        f"actuator gain {which!r} must be a scalar, a joint-name/regex "
        f"dict, or a length-{n} vector, got {type(spec).__name__}")


class ActuatorBase:
    """Base class + protocol for one ctrl law over a joint group.

    Subclasses declare ``config_keys`` (the group-spec fields they accept;
    unknown keys fail fast with the legal set) and implement ``step`` plus
    the ``_on_bind`` hook.  The base owns joint/slot resolution, target
    feeding and the stateless rewind defaults.
    """

    #: set by ``@register``
    type_name: ClassVar[str] = ""
    #: group-spec keys this law accepts ("type" + the addressing keys are
    #: always legal; subclasses extend)
    config_keys: ClassVar[frozenset] = frozenset(
        {"type", "regex", "joint_names"})

    def __init__(self) -> None:
        self._joint_names: list = []
        self._qadr: np.ndarray | None = None
        self._dadr: np.ndarray | None = None
        self._slots: np.ndarray | None = None
        self._ctrlrange: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._bound = False

    # ---------------------------------------------------------------- #
    # bind-time resolution (fail-fast)
    # ---------------------------------------------------------------- #
    def bind(self, env, joint_group: dict) -> None:
        """Resolve the group's joints and actuator slots against *env*.

        ``joint_group`` is this group's config dict.  Resolution order:
        config-key gate -> joint names (explicit list or regex fullmatch
        over the env's robot hinges, model order) -> per-joint qpos/dof
        addresses + actuator slot -> bounds checks.  Any miss raises
        ``ValueError`` with a teaching payload.
        """
        unknown = set(joint_group) - set(self.config_keys)
        if unknown:
            raise ValueError(
                f"{self.type_name} actuator group got unsupported keys "
                f"{sorted(unknown)}; legal keys: {sorted(self.config_keys)}")
        if "regex" not in joint_group and "joint_names" not in joint_group:
            raise ValueError(
                f"{self.type_name} actuator group needs 'regex' (or an "
                f"explicit 'joint_names' list), got {sorted(joint_group)}")
        names = self._resolve_joint_names(env, joint_group)
        import mujoco

        m = env._model
        act_of_joint = getattr(env, "_act_of_joint", None)
        qadr, dadr, slots = [], [], []
        for name in names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(
                    f"{self.type_name} actuator: joint {name!r} not in the "
                    f"MJ model | hint: group regexes fullmatch against the "
                    f"robot hinge names {self._universe(env)[:4]} ...")
            if int(m.jnt_type[jid]) not in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(
                    f"{self.type_name} actuator: joint {name!r} is not a "
                    f"1-dof hinge/slide (type {int(m.jnt_type[jid])}); the "
                    f"ctrl laws address per-joint torque/position targets")
            slot = None if act_of_joint is None else act_of_joint.get(name)
            if slot is None:
                raise ValueError(
                    f"{self.type_name} actuator: no actuator drives joint "
                    f"{name!r} (the law would have nothing to write)")
            if not 0 <= int(slot) < int(m.nu):
                raise ValueError(
                    f"{self.type_name} actuator: joint {name!r} maps to "
                    f"actuator slot {slot} outside the model's ctrl vector "
                    f"(nu={m.nu})")
            qadr.append(int(m.jnt_qposadr[jid]))
            dadr.append(int(m.jnt_dofadr[jid]))
            slots.append(int(slot))
        self._joint_names = list(names)
        self._qadr = np.asarray(qadr, dtype=np.int64)
        self._dadr = np.asarray(dadr, dtype=np.int64)
        self._slots = np.asarray(slots, dtype=np.int64)
        self._ctrlrange = m.actuator_ctrlrange[self._slots].copy()
        self._on_bind(env, joint_group)
        self._bound = True

    def _on_bind(self, env, joint_group: dict) -> None:
        """Subclass hook: gains/parts resolution after slot resolution."""

    def _universe(self, env) -> list:
        """The env's ROBOT hinge universe.

        Prop freejoints excluded -- they are re-placed by the props
        backend, never actuator-owned.
        """
        names = getattr(env, "asset_joint_names", None)
        if names is None:
            arts = getattr(env, "articulations", None) or {}
            if arts:
                names = [j for art in arts.values() for j in art.joint_names]
        if names is None:
            raise ValueError(
                "actuator bind needs the env's robot joint universe "
                "(controller.asset_joint_names); the object passed to bind "
                "is not a bound mujoco backend env")
        names = list(names)
        # A backend binding may hand the law compiled joint names (the
        # controller's neutral names stay authoritative on the public face).
        names.extend(getattr(env, "_compiled_joint_names", ()) or ())
        return [n for n in dict.fromkeys(names)
                if not str(n).startswith("Prop_")]

    def _resolve_joint_names(self, env, joint_group: dict) -> list:
        universe = self._universe(env)
        if "joint_names" in joint_group:
            names = list(joint_group["joint_names"])
            missing = [n for n in names if n not in set(universe)]
            if missing:
                raise ValueError(
                    f"{self.type_name} actuator: joint_names {missing} not "
                    f"in the robot joint universe")
            return names
        expr = str(joint_group["regex"])
        names = [n for n in universe if re.fullmatch(expr, n)]
        if not names:
            raise ValueError(
                f"{self.type_name} actuator: regex {expr!r} matches none "
                f"of the robot joints | hint: regexes fullmatch against "
                f"the model hinge names, e.g. {universe[:4]} ...")
        return names

    # ---------------------------------------------------------------- #
    # runtime faces
    # ---------------------------------------------------------------- #
    @property
    def joint_names(self) -> list:
        """The bound joint names (model-order subset, resolution order)."""
        return list(self._joint_names)

    @property
    def bound_slots(self) -> np.ndarray:
        """The ctrl slots this law owns (int64 array; set at bind)."""
        if self._slots is None:
            raise RuntimeError(
                f"{self.type_name} actuator queried for slots before "
                f"bind(); call bind(env, joint_group) first")
        return self._slots

    def set_targets(self, qpos_targets) -> None:
        """Feed this control step's absolute position targets (joint order).

        Called once per control step by the backend, BEFORE the decimation
        loop; the law holds them constant across the substeps (the audited
        GR1 stack semantics: goal set once, ctrl recomputed per substep).
        """
        t = np.asarray(qpos_targets, dtype=np.float64).reshape(-1)
        if not self._bound or t.shape[0] != len(self._joint_names):
            raise ValueError(
                f"{self.type_name} actuator: target vector shape {t.shape} "
                f"does not match the {len(self._joint_names)} bound joints "
                f"| hint: bind(env, joint_group) first, then feed one "
                f"target per bound joint in joint_names order")
        self._targets = t

    def step(self, model, data) -> None:
        """One sim substep: compute + write this law's ctrl slots."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement step(model, data)")

    def snapshot_internal(self):
        """Stateful-law rewind payload.

        ``None`` = stateless: the mjSTATE ctrl bits already carry the
        full write face.
        """
        return None

    def restore_internal(self, state) -> None:
        """Rewind counterpart of :meth:`snapshot_internal`."""
        if state is not None:
            raise ValueError(
                f"{self.type_name} actuator returned no internal state at "
                f"snapshot time but restore got {type(state).__name__}")

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError(
                f"{self.type_name} actuator stepped before bind(); call "
                f"bind(env, joint_group) first")
