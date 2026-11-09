"""Anchored articulation prop handler (usd_articulation + latch).

The articulated-prop kind done through the existing named surfaces:

* **asset**  -- a URDF or MJCF fragment whose joints are defined inside the
  asset. The root body is FIXED to the world at the sampled placement
  (anchored semantics; a fragment carrying a freejoint fails fast at the
  backend build/bind time). The former free-floating door handler
  (props/door.py) is subsumed by this one -- door.urdf is reused as-is.
  The ``type: usd_articulation`` kind: a native USD articulation asset
  (embedded revolute joints) spawned via ``UsdFileCfg`` with
  ``fix_root_link`` anchoring -- the HumanoidArena open_door door route.
  isaaclab-only (the mujoco backend
  has no USD articulation loader and fails fast at spec-build time).
* **action surface** -- every joint declared ``mode: actuated``
  automatically becomes a NamedAction entry named after the prop
  (``{prop_name: NamedAction(values, joint_names)}``, alongside ``g1`` /
  ``hand``); no extra action-section wiring. ``mode: passive`` joints get
  damping injection only and never join the action surface (a
  passive joint MAY still declare ``stiffness`` / ``effort_limit`` so
  USD-authored drive semantics are reproducible without policy driving --
  e.g. the HA door handle: stiffness 5.0 behind an effort cap of 0.001).
* **state surface** -- the anchor root pose flows into the
  ``object/<name>/{pos,quat}`` record keys like any prop, plus
  ``object/<name>/joints/<jname>`` (N, 1) per declared joint, consumed
  directly by FormulaTask terms (``final`` / ``range_check_fraction``).
* **latch** (optional ``spec.latch``) -- runtime drive switching
  on ONE declared joint, gated by ANOTHER joint's angle crossing a
  threshold: locked drive on reset, and once the sense joint crosses the
  threshold the lock joint's drive is restored to its declared (default)
  values. Mirrors the HumanoidArena door latch (handle turned past -20deg
  releases the leaf's 1e5-stiffness lock; move_open_door_*_env_cfg.py:
  401-422,540-656,1481-1485). Angles are ABSOLUTE joint positions in
  degrees -- our on_reset zeros the declared joints, so absolute equals
  relative-to-reset as upstream computes it.

fail-fast (declaration level here, asset level at backend bind time):
unknown mode / ``actuated`` without ``stiffness`` / ``passive`` without
``damping`` / an asset joint left undeclared / a declared joint missing
from the asset / a latch naming a joint outside the declared table.
"""
from __future__ import annotations

import math

import numpy as np

from .base import (
    PropSpec,
)

JOINT_MODES = ("passive", "actuated")


def validate_joints(spec: PropSpec) -> dict[str, dict]:
    """Normalize + validate ``spec.joints``; returns the declared table.

    Each entry becomes ``{mode, damping, stiffness, effort_limit, range,
    init}`` with float coercion.  Declaration-level fail-fast: mode must
    be ``passive`` or ``actuated``; ``actuated`` requires ``stiffness``;
    ``range`` (when given) is ``[lo,
    hi]`` with lo <= hi.  BOTH modes accept optional ``stiffness`` (a
    passive joint may keep its USD-authored drive stiffness) and optional
    ``effort_limit`` (fed to ImplicitActuatorCfg.effort_limit_sim; a capped
    passive drive, like the HA door handle's 0.001 N*m, is expressible
    without joining the action surface).  ``init`` (T-B1, optional,
    default 0.0) is the declared RESET joint position -- the official
    fixture semantics (robocasa ``set_door_state`` writes the open
    fraction straight into qpos) need a declared non-zero rest state; it
    must lie inside a declared ``range`` (fail-fast otherwise; the
    backend build re-checks against the ASSET range when the declaration
    omits one).  Both backends write ``init`` on every reset (was: an
    unconditional zero).

    A ``passive`` joint's ``damping`` is OPTIONAL (the composite pilot).
    The requirement existed for the
    damping-INJECTION face (a passive joint with no asset-authored
    dynamics needs declared damping to be movable at all); the composite
    chains carry their dynamics IN the asset (the official
    ``springdamper`` mass-spring parametrization, whose compiled damping
    is mass-derived per joint -- a declaration could only override it
    with wrong numbers).  A silent damping keeps the asset's own
    dynamics everywhere (no injection, no override); the real
    forgotten-declaration protection stays the joint-set coverage gates
    (the step-4 ruling).
    """
    out: dict[str, dict] = {}
    for jname, params in (spec.joints or {}).items():
        p = dict(params or {})
        mode = str(p.get("mode", ""))
        if mode not in JOINT_MODES:
            raise ValueError(
                f"articulation prop {spec.name!r}: joint {jname!r} mode "
                f"must be one of {JOINT_MODES}, got {mode!r}")
        stiffness = p.get("stiffness")
        damping = p.get("damping")
        if mode == "actuated" and stiffness is None:
            raise ValueError(
                f"articulation prop {spec.name!r}: actuated joint "
                f"{jname!r} requires stiffness")
        # a passive damping is OPTIONAL: an
        # asset-driven passive joint (the composite chains' springdamper)
        # declares none -- a required declaration could only override the
        # asset's own mass-derived dynamics with wrong numbers
        rng = p.get("range")
        if rng is not None:
            rng = [float(rng[0]), float(rng[1])]
            if rng[0] > rng[1]:
                raise ValueError(
                    f"articulation prop {spec.name!r}: joint {jname!r} "
                    f"range {rng} has lo > hi")
        init = float(p.get("init", 0.0))
        if not np.isfinite(init):
            raise ValueError(
                f"articulation prop {spec.name!r}: joint {jname!r} init "
                f"must be finite, got {init!r}")
        if rng is not None and not (rng[0] <= init <= rng[1]):
            raise ValueError(
                f"articulation prop {spec.name!r}: joint {jname!r} init "
                f"{init!r} outside the declared range {rng}")
        out[jname] = {
            "mode": mode,
            "stiffness": float(stiffness) if stiffness is not None else 0.0,
            "damping": float(damping) if damping is not None else 0.0,
            "effort_limit": (None if p.get("effort_limit") is None
                             else float(p["effort_limit"])),
            "range": rng,
            "init": init,
        }
    if not out and spec.type == "usd_articulation":
        # the usd_articulation kind keeps its >= 1 net here (mirroring
        # the resolve-time gate in props.base).  The anchored kinds
        # carry no gate anymore: the jointless
        # robocasa Counter is a legitimate anchored fragment with zero
        # joints, and the REAL protection is joint-set coverage, which
        # lives elsewhere and still applies -- generated payloads
        # enforce the exact declared==IR-joint-set equality at resolve
        # time (props.base._resolve_generated_spec, an empty set
        # included), and fragments enforce declared==asset-joint
        # coverage at backend bind time (a forgotten declaration on a
        # jointed fragment surfaces there, one phase later, same
        # fail-fast class; the free kind was exempt from the start --
        # the injected freejoint is the whole dof face).
        raise ValueError(
            f"articulation prop {spec.name!r}: joints must declare at "
            f"least one joint")
    return out


def validate_latch(spec: PropSpec, joints: dict[str, dict]) -> dict | None:
    """Normalize + validate ``spec.latch``; ``None`` when not declared.

    Declared shape::

        latch:
          lock_joint: <declared joint>     # gets its drive switched
          sense_joint: <declared joint>    # threshold-gating angle
          unlock_threshold_deg: -20.0      # negative: unlock when <= it
          locked: {stiffness: 1e5, damping: 5e4, friction: 1e4}
          # optional; defaults to the lock joint's declared passive values
          unlocked: {stiffness: 0.0, damping: 0.0, friction: 0.0}

    fail-fast: lock_joint/sense_joint must be declared in ``spec.joints``;
    ``locked`` requires at least one of stiffness/damping/friction; the
    threshold must be a finite float.  The comparison direction mirrors the
    upstream latch (threshold >= 0 -> unlock when sense >= threshold,
    negative -> sense <= threshold; move_open_door_*_env_cfg.py:1481-1485).
    """
    cfg = getattr(spec, "latch", None) or None
    if not cfg:
        return None
    lock_joint = cfg.get("lock_joint")
    sense_joint = cfg.get("sense_joint")
    if lock_joint not in joints:
        raise ValueError(
            f"articulation prop {spec.name!r}: latch.lock_joint "
            f"{lock_joint!r} is not a declared joint")
    if sense_joint not in joints:
        raise ValueError(
            f"articulation prop {spec.name!r}: latch.sense_joint "
            f"{sense_joint!r} is not a declared joint")
    locked = dict(cfg.get("locked") or {})
    if not any(k in locked for k in ("stiffness", "damping", "friction")):
        raise ValueError(
            f"articulation prop {spec.name!r}: latch.locked must declare "
            f"at least one of stiffness/damping/friction")
    threshold = float(cfg.get("unlock_threshold_deg", -20.0))
    if not math.isfinite(threshold):
        raise ValueError(
            f"articulation prop {spec.name!r}: latch.unlock_threshold_deg "
            f"must be finite, got {cfg.get('unlock_threshold_deg')!r}")
    default_joint = joints[lock_joint]
    unlocked_in = dict(cfg.get("unlocked") or {})
    unlocked = {
        "stiffness": float(unlocked_in.get("stiffness",
                                           default_joint["stiffness"])),
        "damping": float(unlocked_in.get("damping",
                                         default_joint["damping"])),
        "friction": float(unlocked_in.get("friction", 0.0)),
    }
    return {
        "lock_joint": lock_joint,
        "sense_joint": sense_joint,
        "unlock_threshold_deg": threshold,
        "locked": {
            "stiffness": float(locked.get("stiffness", 0.0)),
            "damping": float(locked.get("damping", 0.0)),
            "friction": float(locked.get("friction", 0.0)),
        },
        "unlocked": unlocked,
    }


def __getattr__(name: str):
    """Lazy compat re-export of the isaaclab articulation handlers (G4)."""
    if name in ("PropArticulation", "UsdArticulationProp",
                "FreeArticulationProp"):
        from .base import _compat_class

        return _compat_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
