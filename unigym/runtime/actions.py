"""NamedAction: the name-addressed action contract (feat/named-actions).

The action boundary changes from an ORDERED TENSOR to explicit NAME ADDRESSING:
an adapter returns ``dict[str, NamedAction]`` — one named entry per action
face (e.g. ``{"g1": <29 body joints>, "hand": <14 finger joints>}``), each
carrying its values plus the joint name for every column.  Consumers
(backends) resolve the names to simulator addresses ONCE per name-tuple
(cached) and scatter the values; users never touch simulator joint order
(same philosophy as the name-based obs/contact/foot layers — actions are the
last piece).

Backends:
  * isaaclab — each action term is built with ``preserve_order=True`` and an
    EXPLICIT joint-name list, so term input dim ``i`` == ``joint_names[i]``;
    :class:`IsaaclabJointResolver` maps an entry's names to positions in the
    term's flat input slice.
  * mujoco — :class:`MujocoJointResolver` maps names to actuator/qpos/dof
    addresses (reusing the same name tables the PD path already uses,
    extended to the finger actuators).

This module must stay torch-only (importable by adapters/base.py without
isaaclab/mujoco; the mujoco resolver imports mujoco lazily).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

#: legal NamedAction target kinds (must equal the owning ControlRuntime.target)
LEGAL_ACTION_KINDS = ("position", "velocity", "torque", "effort", "current")


# --------------------------------------------------------------------------- #
# the contract type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NamedAction:
    """One named action face: values, joint names and an explicit kind.

    Attributes:
        values: ``(N, dim)`` float tensor — targets in the units of *kind*
            (rad for position, rad/s for velocity, N·m for torque/effort,
            A for current), one column per entry in *joint_names*.
        joint_names: joint names, ``len(joint_names) == values.shape[1]``.
            The list IS the contract: consumers resolve by name, never by
            position.
        kind: the explicit target kind
            (``position | velocity | torque | effort | current``).  The
            engine validates it against the configured control group's
            ``target`` every tick.

    Construction validates the shape/name/kind agreement (fail-fast) and
    rejects duplicate names (a duplicate would make the routing ambiguous).
    """

    values: torch.Tensor
    joint_names: list
    kind: str

    def __post_init__(self):
        if self.kind not in LEGAL_ACTION_KINDS:
            raise ValueError(
                f"NamedAction.kind must be one of {list(LEGAL_ACTION_KINDS)}, "
                f"got {self.kind!r}")
        if not isinstance(self.values, torch.Tensor):
            raise ValueError(
                f"NamedAction.values must be a torch tensor, got "
                f"{type(self.values).__name__}")
        if self.values.ndim != 2:
            raise ValueError(
                f"NamedAction.values must be (N, dim), got "
                f"{tuple(self.values.shape)}")
        names = list(self.joint_names)
        if len(names) != self.values.shape[1]:
            raise ValueError(
                f"NamedAction: {self.values.shape[1]} value columns vs "
                f"{len(names)} joint_names (must match 1:1)")
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"NamedAction: duplicate joint names {dupes}")

    @property
    def num_envs(self) -> int:
        """Batch dimension N of the values tensor."""
        return int(self.values.shape[0])

    @property
    def dim(self) -> int:
        """Per-env action dimension (== len(joint_names))."""
        return int(self.values.shape[1])

    @classmethod
    def zeros(cls, num_envs: int, joint_names, kind: str, device="cpu",
              dtype=torch.float32) -> NamedAction:
        """All-zero action over *joint_names* with an explicit *kind*."""
        names = list(joint_names)
        return cls(torch.zeros(int(num_envs), len(names), dtype=dtype,
                               device=device), names, kind)


# --------------------------------------------------------------------------- #
# the resolver protocol (backend-traits M2: the typed seam both backends
# implement; NamedAction consumers dispatch here, never on engine names)
# --------------------------------------------------------------------------- #
@runtime_checkable
class JointResolver(Protocol):
    """Name-addressed action entry -> backend addressing primitive.

    isaaclab: :class:`IsaaclabJointResolver` (gather indices into one
    preserve_order action term's flat input); mujoco:
    :class:`MujocoJointResolver` (pol/hand/prop ctrl-address routing).
    Both are cached per name-tuple and fail fast on unknown names.
    """

    def resolve(self, joint_names):
        """Resolve entry names to the backend addressing primitive."""


# --------------------------------------------------------------------------- #
# isaaclab side: entry names -> positions in a term's flat input slice
# --------------------------------------------------------------------------- #
class IsaaclabJointResolver:
    """Maps entry joint names to indices into one action term's input.

    The term must be configured with ``preserve_order=True`` and an explicit
    joint-name list, so its flat input dim ``i`` corresponds to
    ``term_joint_names[i]`` (owner ruling 2026-09-14, plan §1).  The gather
    index is cached per name-tuple (first resolve, then reuse).

    Unknown names fail fast listing the accepted names.
    """

    def __init__(self, term_joint_names):
        self.term_names = list(term_joint_names)
        self._index = {n: i for i, n in enumerate(self.term_names)}
        self._cache: dict = {}

    def gather(self, joint_names) -> torch.Tensor:
        """``(len(joint_names),)`` int64 gather index.

        For each entry column its position in the term's flat input
        (``term_input[:, gather] = entry.values``); unknown names raise
        ValueError.
        """
        key = tuple(joint_names)
        idx = self._cache.get(key)
        if idx is None:
            missing = [n for n in key if n not in self._index]
            if missing:
                raise ValueError(
                    f"unknown joint name(s) {missing} for this action term; "
                    f"accepted names: {self.term_names}")
            idx = torch.tensor([self._index[n] for n in key], dtype=torch.int64)
            self._cache[key] = idx
        return idx

    # JointResolver protocol name (M2): the isaaclab addressing primitive
    # is the gather index; ``gather`` stays the historical caller face.
    resolve = gather


# --------------------------------------------------------------------------- #
# mujoco side: entry names -> actuator addresses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedJointTargets:
    """Per-entry routing table into the MJ target vectors (numpy).

    ``pol_idx`` / ``hand_idx`` / ``prop_adr`` have one slot per entry
    column; a column that addresses a policy joint carries its index in
    *pol_idx* (and ``-1`` elsewhere), a finger joint carries its
    hand-vector index in *hand_idx*, and a PROP actuated joint
    carries its actuator ctrl address in *prop_adr*.  The three are
    mutually exclusive; the env scatters values accordingly (policy joints
    MUST be fully covered, finger/prop slots fall back to their hold
    value when no entry addresses them).
    """

    pol_idx: object   # np.ndarray int64, -1 where not a policy joint
    hand_idx: object  # np.ndarray int64, -1 where not a hand joint
    prop_adr: object  # np.ndarray int64, ctrl adr where a prop joint


class MujocoJointResolver:
    """Name -> address resolution over an MJ model (cached per name-tuple).

    Built from the env's two name tables — the policy joint order (the 29
    SONIC body joints) and the hand joint order (the non-policy finger DOFs,
    i.e. the env's ``_hand_names``) — plus the PROP actuator tables
    (``{prop_name: {jname: ctrl_adr}}`` for wired articulation
    props; prop-scoped names must not collide with the robot tables --
    namespace fail-fast).  Names outside all tables (or without an
    actuator) fail fast listing the accepted names.
    """

    def __init__(self, model, policy_names, hand_joint_names,
                 prop_actuators=None, model_ir=None):
        import mujoco

        self._mujoco = mujoco
        self._pol_names = list(policy_names)
        self._hand_names = list(hand_joint_names)
        self._pol = {n: i for i, n in enumerate(self._pol_names)}
        self._hand = {n: i for i, n in enumerate(self._hand_names)}
        self._prop = {str(k): dict(v)
                      for k, v in (prop_actuators or {}).items()}
        conflict = sorted({n for t in self._prop.values() for n in t}
                          & (set(self._pol) | set(self._hand)))
        if conflict:
            raise ValueError(
                f"mujoco action resolver: prop joint names {conflict} "
                f"collide with robot joint names (namespace conflict)")
        # name -> (qpos_adr, dof_adr) from the compiled ModelIR (M2); the
        # address table is derived here once and is not a hot-path lookup.
        # The MjModel fallback is the pre-M2 legacy path for stand-alone
        # callers (tests) that do not assemble an IR: this module stays
        # torch-only and may not import unigym.model (layer contract).
        if model_ir is None:
            self.addresses = {
                model.joint(idx).name: (int(model.jnt_qposadr[idx]),
                                        int(model.jnt_dofadr[idx]))
                for idx in range(model.njnt)}
            actuated = {model.joint(int(model.actuator_trnid[a, 0])).name
                        for a in range(model.nu)}
        else:
            self.addresses = {
                row.name: (int(row.qpos_adr), int(row.dof_adr))
                for row in model_ir.joints
                if row.qpos_adr is not None and row.dof_adr is not None}
            actuated = {
                act.target for act in model_ir.actuators
                if act.trntype in ("joint", "jointinparent")}
        # every resolvable joint must have an actuator (fail-fast here, not
        # at PD-write time)
        missing_act = sorted(
            n for n in self._pol_names + self._hand_names
            if n not in actuated)
        if missing_act:
            raise ValueError(
                f"mujoco action resolver: no actuator for joints {missing_act}")
        self._cache: dict = {}

    def resolve(self, joint_names, prop=None) -> ResolvedJointTargets:
        """Resolve entry names to pol/hand/prop addresses (cached).

        Args:
            joint_names: Entry column names (policy / hand / prop-scoped).
            prop: Prop name routing the entry into that prop's actuator
                table (None = the robot tables).

        Returns:
            ResolvedJointTargets with one slot per column.

        Raises:
            ValueError: Unknown name / unknown prop / name outside the
                prop's actuated set.
        """
        key = (prop, tuple(joint_names))
        out = self._cache.get(key)
        if out is None:
            import numpy as np
            pol_idx, hand_idx, prop_adr = [], [], []
            if prop is not None:
                table = self._prop.get(prop)
                if table is None:
                    raise ValueError(
                        f"no prop action entry {prop!r}; known prop "
                        f"entries: {sorted(self._prop)}")
                for n in key[1]:
                    if n not in table:
                        raise ValueError(
                            f"joint name {n!r} is not an actuated joint "
                            f"of prop {prop!r}; actuated joints: "
                            f"{sorted(table)}")
                    pol_idx.append(-1)
                    hand_idx.append(-1)
                    prop_adr.append(table[n])
            else:
                for n in key[1]:
                    if n in self._pol:
                        pol_idx.append(self._pol[n])
                        hand_idx.append(-1)
                        prop_adr.append(-1)
                    elif n in self._hand:
                        pol_idx.append(-1)
                        hand_idx.append(self._hand[n])
                        prop_adr.append(-1)
                    else:
                        raise ValueError(
                            f"unknown joint name {n!r} in the MJ model action "
                            f"tables; policy joints: {self._pol_names}; hand "
                            f"joints: {self._hand_names}; prop entries: "
                            f"{sorted(self._prop)}")
            out = ResolvedJointTargets(
                pol_idx=np.asarray(pol_idx, dtype=np.int64),
                hand_idx=np.asarray(hand_idx, dtype=np.int64),
                prop_adr=np.asarray(prop_adr, dtype=np.int64))
            self._cache[key] = out
        return out
