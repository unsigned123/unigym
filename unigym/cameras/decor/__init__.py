"""Decor generator registry (render-layers design §2/§7): pure functions
to decor geoms.

A generator turns validated params + a :class:`DecorContext` into a list
of decor geometry descriptors (or scene-option toggles).  Generators are
ENGINE-FREE (numpy over precomputed context arrays) so they are
unit-testable without mujoco and honor the "render face model read-only"
iron rule by construction; the mujoco rig builds the context (it owns
the ``mj_contactForce`` call) and injects the descriptors into the scene
buffer with the probe5 field-scrub discipline.

Descriptor shape (dict, JSON-safe)::

    {"type": "capsule"|"arrow"|"cylinder"|"line",
     "from": [x, y, z], "to": [x, y, z],
     "radius": float, "rgba": [r, g, b, a]}

    or an option toggle: ``{"option_flags": {"mjVIS_CONTACTPOINT": True}}``
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..appearance import validate_selector

__all__ = ["GENERATORS", "register_gen", "DecorContext",
           "validate_decor_entry", "run_generators", "gen_names"]

GENERATORS: dict[str, dict] = {}


def register_gen(name: str):
    """Class decorator: ``validate(params, ctx) -> normalized params``
    and ``generate(params, ctx) -> list[descriptor]`` (or the option
    toggle dict).  Re-registering the identical pair is a no-op (module
    re-import safety); a different pair under one name fails fast."""
    def deco(cls):
        entry = {"validate": cls.validate, "generate": cls.generate}
        if name in GENERATORS:
            if (GENERATORS[name]["validate"] is not cls.validate
                    or GENERATORS[name]["generate"] is not cls.generate):
                raise ValueError(
                    f"decor generator {name!r} registered twice with "
                    f"different implementations")
        else:
            GENERATORS[name] = entry
        return cls
    return deco


def gen_names() -> list[str]:
    return sorted(GENERATORS)


@dataclass
class DecorContext:
    """Engine-precomputed render context (built by the mujoco rig).

    ``contact_*`` arrays are row-aligned snapshots of ``mj_data``'s
    active contacts (pos / frame row 0 normal / linear force from
    ``mj_contactForce`` / the (g0, g1) geom pair)."""
    model: object = None
    data: object = None
    contact_pos: np.ndarray = field(default_factory=lambda:
                                   np.zeros((0, 3)))
    contact_normal: np.ndarray = field(default_factory=lambda:
                                       np.zeros((0, 3)))
    contact_force: np.ndarray = field(default_factory=lambda:
                                      np.zeros((0, 3)))
    contact_geoms: np.ndarray = field(default_factory=lambda:
                                      np.zeros((0, 2), dtype=int))
    body_names: list = field(default_factory=list)
    geom_bodyid: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    geom_contype: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    geom_conaffinity: np.ndarray = field(default_factory=lambda:
                                         np.zeros(0, int))
    box_type: int = -1     # mjtGeom.mjGEOM_BOX (rig fills)
    mesh_type: int = -1    # mjtGeom.mjGEOM_MESH (rig fills)
    _body_lower: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._body_lower = [("" if n is None else str(n)).lower()
                            for n in self.body_names]

    # -- selectors -------------------------------------------------------- #
    def _geom_mask(self, sel: dict) -> np.ndarray:
        ngeom = int(self.geom_bodyid.shape[0])
        gb = np.clip(np.asarray(self.geom_bodyid).astype(int), 0,
                     max(len(self._body_lower) - 1, 0))
        if "body" in sel:
            m = np.asarray([self._body_lower[b] == sel["body"].lower()
                            for b in gb])
        elif "body_prefix" in sel:
            p = sel["body_prefix"].lower()
            m = np.asarray([self._body_lower[b].startswith(p) for b in gb])
        elif "class" in sel:
            coll = ((np.asarray(self.geom_contype) != 0)
                    | (np.asarray(self.geom_conaffinity) != 0))
            m = coll if sel["class"] == "collision" else ~coll
        else:  # all
            m = np.ones(ngeom, dtype=bool)
        return m

    def contacts_where(self, selector: dict | None) -> np.ndarray:
        """Contact indices whose either geom matches the (validated)
        selector; ``None`` = all."""
        if not selector or "all" in selector:
            return np.arange(len(self.contact_pos))
        gm = self._geom_mask(selector)
        pairs = np.asarray(self.contact_geoms).reshape(-1, 2)
        if not len(pairs):
            return np.arange(0)
        return np.where(gm[pairs[:, 0]] | gm[pairs[:, 1]])[0]

    def centroid(self, selector: dict | None) -> np.ndarray:
        idx = self.contacts_where(selector)
        if not len(idx):
            raise ValueError(
                "contact_centroid requested but no active contact matches "
                f"selector {selector} ({len(self.contact_pos)} active "
                "contact(s) total)")
        return self.contact_pos[idx].mean(axis=0)

    def target_geoms(self, target: str, *, collision_only: bool = True,
                     types=None) -> tuple[str, list[int]]:
        """Resolve a decor ``target``: case-insensitive body-name prefix
        match -> the body's geoms (collision-class by default -- decor
        annotates physical truth, not visual shells).  Fail-fast with the
        candidate bodies; ``types`` optionally constrains geom types
        (probe4 lesson: cage layout derives from ``geom_size``)."""
        if target == "*":
            raise ValueError("decor target '*' only supports contact-based "
                             "generators (contact_dots/contact_normals/"
                             "contact_mjvis)")
        p = target.lower()
        hits = [b for b, n in enumerate(self._body_lower) if n.startswith(p)]
        if not hits:
            raise ValueError(
                f"decor target {target!r} matches no body; bodies: "
                f"{[n for n in self._body_lower if n]}")
        out: list[int] = []
        for b in hits:
            ids = np.where(np.asarray(self.geom_bodyid) == b)[0]
            for g in ids:
                if collision_only and not (int(self.geom_contype[g])
                                           or int(self.geom_conaffinity[g])):
                    continue
                if types is not None and int(self.model.geom_type[g]) \
                        not in types:
                    continue
                out.append(int(g))
        name = next((self.body_names[b] for b in hits
                     if self.body_names[b]), target)
        if not out:
            raise ValueError(
                f"decor target {target!r}: no "
                f"{'collision ' if collision_only else ''}geom on body(es) "
                f"{[self.body_names[b] for b in hits]}"
                + (f" of types {types}" if types is not None else ""))
        return name, out


def validate_decor_entry(entry, label: str = "decor") -> dict:
    """Fragment-stage validation of one decor entry: registry lookup
    (fail-fast with the available generators) + the generator's own
    param validation.  Returns the normalized entry ``{"gen", "target",
    "params"}`` (params nested)."""
    if not isinstance(entry, dict):
        raise ValueError(f"{label}: decor entry must be a dict, got "
                         f"{entry!r}")
    if "gen" not in entry:
        raise ValueError(
            f"{label}: decor entry missing 'gen' (available generators: "
            f"{gen_names()})")
    name = entry["gen"]
    if name not in GENERATORS:
        raise ValueError(
            f"{label}: unknown decor generator {name!r}; available: "
            f"{gen_names()} (camera_contract §14)")
    target = entry.get("target", "*")
    if not isinstance(target, str) or not target:
        raise ValueError(f"{label}: decor target must be a non-empty "
                         f"string (body name prefix or '*'), got {target!r}")
    reserved = {"gen", "target"}
    if set(entry) == {"gen", "target", "params"}:
        # already-normalized shape (idempotent: fragments carry the
        # nested-params form through combine back into validation)
        params = dict(entry["params"])
    else:
        params = {k: v for k, v in entry.items() if k not in reserved}
    norm = GENERATORS[name]["validate"](params, label)
    return {"gen": name, "target": target, "params": norm}


def run_generators(entries: list[dict], ctx: DecorContext) -> tuple[list, dict]:
    """Run the (already validated) decor entries -> (descriptors,
    option_flags).  Generators are pure: same context + params -> same
    output (the cage dedup and SDF passes are deterministic)."""
    descs: list[dict] = []
    opt_flags: dict = {}
    for e in entries:
        out = GENERATORS[e["gen"]]["generate"](e["params"], e["target"], ctx)
        for d in out:
            if "option_flags" in d:
                opt_flags.update(d["option_flags"])
            else:
                descs.append(d)
    return descs, opt_flags


def _norm_contact_select(params: dict, label: str, key="contact_select"):
    sel = params.get(key)
    return validate_selector(sel) if sel is not None else None


def _check_keys(params: dict, legal: tuple, label: str) -> None:
    unknown = sorted(set(params) - set(legal))
    if unknown:
        raise ValueError(f"{label}: unknown decor param(s) {unknown}; "
                         f"legal: {list(legal)}")


def _pos_float(params, key, default, label) -> float:
    v = params.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"{label}: {key} must be a positive number, got "
                         f"{v!r}")
    return float(v)


# builtin generators self-register at package import (the registry is the
# only coupling between spec validation and the generator modules).
from . import axes as _axes          # noqa: F401  (registry side effect)
from . import cage as _cage          # noqa: F401
from . import contacts as _contacts  # noqa: F401


# generator param shared shape: contact_select (a selector or None)
