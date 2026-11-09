"""Appearance selectors: declarative scene-buffer recoloring (design §3/§4).

An appearance entry is pure JSON::

    {"select": {"body_prefix": "right_"}, "rgba": [1, 1, 1, 1.0],
     "matid": -1}

selector vocabulary: ``body`` (exact body name) /
``body_prefix`` / ``class("collision"|"visual")`` (resolved through the
load-time reclassification's contype/conaffinity audit rule -- the SAME
rule as the flags face) / ``all``.

Semantics (render-layers Phase 1):

* The engine is ENGINE-FREE (model arrays in -> per-geom paint table out)
  so selector resolution is unit-testable without mujoco; the mujoco rig
  applies the table to ``scn.geoms`` after ``mjv_updateScene`` (the
  L1.5 scene-buffer route probe1(f) proved bitwise-identical to
  load-time material stripping).
* Paint rule: FIRST match wins, in a canonical selector precedence order
  (exact body > longer prefix > class > all) so merged specs are
  order-independent (A+B == B+A) and a ``body: world`` floor rule beats a
  ``body_prefix: ""`` ghost rule regardless of declaration order.
* ``matid``/``texid`` default to -1 (strip material -- the recolor is
  "flat rgba"); an entry may keep a material by declaring ``matid``
  explicitly.  Unmatched geoms keep their historical appearance (a
  partial spec never silently repaints the world; view ``visual`` with
  no entries is the historical render bitwise).
"""
from __future__ import annotations

import numpy as np

__all__ = ["SELECTOR_KEYS", "SELECTOR_CLASSES",
           "validate_selector", "selector_key", "parse_color",
           "appearance_plan", "APPEARANCE_ENTRY_KEYS"]

SELECTOR_KEYS = ("body", "body_prefix", "class", "all")
SELECTOR_CLASSES = ("collision", "visual")
APPEARANCE_ENTRY_KEYS = ("select", "rgba", "matid")

_RANK = {"body": 0, "body_prefix": 1, "class": 2}
# selector precedence: exact body, then NON-EMPTY prefixes (longest
# first), then class rules, then the catch-alls (the empty prefix and
# 'all' are the same "everything" intent).  Deterministic +
# order-independent.


def validate_selector(sel) -> dict:
    """Normalize one selector dict; fail-fast with the legal vocabulary.

    Exactly ONE key must be present (a two-key selector is ambiguous by
    construction).  ``all`` must be the literal ``True``.
    """
    if not isinstance(sel, dict) or not sel:
        raise ValueError(
            f"appearance selector must be a non-empty dict with exactly "
            f"one of {list(SELECTOR_KEYS)}, got {sel!r}")
    unknown = sorted(set(sel) - set(SELECTOR_KEYS))
    if unknown:
        raise ValueError(
            f"appearance selector: unknown key(s) {unknown}; legal "
            f"selector keys: {list(SELECTOR_KEYS)} (camera_contract §14)")
    if len(sel) > 1:
        raise ValueError(
            f"appearance selector takes exactly ONE key, got "
            f"{sorted(sel)} -- split it into separate appearance entries")
    kind, val = next(iter(sel.items()))
    if kind == "all":
        if val is not True:
            raise ValueError(f"appearance selector 'all' must be true, "
                             f"got {val!r}")
        return {"all": True}
    if kind == "class":
        if val not in SELECTOR_CLASSES:
            raise ValueError(
                f"appearance selector 'class' must be one of "
                f"{list(SELECTOR_CLASSES)}, got {val!r}")
        return {"class": str(val)}
    if not isinstance(val, str):
        raise ValueError(
            f"appearance selector {kind!r} must be a string, got {val!r}")
    return {kind: val}


def selector_key(sel: dict) -> str:
    """Canonical merge key of a VALIDATED selector (``body_prefix:`` for
    the empty prefix).  The appearance merge is keyed on this string:
    two entries with the same key from two presets = a conflict, not a
    silent override (camera_contract §14.10)."""
    kind, val = next(iter(sel.items()))
    return "all" if kind == "all" else f"{kind}:{val}"


def parse_color(color) -> list[float]:
    """``"#rrggbb"`` -> ``[r, g, b]`` floats; a 3-sequence passes through
    (validated).  Decor params and preset tables share this parser."""
    if isinstance(color, str):
        c = color.strip()
        if not c.startswith("#") or len(c) != 7:
            raise ValueError(
                f"color must be '#rrggbb' or [r,g,b] floats, got {color!r}")
        try:
            return [int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        except ValueError:
            raise ValueError(
                f"color must be '#rrggbb' or [r,g,b] floats, got {color!r}"
            ) from None
    if (isinstance(color, (list, tuple)) and len(color) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in color)):
        return [float(x) for x in color]
    raise ValueError(
        f"color must be '#rrggbb' or [r,g,b] floats, got {color!r}")


def _validate_entry(entry, label: str) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"{label}: appearance entry must be a dict, got "
                         f"{entry!r}")
    unknown = sorted(set(entry) - set(APPEARANCE_ENTRY_KEYS))
    if unknown:
        raise ValueError(
            f"{label}: unknown appearance entry key(s) {unknown}; legal "
            f"keys: {list(APPEARANCE_ENTRY_KEYS)}")
    if "select" not in entry:
        raise ValueError(f"{label}: appearance entry missing 'select'")
    if "rgba" not in entry:
        raise ValueError(
            f"{label}: appearance entry missing 'rgba' (a recolor with "
            f"no color is a no-op -- declare it or drop the entry)")
    rgba = entry["rgba"]
    if (not isinstance(rgba, (list, tuple)) or len(rgba) != 4
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and 0.0 <= float(x) <= 1.0 for x in rgba)):
        raise ValueError(
            f"{label}: appearance rgba must be 4 floats in [0, 1], got "
            f"{rgba!r}")
    matid = entry.get("matid", -1)
    if isinstance(matid, bool) or not isinstance(matid, int):
        raise ValueError(f"{label}: appearance matid must be an int, got "
                         f"{matid!r}")
    return {"select": validate_selector(entry["select"]),
            "rgba": [float(x) for x in rgba], "matid": int(matid)}


def validate_appearance(entries, label: str = "appearance") -> list[dict]:
    """Validate a whole appearance list (fragment stage; the merge keys
    off :func:`selector_key`)."""
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"{label} must be a list of entries, got "
                         f"{entries!r}")
    return [_validate_entry(e, label) for e in entries]


def appearance_plan(entries: list[dict], body_names, geom_bodyid,
                    geom_contype, geom_conaffinity, geom_group,
                    render_groups=None) -> dict:
    """Per-geom-id paint table from validated appearance entries.

    Args:
        entries: validated appearance entries (validate_appearance).
        body_names: sequence of body names (index = body id).
        geom_bodyid / geom_contype / geom_conaffinity / geom_group: model
            arrays (geom_group = the reclassified layout).
        render_groups: optional per-config class override (flags face).

    Returns ``{"painted": bool (ngeom,), "rgba": float (ngeom, 4),
    "matid": int (ngeom,)}``.  FIRST match wins in the canonical
    precedence order, so the plan is order-independent and a partial
    spec leaves unmatched geoms untouched (``painted`` False)."""
    ngeom = int(np.asarray(geom_bodyid).shape[0])
    names = np.asarray(["" if n is None else str(n) for n in body_names],
                       dtype=object)
    gb = np.asarray(geom_bodyid).astype(int)
    body_of_geom = names[np.clip(gb, 0, len(names) - 1)]
    is_collision = np.asarray(geom_contype) != 0
    is_collision |= np.asarray(geom_conaffinity) != 0
    if render_groups:
        from .geomflags import collision_mask
        is_collision = collision_mask(geom_contype, geom_conaffinity,
                                      geom_group, render_groups)

    def ordered(entries_):
        def sort_key(e):
            sel = e["select"]
            kind, val = next(iter(sel.items()))
            catchall = kind == "all" or (kind == "body_prefix"
                                         and str(val) == "")
            rank = 3 if catchall else _RANK[kind]
            spec = 0 if catchall else len(str(val))
            return (rank, -spec, selector_key(sel))
        return sorted(entries_, key=sort_key)

    painted = np.zeros(ngeom, dtype=bool)
    rgba = np.zeros((ngeom, 4), dtype=np.float64)
    matid = np.full(ngeom, -1, dtype=np.int64)
    for e in ordered(entries):
        sel = e["select"]
        kind, val = next(iter(sel.items()))
        if kind == "body":
            mask = body_of_geom == val
        elif kind == "body_prefix":
            mask = np.asarray([n.startswith(val) for n in body_of_geom])
        elif kind == "class":
            mask = is_collision if val == "collision" else ~is_collision
        else:  # all
            mask = np.ones(ngeom, dtype=bool)
        fresh = mask & ~painted
        painted |= fresh
        rgba[fresh] = np.asarray(e["rgba"], dtype=np.float64)
        matid[fresh] = int(e["matid"])
    return {"painted": painted, "rgba": rgba, "matid": matid}
