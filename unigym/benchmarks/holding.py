"""Hold-streak machinery: per-frame predicate derivations (S2 additive).

The engine behind the ``hold_streak`` whitelist template
(benchmarks/terms.py, benchmark_contract.md §9.2): the per-entry
predicate derivations (frame / contact / keypoint-in-region / nested
``all:`` group) and their structural validation.  terms.py keeps the
whitelist registration + delegation; this module owns the derivation
semantics.

Forms (exactly one per entry; ``negate: true`` applies a frame-level
NOT to whichever form):

* frame predicate -- the ``frame_fraction`` derivation + predicate
  forms verbatim (``key``/``axis``/``dims``/``key2``/``post`` plus
  ``{target, tol}`` or ``{op, value}``);
* contact predicate -- the ``contact_any`` derivation
  (``contact``/``reduce_bodies``);
* keypoint-in-region predicate -- the
  ``keypoint_in_region_fraction`` derivation under a nested
  ``keypoint_in_region:`` dict (benchmarks/regions.py);
* latch predicate -- a nested ``latch:`` dict (T-B2): the official
  contact-driven state machines (robocasa Microwave fixture,
  microwave.py:88-110 ``update_state``) as a persistent per-frame 0/1
  bit.  ``state[0] = init`` (the official reset-time value --
  ``update_state`` runs only after a step); for t >= 1 the transition
  is ``off`` FIRST (any condition true -> 0), then ``on`` (rise), then
  hold, reproducing the official ``if door_open: off; elif turned_on
  and stop: off; elif not turned_on and start: on`` priority.  One
  DECLARED narrowing (unit-test pinned): with the latch already off and
  the ``on`` condition and an ``off`` condition simultaneously true,
  the official machine rises (its start branch ignores the stop
  contact -- it oscillates while both stay pressed) while this form
  stays off; every other input cell is transition-exact.
* group predicate -- ``all:`` with a list of the forms above (nested
  conjunction; the group may carry ``negate``) -- e.g. the official
  distractor clause NOT(in_contact AND in_band), which has no
  conjunctive decomposition.  One group level; groups do not nest
  (latch children are the non-group forms only).

The recorded-key readers mirror terms.py's ``_frames``/``_check_frames``
semantics verbatim (kept local to avoid an import cycle -- terms.py
imports this module).  numpy-only.
"""

from __future__ import annotations

import numpy as np

from .base import resolve_episode_key
from .regions import keypoint_region_series

_FRAME_KEYS = {"key", "axis", "dims", "key2", "post", "target", "tol", "op", "value", "negate"}
_CONTACT_KEYS = {"contact", "reduce_bodies", "negate"}
_REGION_KEYS = {"keypoint_in_region", "negate"}
_LATCH_KEYS = {"latch", "negate"}
_GROUP_KEYS = {"all", "negate"}
#: the per-entry form keys, one (and only one) per entry
_FORM_KEYS = ("contact", "key", "keypoint_in_region", "latch", "all")


def _frames(ep, key, articulation=None) -> np.ndarray:
    """Episode key -> float ndarray (T, .); missing key -> ValueError."""
    if not isinstance(key, str):
        raise ValueError(f"key must be a recorded-key string, got {key!r}")
    key = resolve_episode_key(ep.data, key, articulation)
    if key not in ep.data:
        raise ValueError(f"episode data has no key '{key}' (available: {sorted(ep.data)})")
    return np.asarray(ep.data[key], dtype=float)


def _check_frames(x: np.ndarray, what: str) -> np.ndarray:
    """Empty episode (T=0) is a definition error, not a 0/NaN metric."""
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError(f"{what}: empty episode (T=0) cannot be scored")
    return x


# --------------------------------------------------------------------------- #
# derivations
# --------------------------------------------------------------------------- #
def contact_series(ep, spec: dict, what: str, articulation=None) -> np.ndarray:
    """Per-frame boolean series behind ``contact_any``.

    Shared with the ``hold_streak`` contact sub-specs: reduce the group
    table over all non-time axes with any/all.
    """
    key = spec.get("contact")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{what}: 'contact' must be a recorded-key " f"string, got {key!r}")
    full_key = key if key.startswith("contact/") else f"contact/{key}"
    x = _check_frames(_frames(ep, full_key, articulation), what)
    red = spec.get("reduce_bodies")
    if red not in ("any", "all"):
        raise ValueError(f"{what}: reduce_bodies must be 'any' or 'all', got {red!r}")
    if not np.isin(x, (0.0, 1.0)).all():
        raise ValueError(
            f"{what}: key '{full_key}' is not boolean (values outside "
            f"{{0, 1}}) -- declare the group with format: contact"
        )
    axes = tuple(range(1, x.ndim))
    reduced = (x.any(axis=axes) if red == "any" else x.all(axis=axes)) if axes else x.astype(bool)
    return np.asarray(reduced, dtype=bool)


def _frame_series(ep, sub: dict, what: str, articulation=None) -> np.ndarray:
    """The frame-predicate form.

    The frame_fraction derivation plus predicate forms verbatim (band
    {target, tol} or one-sided {op, value}).
    """
    from .terms import _frame_scalars  # lazy: avoids the import cycle

    v = _frame_scalars(ep, sub, what, articulation)
    target = sub.get("target")
    if target is not None:
        tol = float(sub.get("tol", 0.0))
        return np.abs(v - float(target)) <= tol
    op = sub["op"]
    from .terms import _OP_FNS, OPS

    if op not in _OP_FNS:
        raise ValueError(f"{what}.op: unknown op {op!r} " f"(whitelist: {', '.join(OPS)})")
    if "value" not in sub:
        raise ValueError(f"{what}: op form requires a constant 'value'")
    return _OP_FNS[op](v, float(sub["value"]))


def latch_series(ep, inner: dict, what: str, articulation=None) -> np.ndarray:
    """Per-frame boolean state series behind the ``latch`` hold entry.

    ``inner`` (the ``latch:`` dict): ``init`` 0|1 (required, int not
    bool), ``on`` ONE predicate entry (the rise condition, the official
    ``start_button`` contact), ``off`` a NON-EMPTY list of predicate
    entries, any true forcing 0 -- the official machine's forced-off
    branch (door open) and its stop branch, checked before ``on``
    exactly like the official nested ``if door_open / elif stop / elif
    start`` priority.  Each entry is a NON-GROUP ``hold_predicate_series``
    form (frame / contact / region).

    Timeline (the official one): ``state[0] = init`` -- the official
    ``update_state`` runs only AFTER a step, so the reset frame carries
    the declared initial state unconditionally; the transitions consume
    frame t's own conditions from t = 1.
    """
    init = inner.get("init")
    if isinstance(init, bool) or init not in (0, 1):
        raise ValueError(f"{what}.init: must be 0 or 1 (int), got {init!r}")
    on = inner.get("on")
    if not isinstance(on, dict):
        raise ValueError(f"{what}.on: must be one predicate entry (a " f"dict), got {on!r}")
    off = inner.get("off")
    if not isinstance(off, (list, tuple)) or not off:
        raise ValueError(
            f"{what}.off: must be a non-empty list of " f"predicate entries, got {off!r}"
        )
    on_s = hold_predicate_series(ep, on, f"{what}.on", articulation)
    off_s = np.zeros(on_s.shape, dtype=bool)
    for i, sub in enumerate(off):
        off_s |= hold_predicate_series(ep, sub, f"{what}.off[{i}]", articulation)
    out = np.empty(on_s.shape, dtype=bool)
    state = bool(init)
    for t in range(on_s.shape[0]):
        if t > 0:
            if off_s[t]:
                state = False
            elif on_s[t]:
                state = True
        out[t] = state
    return out


def hold_predicate_series(ep, sub: dict, what: str, articulation=None) -> np.ndarray:
    """One ``hold_streak`` sub-spec -> its per-frame boolean series.

    The module docstring is the form whitelist.
    """
    if not isinstance(sub, dict):
        raise ValueError(f"{what}: each entry must be a dict, got {type(sub).__name__}")
    negate = sub.get("negate", False)
    if not isinstance(negate, bool):
        raise ValueError(f"{what}.negate: must be a bool, got {negate!r}")
    out = _dispatch_series(ep, sub, what, articulation)
    return ~out if negate else np.asarray(out, dtype=bool)


def _dispatch_series(ep, sub: dict, what: str, articulation=None) -> np.ndarray:
    """Route one entry to its form derivation (exactly-one rule)."""
    has_contact, has_key = "contact" in sub, "key" in sub
    has_region = "keypoint_in_region" in sub
    has_latch = "latch" in sub
    has_group = "all" in sub
    if has_contact + has_key + has_region + has_latch + has_group != 1:
        raise ValueError(
            f"{what}: exactly one of 'key' (frame predicate), 'contact' "
            "(contact predicate), 'keypoint_in_region' (region predicate)"
            ", 'latch' (state-machine predicate) or 'all' (nested "
            "conjunction group) is required per entry"
        )
    if has_contact:
        _require_keys(sub, _CONTACT_KEYS, "contact", what)
        return contact_series(ep, sub, what, articulation)
    if has_region:
        _require_keys(sub, _REGION_KEYS, "region", what)
        return keypoint_region_series(ep, sub["keypoint_in_region"], what, articulation)
    if has_latch:
        _require_keys(sub, _LATCH_KEYS, "latch", what)
        return latch_series(ep, sub["latch"], what, articulation)
    if has_group:
        return _group_series(ep, sub, what, articulation)
    _require_keys(sub, _FRAME_KEYS, "frame", what)
    return _frame_series(ep, sub, what, articulation)


def _require_keys(sub: dict, allowed: set, name: str, what: str) -> None:
    extra = set(sub) - allowed
    if extra:
        raise ValueError(
            f"{what}: unknown keys {sorted(extra)} "
            f"({name} form allows {', '.join(sorted(allowed))})"
        )


def _group_series(ep, sub: dict, what: str, articulation=None) -> np.ndarray:
    """The nested-conjunction form: frame-level AND over children.

    One level; children may be any non-group form, each optionally
    negated.
    """
    _require_keys(sub, _GROUP_KEYS, "group", what)
    kids = sub["all"]
    if not isinstance(kids, list) or not kids:
        raise ValueError(
            f"{what}: 'all' must be a non-empty list of predicate " f"declarations, got {kids!r}"
        )
    out = hold_predicate_series(ep, kids[0], f"{what}.all[0]", articulation)
    for i, kid in enumerate(kids[1:], start=1):
        s = hold_predicate_series(ep, kid, f"{what}.all[{i}]", articulation)
        out = out & s
    return out


# --------------------------------------------------------------------------- #
# structural validation (construction gate)
# --------------------------------------------------------------------------- #
def validate_hold_entry(sub: dict, where: str) -> list[str]:
    """Structural checks for one NON-group ``hold_streak`` entry.

    The contact / keypoint-in-region / frame forms; shared by the
    top-level entries and ``all:`` group children.
    """
    errors: list[str] = []
    if not isinstance(sub, dict):
        errors.append(f"{where}: entry must be a dict, got " f"{type(sub).__name__}")
        return errors
    if "contact" in sub:
        errors.extend(_validate_contact_entry(sub, where))
    elif "keypoint_in_region" in sub:
        errors.extend(_validate_region_entry(sub, where))
    elif "latch" in sub:
        errors.extend(_validate_latch_entry(sub, where))
    else:
        errors.extend(_validate_frame_entry(sub, where))
    return errors


def _validate_contact_entry(sub: dict, where: str) -> list[str]:
    errors: list[str] = []
    extra = set(sub) - _CONTACT_KEYS
    if extra:
        errors.append(
            f"{where}: unknown keys {sorted(extra)} (contact "
            "form allows contact/reduce_bodies/negate)"
        )
    c = sub.get("contact")
    if not isinstance(c, str) or not c:
        errors.append(
            f"{where}.contact: must be a recorded-key string "
            f"(contact/<name> or a bare group name), got {c!r}"
        )
    if "reduce_bodies" not in sub:
        errors.append(f"{where}: missing required param 'reduce_bodies'")
    elif sub["reduce_bodies"] not in ("any", "all"):
        errors.append(
            f"{where}.reduce_bodies: must be 'any' or 'all', " f"got {sub['reduce_bodies']!r}"
        )
    return errors


def _validate_region_entry(sub: dict, where: str) -> list[str]:
    from .regions import validate_spec

    errors: list[str] = []
    extra = set(sub) - _REGION_KEYS
    if extra:
        errors.append(
            f"{where}: unknown keys {sorted(extra)} (region "
            "form allows keypoint_in_region/negate)"
        )
    inner = sub["keypoint_in_region"]
    if not isinstance(inner, dict):
        errors.append(f"{where}.keypoint_in_region: must be a dict, got " f"{type(inner).__name__}")
    else:
        errors.extend(validate_spec(inner, f"{where}.keypoint_in_region"))
    return errors


def _validate_latch_entry(sub: dict, where: str) -> list[str]:
    errors: list[str] = []
    extra = set(sub) - _LATCH_KEYS
    if extra:
        errors.append(f"{where}: unknown keys {sorted(extra)} (latch " "form allows latch/negate)")
    inner = sub.get("latch")
    if not isinstance(inner, dict):
        errors.append(f"{where}.latch: must be a dict, got " f"{type(inner).__name__}")
        return errors
    extra = set(inner) - {"init", "on", "off"}
    if extra:
        errors.append(f"{where}.latch: unknown keys {sorted(extra)} " "(allows init/on/off)")
    init = inner.get("init")
    if isinstance(init, bool) or init not in (0, 1):
        errors.append(f"{where}.latch.init: must be 0 or 1 (int), " f"got {init!r}")
    on = inner.get("on")
    if not isinstance(on, dict):
        errors.append(f"{where}.latch.on: must be one predicate entry " f"(a dict), got {on!r}")
    elif "all" in on:
        errors.append(
            f"{where}.latch.on: group children are not "
            "supported (the latch takes non-group forms only)"
        )
    else:
        errors.extend(validate_hold_entry(on, f"{where}.latch.on"))
    off = inner.get("off")
    if not isinstance(off, (list, tuple)) or not off:
        errors.append(
            f"{where}.latch.off: must be a non-empty list of " f"predicate entries, got {off!r}"
        )
    else:
        for j, kid in enumerate(off):
            kj = f"{where}.latch.off[{j}]"
            if not isinstance(kid, dict):
                errors.append(f"{kj}: must be a dict, got " f"{type(kid).__name__}")
                continue
            if "all" in kid:
                errors.append(
                    f"{kj}: group children are not supported "
                    "(the latch takes non-group forms only)"
                )
                continue
            errors.extend(validate_hold_entry(kid, kj))
    return errors


def _validate_frame_entry(sub: dict, where: str) -> list[str]:
    from .terms import validate_term_spec

    errors: list[str] = []
    extra = set(sub) - _FRAME_KEYS
    if extra:
        errors.append(
            f"{where}: unknown keys {sorted(extra)} (frame "
            "form allows key/axis/dims/key2/post/target/tol/"
            "op/value/negate)"
        )
    if "pred_spec" in sub:
        errors.append(
            f"{where}.pred_spec: free-form expressions are not "
            "supported in v1 (whitelist-only DSL)"
        )
    else:
        for err in validate_term_spec("frame_fraction", sub):
            errors.append(f"{where}: {err}")
    return errors


def validate_hold_specs(specs, where: str) -> list[str]:
    """Structural checks for a ``hold_streak`` ``specs:`` list.

    The non-empty list rule, the per-entry form dispatch and the
    one-group-level rule.
    """
    errors: list[str] = []
    if not isinstance(specs, (list, tuple)) or not specs:
        errors.append(
            f"{where}.specs: must be a non-empty list of per-frame "
            f"predicate declarations, got {specs!r}"
        )
        return errors
    for i, sub in enumerate(specs):
        w = f"{where}.specs[{i}]"
        if not isinstance(sub, dict):
            errors.append(f"{w}: each entry must be a dict, got " f"{type(sub).__name__}")
            continue
        if not isinstance(sub.get("negate", False), bool):
            errors.append(f"{w}.negate: must be a bool, got " f"{sub.get('negate')!r}")
        n_forms = sum(1 for k in _FORM_KEYS if k in sub)
        if n_forms != 1:
            errors.append(
                f"{w}: exactly one of 'key' (frame predicate), 'contact' "
                "(contact predicate), 'keypoint_in_region' (region "
                "predicate), 'latch' (state-machine predicate) or 'all' "
                "(nested conjunction group) is required per entry"
            )
            continue
        if "all" in sub:
            errors.extend(_validate_group_entry(sub, w))
        else:
            errors.extend(validate_hold_entry(sub, w))
    return errors


def _validate_group_entry(sub: dict, w: str) -> list[str]:
    errors: list[str] = []
    extra = set(sub) - _GROUP_KEYS
    if extra:
        errors.append(f"{w}: unknown keys {sorted(extra)} (group form " "allows all/negate)")
    kids = sub.get("all")
    if not isinstance(kids, list) or not kids:
        errors.append(
            f"{w}.all: must be a non-empty list of predicate " f"declarations, got {kids!r}"
        )
        return errors
    for j, kid in enumerate(kids):
        kj = f"{w}.all[{j}]"
        if not isinstance(kid, dict):
            errors.append(f"{kj}: must be a dict, got " f"{type(kid).__name__}")
            continue
        if "all" in kid:
            errors.append(f"{kj}: nested groups are not supported (one group " "level)")
            continue
        if not isinstance(kid.get("negate", False), bool):
            errors.append(f"{kj}.negate: must be a bool, got " f"{kid.get('negate')!r}")
        errors.extend(validate_hold_entry(kid, kj))
    return errors
