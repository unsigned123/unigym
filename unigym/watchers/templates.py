"""watchers.templates -- the built-in predicate template registry (tier 1).

Design docs/designs/watchers.md §8.1: system templates
are the FIRST-CLASS predicate tier -- typed parameters + a contract
entry + validation, FormulaTask-style (benchmarks/formula.py: validate
returns a list of field-naming errors, check is strict).  Adding a
template = a framework extension (same review flow as FormulaTask
terms): register here + document the entry in
docs/contracts/watchers_contract.md §5.

Every template:

* ``tier``      1 (bundle-resident reads, µs) or 2 (tensor/scan reads).
* ``suggest``   trigger defaults the spec applies unless the user
                overrode them (e.g. ``log`` fires per evaluation).
* ``validate(params, keys) -> [errors]``   types/ranges + key existence
                (keys may be empty -- born-armed mount before the first
                tick validates shape only, first tick re-raises through
                the fail-safe).
* ``check(ctx, params) -> WatchResult``    the semantics; NEVER raises
                for data reasons is NOT guaranteed -- the engine's
                fail-safe wraps every check (3 strikes -> auto-disable).

Templates only answer "is it true right now"; edge/level/once/cooldown
live in the trigger layer (design §8.5 -- semantics decoupled).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from .base import Level, WatchContext, WatchError, WatchResult, finite

if TYPE_CHECKING:
    from unigym.evaluation import EvalContext

__all__ = ["TEMPLATE_REGISTRY", "get_template", "validate_template_params"]

_OPS = ("gt", "lt", "gte", "lte", "eq", "abs_gt", "abs_lt", "abs_gte", "abs_lte")

# shared no-fire result (fast path: templates only pay for formatting
# when they actually fire; the engine never mutates a result)
NOT_FIRED = WatchResult(fired=False)
_REDUCES = ("index", "abs", "last", "min", "max", "abs_max", "norm", "mean")


def _err(params: dict, key: str, expected: str) -> str:
    return f"{key}: expected {expected}, got {params.get(key)!r}"


def _req_num(params: dict, key: str, errors: list, *, positive=False, nonneg=False) -> None:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errors.append(_err(params, key, "a number"))
    elif positive and v <= 0:
        errors.append(f"{key}: must be > 0, got {v}")
    elif nonneg and v < 0:
        errors.append(f"{key}: must be >= 0, got {v}")


def _req_int(params: dict, key: str, errors: list, *, minimum: int = 1) -> None:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        errors.append(_err(params, key, "an integer"))
    elif v < minimum:
        errors.append(f"{key}: must be >= {minimum}, got {v}")


def _req_key(
    params: dict, key: str, errors: list, keys: list[str], prefixes=("obs/", "record/", "action/")
) -> None:
    k = params.get(key)
    if not isinstance(k, str) or not k:
        errors.append(_err(params, key, "a trace-face key string " f"({' | '.join(prefixes)}...)"))
        return
    if not any(k.startswith(p) for p in prefixes):
        errors.append(f"{key}: {k!r} must use the trace face ({' | '.join(prefixes)})")
        return
    if keys and k not in keys:
        errors.append(f"{key}: unknown key {k!r} (available: {keys})")


def _req_entity(params: dict, key: str, errors: list, keys: list[str]) -> None:
    e = params.get(key)
    if not isinstance(e, str) or not e:
        errors.append(_err(params, key, "an entity name ('base'|'pelvis'|" "'robot'|<prop name>)"))
        return
    if e in ("base", "pelvis", "robot"):
        return
    if keys and f"record/object/{e}/pos" not in keys:
        obj_keys = [k for k in keys if k.startswith("record/object/")] or keys
        errors.append(
            f"{key}: unknown prop {e!r} (no record/object/{e}/pos key; " f"available: {obj_keys})"
        )


# --------------------------------------------------------------------------- #
# proximity -- entity/entity or entity/point distance (§8.1)
# --------------------------------------------------------------------------- #
def _proximity_check(ctx: WatchContext, p: dict) -> WatchResult:
    d = ctx.dist(p["a"], p["b"], env=int(p.get("env") or 0))
    if d >= float(p["max_dist"]):
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=f"dist({p['a']}, {p['b']}) = {finite(d)} < " f"{p['max_dist']}",
        data={"dist": finite(d), "max_dist": p["max_dist"], "env": p.get("env") or 0},
    )


# --------------------------------------------------------------------------- #
# key_compare -- trace-key vs threshold (the workhorse)
# --------------------------------------------------------------------------- #
def _key_compare_check(ctx: WatchContext, p: dict) -> WatchResult:
    fired, v, env = ctx.compare(
        p["key"],
        p["op"],
        float(p["value"]),
        env=p.get("env"),
        dim=p.get("dim"),
        reduce=p.get("reduce", "index"),
    )
    if not fired:
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=(f"{p['key']} = {finite(v)} {p['op']} {p['value']} " f"(env {env})"),
        data={
            "key": p["key"],
            "value": finite(v),
            "op": p["op"],
            "threshold": p["value"],
            "env": env,
        },
    )


# --------------------------------------------------------------------------- #
# key_window -- framework-managed sliding window over a scalar statistic
# --------------------------------------------------------------------------- #
def _key_window_check(ctx: WatchContext, p: dict) -> WatchResult:
    v, env = ctx.scalar(
        p["key"], env=p.get("env"), dim=p.get("dim"), reduce=p.get("reduce", "mean")
    )
    win = int(p["window"])
    store: dict = _window_store(ctx, p["key"], win)
    store["buf"].append(v)
    vals = [x for x in store["buf"] if x is not None]
    fired = False
    stat = None
    if len(vals) >= win:
        if p.get("window_op") == "all":
            fired = all(_cmp(x, p["op"], float(p["value"])) for x in vals)
        else:  # mean over the window
            stat = sum(vals) / len(vals)
            fired = _cmp(stat, p["op"], float(p["value"]))
    if not fired:
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=(
            f"{p['key']} window({len(vals)}/{win}) "
            f"{'mean' if stat is not None else 'all'} "
            f"{p['op']} {p['value']}"
        )
        if fired
        else "",
        data={
            "key": p["key"],
            "window_stat": finite(stat) if stat is not None else None,
            "samples": len(vals),
            "window": win,
        },
    )


def _cmp(x: float, op: str, value: float) -> bool:
    return {
        "gt": x > value,
        "lt": x < value,
        "gte": x >= value,
        "lte": x <= value,
        "eq": x == value,
        "abs_gt": abs(x) > value,
        "abs_lt": abs(x) < value,
        "abs_gte": abs(x) >= value,
        "abs_lte": abs(x) <= value,
    }[op]


def _window_store(ctx: WatchContext, key: str, win: int) -> dict:
    """Per-(engine, key, window) deque lives on the engine via the ctx
    hook (keyed by BOTH so concurrent watchers on one key with different
    windows keep independent buffers instead of thrashing one entry)."""
    store = _CURRENT_ENGINE.window_cache
    entry = store.get((key, int(win)))
    if entry is None:
        entry = {"win": int(win), "buf": deque(maxlen=int(win))}
        store[(key, int(win))] = entry
    return entry


# --------------------------------------------------------------------------- #
# contact_state -- group truth + the trigger layer's edge (§8.1)
# --------------------------------------------------------------------------- #
def _contact_check(ctx: WatchContext, p: dict) -> WatchResult:
    on = ctx.contact(p["group"], env=p.get("env"))
    if on != (p.get("state", "on") == "on"):
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.INFO,
        message=f"contact({p['group']}) " f"{'ON' if on else 'OFF'} (env {p.get('env')})",
        data={"group": p["group"], "on": on, "env": p.get("env")},
    )


# --------------------------------------------------------------------------- #
# timeout -- watchdog (fire at tick N; pair with once)
# --------------------------------------------------------------------------- #
def _timeout_check(ctx: WatchContext, p: dict) -> WatchResult:
    if ctx.t < int(p["ticks"]):
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=f"t={ctx.t} >= timeout ticks={p['ticks']}",
        data={"t": ctx.t, "ticks": p["ticks"]},
    )


# --------------------------------------------------------------------------- #
# log -- per-evaluation key recorder (the per-tick trajectory extractor)
# --------------------------------------------------------------------------- #
def _log_check(ctx: WatchContext, p: dict) -> WatchResult:
    env = int(p.get("env") or 0)
    data: dict = {"t": ctx.t, "env": env}
    for k in p["keys"]:
        try:
            v, _ = ctx.scalar(k, env=env, dim=p.get("dim"), reduce=p.get("reduce", "index"))
        except WatchError:
            v = None
        data[k] = finite(v) if v is not None else None
    return WatchResult(
        fired=True,
        level=Level.INFO,
        message="log " + ", ".join(f"{k}={data[k]}" for k in p["keys"]),
        data=data,
    )


# --------------------------------------------------------------------------- #
# tilt / height -- entity posture (settle/fall family)
# --------------------------------------------------------------------------- #
def _entity_quat(ctx: WatchContext, entity: str, env: int):
    name = {
        "base": "record/base_quat_w",
        "pelvis": "record/base_quat_w",
        "robot": "record/base_quat_w",
    }.get(entity, f"record/object/{entity}/quat")
    return ctx.key(name), name


def _tilt_check(ctx: WatchContext, p: dict) -> WatchResult:
    import numpy as np

    from unigym.conditions.tilt import TiltCondition

    explicit_env = p.get("env")
    env = int(explicit_env if explicit_env is not None else 0)
    source, name = _entity_quat(ctx, p["entity"], env)
    row = np.asarray(_to_np(source)[env], dtype=float)  # wxyz (contract)
    w, x, y, z = row / max(np.linalg.norm(row), 1e-9)
    up_z = 1.0 - 2.0 * (y * y + z * z)  # rotated world-up z
    deg = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
    record_key = name[len("record/") :]
    condition = TiltCondition(
        source="base_quat_w",
        record_key=record_key,
        threshold=float(p["max_deg"]),
        unit="deg",
    )
    mask = condition.evaluate(_condition_context(ctx, record_key, source))
    fired = bool(mask[env]) if explicit_env is not None else bool(mask.any())
    if not fired:
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=f"{p['entity']} tilt {deg:.1f} deg > {p['max_deg']}",
        data={
            "entity": p["entity"],
            "tilt_deg": round(deg, 2),
            "max_deg": p["max_deg"],
            "key": name,
            "env": env,
        },
    )


def _to_np(v):
    from .base import _to_numpy

    return _to_numpy(v)


def _condition_context(ctx: WatchContext, record_key: str, value: object) -> EvalContext:
    """Build the read-only EvalContext a template condition evaluates against.

    Live watcher ticks already carry the tick's vector view.  Offline unit
    contexts have no view, so the predicate gets a transient view carrying the
    one record leaf the condition reads; this preserves the documented
    bundle-only fallback without exposing a raw env.
    """
    from unigym.evaluation import EnvView, EvalContext

    if ctx.view is not None:
        return EvalContext(env=ctx.view, t=ctx.t)
    shape = getattr(value, "shape", None)
    size = int(shape[0]) if isinstance(shape, tuple) and shape else 1
    view = EnvView(
        record={record_key: value},
        obs=None,
        env_id=None,
        num_envs=size if size > 0 else 1,
        device="cpu",
        backend="offline watcher",
    )
    return EvalContext(env=view, t=ctx.t)


def _height_check(ctx: WatchContext, p: dict) -> WatchResult:
    from unigym.conditions.height import HeightCondition

    explicit_env = p.get("env")
    env = int(explicit_env if explicit_env is not None else 0)
    pos = ctx.pose(p["entity"], env=env)
    z = float(pos[2])
    key = (
        "record/base_pos_w"
        if p["entity"] in ("base", "pelvis", "robot")
        else f"record/object/{p['entity']}/pos"
    )
    record_key = key[len("record/") :]
    source = ctx.key(key)
    band = p.get("band")
    try:
        if band is not None:
            condition = HeightCondition(record_key=record_key, band=band, dim=2)
            desc = f"outside band {band}"
        else:
            condition = HeightCondition(
                record_key=record_key,
                threshold=float(p["z"]),
                op=p.get("op", "lt"),
                dim=2,
            )
            desc = f"{p.get('op', 'lt')} {p['z']}"
        mask = condition.evaluate(_condition_context(ctx, record_key, source))
    except ValueError as exc:
        raise WatchError(str(exc), ctx.keys()) from exc
    fired = bool(mask[env]) if explicit_env is not None else bool(mask.any())
    if not fired:
        return NOT_FIRED
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=f"{p['entity']} z={z:.3f} {desc}",
        data={"entity": p["entity"], "z": round(z, 4), "rule": desc},
    )


# --------------------------------------------------------------------------- #
# joint_limit -- |q| approaching the policy-order limits (tier 2)
# --------------------------------------------------------------------------- #
def _limit_magnitude(limits):
    """Per-joint |limit| broadcastable against an (N, J) position row.

    ``EnvView.joint_pos_limits`` returns the articulation's ``(lower, upper)``
    pair of ``(N, J)`` tensors in record/body-control policy order, or
    ``None`` when the asset declares no limits.  Anything else is a broken
    view fact -> WatchError (fail-safe domain; never a silent zero-margin
    pass).
    """
    import numpy as np

    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise WatchError(
            f"joint_limit: expected (lower, upper) limits from the view, "
            f"got {type(limits).__name__}"
        )
    lo = np.abs(_to_np(limits[0])).astype(float)
    hi = np.abs(_to_np(limits[1])).astype(float)
    return np.maximum(lo, hi)  # (N, J)


def _infer_joint_articulation(ctx: WatchContext) -> str:
    """Resolve the single articulation exposing ``articulation/<name>/joint_pos``.

    The legacy fixed ``record/joint_pos`` key no longer exists: each
    articulation publishes its own joint positions.  Inference is only
    unambiguous when exactly one articulation exposes the leaf.
    """
    names = sorted(
        key[len("articulation/") : -len("/joint_pos")]
        for key in ctx.record
        if key.startswith("articulation/") and key.endswith("/joint_pos") and key.count("/") == 2
    )
    if not names:
        raise WatchError(
            "joint_limit: no articulation given and no "
            "record/articulation/<name>/joint_pos key in this tick; pass "
            "articulation=<name>"
        )
    if len(names) > 1:
        raise WatchError(
            f"joint_limit: articulation is ambiguous ({names}); pass " f"articulation=<name>"
        )
    return names[0]


def _joint_limit_check(ctx: WatchContext, p: dict) -> WatchResult:
    import numpy as np

    from unigym.conditions.joint_limit import JointLimitCondition
    from unigym.evaluation import EvalContext

    articulation = p.get("articulation")
    key = p.get("key")
    if articulation is None and isinstance(key, str) and key.startswith("record/articulation/"):
        articulation = key.split("/")[2]
    if articulation is None:
        articulation = _infer_joint_articulation(ctx)
    view = ctx.view
    if view is None:
        raise WatchError(
            "joint_limit: no live evaluation view in this context "
            "(offline watch contexts expose no articulation limits)"
        )
    limits = view.joint_pos_limits(articulation)
    if limits is None:
        raise WatchError(f"joint_limit: articulation {articulation!r} declares no joint limits")
    lim = _limit_magnitude(limits)  # (N, J)
    record_key = f"articulation/{articulation}/joint_pos"
    q = _to_np(ctx.key(f"record/{record_key}")).astype(float)
    margin = float(p.get("margin", 0.1))
    band = lim - margin
    condition = JointLimitCondition(
        articulation=articulation,
        record_key=record_key,
        mode="magnitude",
        margin=margin,
    )
    mask = condition.evaluate(EvalContext(env=view, t=ctx.t))
    explicit_env = p.get("env")
    fired = bool(mask[int(explicit_env)]) if explicit_env is not None else bool(mask.any())
    if not fired:
        return NOT_FIRED
    envs = [int(i) for i in np.nonzero(mask.detach().cpu().numpy())[0]]
    worst = float((np.abs(q) - band).max()) if q.size else 0.0
    return WatchResult(
        fired=True,
        level=Level.WARN,
        message=f"joint limit margin {margin}: envs {envs[:8]}" f"{'...' if len(envs) > 8 else ''}",
        data={
            "articulation": articulation,
            "envs": envs,
            "worst_over": round(worst, 4),
            "margin": margin,
        },
    )


# --------------------------------------------------------------------------- #
# registry (contract §5 mirrors this table -- change both)
# --------------------------------------------------------------------------- #
def _t(name, tier, validate, check, params_doc, suggest=None):
    return {
        "name": name,
        "tier": tier,
        "validate": validate,
        "check": check,
        "params": params_doc,
        "suggest": suggest or {},
    }


def _v_proximity(p, keys):
    e: list[str] = []
    _req_entity(p, "a", e, keys)
    b = p.get("b")
    if isinstance(b, str):
        _req_entity(p, "b", e, keys)
    elif not (
        isinstance(b, (list, tuple))
        and len(b) == 3
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b)
    ):
        e.append("b: expected an entity name or [x,y,z] point, got " f"{b!r}")
    _req_num(p, "max_dist", e, positive=True)
    return e


def _v_key_compare(p, keys):
    e: list[str] = []
    _req_key(p, "key", e, keys)
    if p.get("op") not in _OPS:
        e.append(f"op: must be one of {_OPS}, got {p.get('op')!r}")
    _req_num(p, "value", e)
    return e


def _v_key_window(p, keys):
    e: list[str] = []
    _req_key(p, "key", e, keys)
    if p.get("op") not in _OPS:
        e.append(f"op: must be one of {_OPS}, got {p.get('op')!r}")
    _req_num(p, "value", e)
    _req_int(p, "window", e, minimum=2)
    red = p.get("reduce")
    if red is not None and red not in _REDUCES:
        e.append(f"reduce: must be one of {_REDUCES}, got {red!r}")
    return e


def _v_contact(p, keys):
    e: list[str] = []
    g = p.get("group")
    if not isinstance(g, str) or not g:
        e.append(_err(p, "group", "a contact group name string"))
    elif keys and f"record/contact/{g}" not in keys:
        groups = [k[len("record/contact/") :] for k in keys if k.startswith("record/contact/")]
        e.append(
            f"group: unknown contact group {g!r} (available: "
            f"{groups or 'none declared in this config'})"
        )
    if p.get("state", "on") not in ("on", "off"):
        e.append(f"state: must be on|off, got {p.get('state')!r}")
    return e


def _v_timeout(p, keys):
    e: list[str] = []
    _req_int(p, "ticks", e, minimum=1)
    return e


def _v_log(p, keys):
    e: list[str] = []
    ks = p.get("keys")
    if not isinstance(ks, list) or not ks or not all(isinstance(k, str) for k in ks):
        e.append("keys: required non-empty list of trace-face keys")
        return e
    for k in ks:
        _req_key({"key": k}, "key", e, keys)
    return e


def _v_entity_num(p, keys, num_key):
    e: list[str] = []
    _req_entity(p, "entity", e, keys)
    _req_num(p, num_key, e, positive=(num_key == "max_deg"))
    if p.get("band") is None and num_key == "z":
        if p.get("op") not in _OPS:
            e.append(f"op: must be one of {_OPS}, got {p.get('op')!r}")
        _req_num(p, "z", e)
    return e


def _v_height(p, keys):
    e: list[str] = []
    _req_entity(p, "entity", e, keys)
    if p.get("band") is not None:
        band = p["band"]
        if not (
            isinstance(band, (list, tuple))
            and len(band) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in band)
            and band[0] <= band[1]
        ):
            e.append(f"band: must be [lo, hi] with lo <= hi, got {band!r}")
    else:
        if p.get("op") not in _OPS:
            e.append(f"op: must be one of {_OPS}, got {p.get('op')!r}")
        _req_num(p, "z", e)
    return e


def _v_joint_limit(p, keys):
    e: list[str] = []
    art = p.get("articulation")
    if art is not None and (not isinstance(art, str) or not art.strip()):
        e.append(_err(p, "articulation", "a non-empty articulation name"))
    if "margin" in p:
        _req_num(p, "margin", e, nonneg=True)
    key = p.get("key")
    if key is not None:
        if not isinstance(key, str) or not key:
            e.append(_err(p, "key", "a trace-face key string"))
        elif key == "record/joint_pos":
            e.append(
                "key: 'record/joint_pos' is retired; each articulation now "
                "publishes record/articulation/<name>/joint_pos -- pass "
                "articulation=<name> (or omit key)"
            )
        elif not (key.startswith("record/articulation/") and key.endswith("/joint_pos")):
            e.append(
                f"key: {key!r} must be record/articulation/<name>/joint_pos "
                f"(or omit key and pass articulation=<name>)"
            )
        else:
            if keys and key not in keys:
                e.append(f"key: unknown key {key!r} (available: {keys})")
            key_art = key.split("/")[2]
            if isinstance(art, str) and art.strip() and key_art != art:
                e.append(
                    f"key: {key!r} selects articulation {key_art!r}, "
                    f"not the declared articulation {art!r}"
                )
    return e


TEMPLATE_REGISTRY: dict[str, dict] = {
    "proximity": _t(
        "proximity",
        2,
        _v_proximity,
        _proximity_check,
        "a=<entity>, b=<entity|[x,y,z]>, max_dist=<num>, env?<int>",
        {},
    ),
    "key_compare": _t(
        "key_compare",
        1,
        _v_key_compare,
        _key_compare_check,
        "key=<trace key>, op=<gt|lt|gte|lte|eq|abs_*>, value=<num>, "
        "dim?<int>, reduce?<index|abs|last|min|max|abs_max|norm|mean>, "
        "env?<int>",
        {},
    ),
    "key_window": _t(
        "key_window",
        1,
        _v_key_window,
        _key_window_check,
        "key=<trace key>, op=<...>, value=<num>, window=<int>=2, "
        "reduce?<stat for the scalar read>, window_op?<mean|all>, "
        "dim?<int>, env?<int>",
        {},
    ),
    "contact_state": _t(
        "contact_state",
        2,
        _v_contact,
        _contact_check,
        "group=<name>, state?<on|off>=on, env?<int>",
        {},
    ),
    "timeout": _t("timeout", 1, _v_timeout, _timeout_check, "ticks=<int>>=1", {"once": True}),
    "log": _t(
        "log",
        1,
        _v_log,
        _log_check,
        "keys=[<trace key>, ...], env?<int>, dim?<int>, reduce?<...>",
        {"mode": "level", "hold": 1, "cooldown": 0},
    ),
    "tilt": _t(
        "tilt",
        1,
        lambda p, k: _v_entity_num(p, k, "max_deg"),
        _tilt_check,
        "entity=<base|<prop>>, max_deg=<num>, env?<int>",
        {},
    ),
    "height": _t(
        "height",
        1,
        _v_height,
        _height_check,
        "entity=<base|<prop>>, band=[lo,hi] | (op=<...>, z=<num>), " "env?<int>",
        {},
    ),
    "joint_limit": _t(
        "joint_limit",
        2,
        _v_joint_limit,
        _joint_limit_check,
        "articulation?<name> (inferred from "
        "record/articulation/<name>/joint_pos when omitted), margin?<num>=0.1",
        {},
    ),
}

# engine hook: the active WatcherEngine instance (window deques live on
# it -- expr/templates are stateless, framework-managed windows only)
_CURRENT_ENGINE: Any = None


def get_template(name: str) -> dict:
    if name not in TEMPLATE_REGISTRY:
        raise WatchError(f"unknown template {name!r} (available: " f"{sorted(TEMPLATE_REGISTRY)})")
    return TEMPLATE_REGISTRY[name]


def validate_template_params(name: str, params: dict, keys: list[str]) -> list[str]:
    t = get_template(name)
    if not isinstance(params, dict):
        return [f"params: expected a mapping, got {type(params).__name__}"]
    return t["validate"](params, keys or [])
