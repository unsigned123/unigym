"""Behavior report over existing run shards -- zero collection (toolkit C).

`unigym analyze <run_dir>` reads ``meta.json + shard_*.npz`` (contract:
docs/contracts/analyze_contract.md) and produces:

1. entity pairwise min-distance curves (shard-visible entities only:
   ``robot/base`` from ``base_pos`` + ``prop/<name>`` from
   ``object/<name>/pos``; body-part positions need FK -- not in shards,
   declared limitation),
2. contact event timeline per declared ``contact/<group>`` key
   (first-contact tick, enter/exit transitions, contact-tick count),
3. joint travel (per policy joint min/max/range + argmax tick; prop
   joints as ``|leaf|max`` -- the manual-convention value),
4. fall/phase axis: per-episode ``fall_tick`` from the base-z threshold
   (run's fall termination height, else 0.45) + verdict.

Outputs: ``behavior_report.md`` + ``behavior_report.json`` + PNG curves
(PIL; charts.py) -- flat files in the run dir so the portal artifact
whitelist can serve them without daemon changes.  Pure numpy/PIL; zero
sim/daemon imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .charts import line_chart

FALL_THRESHOLD_DEFAULT = 0.45  # design §4.2: manual-convention value
TOP_JOINTS = 10  # joint table rows (43-joint robots)


class AnalyzeError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(
    run_dir: str | Path,
    env: int = 0,
    max_ticks: int | None = None,
    extra_keys: set[str] | None = None,
) -> dict:
    """meta + concatenated shard arrays for one env (fail-fast face).

    ``extra_keys`` (the --series face): shard keys outside the report
    defaults to load verbatim (any recorded key is fair game).
    """
    run = Path(run_dir)
    meta_path = run / "meta.json"
    if not meta_path.is_file():
        raise AnalyzeError(
            f"{run}: no meta.json (not a unigym collect run dir?)",
            hint="pass a run output dir (meta.json + shard_*.npz) or a "
            "run name under the output root",
        )
    meta = json.loads(meta_path.read_text())
    shards = sorted(run.glob("shard_*.npz"))
    if not shards:
        raise AnalyzeError(
            f"{run}: zero shard_*.npz files",
            hint="analyze reads collect shards -- record a " "run first (unigym run CONFIG)",
        )
    keys: set[str] | None = None
    cols: dict[str, list] = {}
    total = 0
    for shard in shards:
        with np.load(shard) as z:
            if keys is None:
                keys = set(z.files)
                cols = {k: [] for k in _NEEDED(keys) | (set(extra_keys or ()) & keys)}
            for k in cols:
                v = np.asarray(z[k])
                cols[k].append(v[:, int(env)])  # shard layout (T, N, ...)
            total += int(z[z.files[0]].shape[0])
            if max_ticks is not None and total >= int(max_ticks):
                break
    data = {k: np.concatenate(v, axis=0)[:max_ticks] for k, v in cols.items()}
    data["tick"] = np.arange(total if max_ticks is None else min(total, int(max_ticks)))
    return {"dir": run, "meta": meta, "data": data, "ticks": data["tick"].size}


def _NEEDED(keys: set[str]) -> set[str]:
    want = {"episode_id", "time_in_episode"}
    want |= {k for k in keys if k.startswith("contact/") or k.startswith("object/")}
    want |= {
        k
        for k in keys
        if k.startswith("articulation/") and (k.endswith("/base_pos") or k.endswith("/joint_pos"))
    }
    return want & keys


def _articulation_names(data: dict) -> list[str]:
    """Sorted articulation names carrying any ``articulation/<n>/...`` key."""
    names = {
        str(k).split("/", 2)[1]
        for k in data
        if isinstance(k, str) and k.startswith("articulation/") and str(k).count("/") >= 2
    }
    return sorted(names)


def _state_key(data: dict, leaf: str):
    """The unique articulation's recorded *leaf* key (``None`` when absent).

    With several articulations the single-vector report refuses to pick a
    primary: the caller gets ``None`` and the section degrades honestly.
    """
    names = _articulation_names(data)
    if len(names) != 1:
        return None
    key = f"articulation/{names[0]}/{leaf}"
    return key if key in data else None


# --------------------------------------------------------------------------- #
# derived views
# --------------------------------------------------------------------------- #
def _episodes(episode_id: np.ndarray) -> list[dict]:
    """(start, end_excl, id) segments of env-0 episode_id changes."""
    eid = np.asarray(episode_id).reshape(episode_id.shape[0], -1)[:, 0]
    bounds = np.nonzero(np.diff(eid) != 0)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [eid.size]])
    return [
        {"ep": i, "start": int(s), "end": int(e), "id": int(eid[s])}
        for i, (s, e) in enumerate(zip(starts, ends))
        if e > s
    ]


def _entities(data: dict) -> dict[str, np.ndarray]:
    ents: dict[str, np.ndarray] = {}
    base_keys = [
        k for k in sorted(data) if k.startswith("articulation/") and k.endswith("/base_pos")
    ]
    for k in base_keys:
        name = k.split("/", 2)[1]
        label = "robot/base" if len(base_keys) == 1 else f"robot/{name}"
        ents[label] = data[k][:, :3]
    for k in sorted(data):
        if k.startswith("object/") and k.endswith("/pos"):
            name = k[len("object/") : -len("/pos")]
            ents[f"prop/{name}"] = data[k][:, :3]
    return ents


def _ent_order(n: str) -> tuple:
    """robot/base leads (design §4.3 table shape), then alphabetical."""
    return (not n.startswith("robot/"), n)


def _pair_distances(ents: dict[str, np.ndarray]) -> list[dict]:
    names = sorted(ents, key=_ent_order)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = np.linalg.norm(ents[a] - ents[b], axis=-1)
            out.append({"pair": f"{a} <-> {b}", "a": a, "b": b, "d": d})
    return out


def _contact_series(data: dict) -> dict[str, np.ndarray]:
    """contact/<group> -> per-tick any-contact bool (B/F axes reduced)."""
    out = {}
    for k in sorted(data):
        if k.startswith("contact/"):
            v = np.asarray(data[k])
            out[k] = v.reshape(v.shape[0], -1).any(axis=1)
    return out


def _fall_threshold(meta: dict) -> tuple[float, str]:
    terms = (meta.get("config") or {}).get("terminations") or []
    for t in terms:
        name = t.get("name", "") if isinstance(t, dict) else ""
        if "fall" in str(name):
            z = (t.get("params") or {}).get("height_threshold")
            if z is not None:
                return float(z), f"terminations:{name}.height_threshold"
    return FALL_THRESHOLD_DEFAULT, "default (no fall termination height)"


def _joint_travel(joint_pos: np.ndarray, names: list[str]) -> list[dict]:
    out = []
    for j in range(joint_pos.shape[-1]):
        v = joint_pos[:, j]
        out.append(
            {
                "joint": names[j] if j < len(names) else f"j{j}",
                "min": float(v.min()),
                "max": float(v.max()),
                "range": float(v.max() - v.min()),
                "argmax_tick": int(v.argmax()),
                "argmin_tick": int(v.argmin()),
            }
        )
    return sorted(out, key=lambda r: -r["range"])


def _prop_joints(data: dict) -> list[dict]:
    out = []
    for k in sorted(data):
        if k.startswith("object/") and "/joints/" in k:
            v = np.asarray(data[k]).ravel()
            out.append(
                {
                    "key": k,
                    "abs_max": float(np.abs(v).max()),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "argmax_tick": int(np.abs(v).argmax()),
                }
            )
    return out


def _director_doc(run: Path, meta: dict) -> dict | None:
    """The director face of a run (director_contract.md §7).

    Live state comes from ``director/status.json`` (Director.on_run_end);
    a run started with --direct but without the file (crash path) still
    gets the enabled flag from meta.json.  None = physics-legal run.
    """
    sp = run / "director" / "status.json"
    if sp.is_file():
        try:
            return json.loads(sp.read_text())
        except ValueError:
            pass
    if meta.get("director_enabled"):
        return {"enabled": True, "commands": None, "log": None, "episodes": []}
    return None


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def _joint_meta(data: dict, meta: dict) -> tuple:
    """``(namespaced joint_pos key, recorded joint names)`` for one run.

    Legacy cardinality projection: only when the recording has exactly ONE
    articulation.  Multi-articulation runs report under
    ``doc["articulations"][name]`` instead (no primary; see
    :func:`_joint_travel_by_articulation`).
    """
    key = _state_key(data, "joint_pos")
    if key is None:
        return None, []
    art = key.split("/", 2)[1]
    record = (meta.get("articulations") or {}).get(art) or {}
    return key, list(record.get("joint_names") or [])


def _articulation_report(data: dict, meta: dict) -> dict:
    """Per-articulation orders + joint travel for EVERY declared articulation.

    v103.5 records robot state under ``articulation/<name>/...`` and the
    per-articulation orders under ``meta["articulations"]``.  There is no
    primary articulation: every declared name gets its own entry (a name
    declared in meta without a joint_pos column keeps its orders and an
    empty travel table).
    """
    records = meta.get("articulations") or {}
    names = list(records)
    names += [n for n in _articulation_names(data) if n not in records]
    out: dict = {}
    for name in names:
        record = records.get(name) or {}
        key = f"articulation/{name}/joint_pos"
        joint_names = list(record.get("joint_names") or [])
        if key in data:
            rows = _joint_travel(data[key], joint_names)
        else:
            rows = []
        out[name] = {
            "joint_names": joint_names,
            "foot_bodies": list(record.get("foot_bodies") or []),
            "joint_pos_key": key if key in data else None,
            "joint_travel": rows[:TOP_JOINTS],
            "joint_travel_truncated": max(0, len(rows) - TOP_JOINTS),
        }
    return out


def analyze_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    env: int = 0,
    max_ticks: int | None = None,
    charts: bool = True,
    gate: bool = True,
) -> dict:
    """Full behavior report (doc dict + md/png artifacts on disk).

    ``gate`` (default on): attach the declaration-driven gate
    section (unigym/analysis/gate.py).  Purely additive: every existing doc
    key and md section keeps its exact values; old runs without
    results.json degrade to the missing-results ``na`` status.
    """
    loaded = load_run(run_dir, env=env, max_ticks=max_ticks)
    data, meta = loaded["data"], loaded["meta"]
    run = loaded["dir"]
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    eps = _episodes(data["episode_id"])
    ents = _entities(data)
    pairs = _pair_distances(ents)
    contacts = _contact_series(data)
    z_thresh, z_src = _fall_threshold(meta)
    base_z_key = _state_key(data, "base_pos")
    base_z = data[base_z_key][:, 2] if base_z_key is not None else None

    articulation_report = _articulation_report(data, meta)
    # legacy cardinality projection (single-articulation runs only)
    joint_pos_key, joint_names = _joint_meta(data, meta)
    joints = _joint_travel(data[joint_pos_key], joint_names) if joint_pos_key is not None else []
    prop_joints = _prop_joints(data)

    # per-episode rows (the design §4.3 table shape)
    director = _director_doc(run, meta)
    dir_by_ep = {
        e.get("episode_id"): bool(e.get("directed")) for e in (director or {}).get("episodes") or []
    }
    ep_rows = []
    for e in eps:
        s, ee = e["start"], e["end"]
        row = {
            "ep": e["ep"],
            "ticks": [s, ee - 1],
            "n": ee - s,
            "fall_tick": None,
            "min_dist": {},
            "first_contact": {},
            "leaf_travel": {},
            "verdict": "stood",
            "directed": bool(dir_by_ep.get(e["id"])),
        }
        if base_z is not None:
            below = np.nonzero(base_z[s:ee] < z_thresh)[0]
            if below.size:
                row["fall_tick"] = int(s + below[0])
        for p in pairs:
            row["min_dist"][p["pair"]] = {
                "min": float(p["d"][s:ee].min()),
                "tick": int(s + p["d"][s:ee].argmin()),
            }
        for name, c in contacts.items():
            on = np.nonzero(c[s:ee])[0]
            row["first_contact"][name] = int(s + on[0]) if on.size else None
        for pj in prop_joints:
            v = np.asarray(data[pj["key"]]).ravel()[s:ee]
            row["leaf_travel"][pj["key"]] = float(np.abs(v).max())
        if row["fall_tick"] is not None:
            row["verdict"] = f"fell@t{row['fall_tick']}"
        else:
            touched = [(n, t) for n, t in row["first_contact"].items() if t is not None]
            if touched:
                n, t = min(touched, key=lambda x: x[1])
                row["verdict"] = f"touched:{n}@t{t}"
        ep_rows.append(row)

    # contact event timeline (enter/exit transitions)
    timeline = {}
    for name, c in contacts.items():
        edges = np.nonzero(np.diff(c.astype(np.int8)))[0] + 1
        events = []
        for t in edges:
            events.append({"tick": int(t), "event": "enter" if c[t] else "exit"})
        if edges.size:
            n_enter = int(c[edges].sum())
            n_exit = int(edges.size) - n_enter
        else:  # constant face: True = born in contact
            n_enter, n_exit = (int(bool(c[0])) if c.size else 0), 0
        timeline[name] = {
            "first_tick": (int(np.nonzero(c)[0][0]) if c.any() else None),
            "contact_ticks": int(c.sum()),
            "enter_exit": [n_enter, n_exit],
            "events": events[:200],
            "events_truncated": max(0, len(events) - 200),
        }

    doc = {
        "schema": 1,
        "run": {
            "dir": str(run),
            "name": meta.get("run_name"),
            "ticks": loaded["ticks"],
            "env": env,
            "num_envs": meta.get("num_envs"),
            "policy_dt": meta.get("policy_dt"),
        },
        "fall_axis": {
            "threshold": z_thresh,
            "source": z_src,
            "episodes_fell": sum(1 for r in ep_rows if r["fall_tick"] is not None),
        },
        # directed boundary (contract §7): directed episodes are
        # labelled independently and NEVER fold into physics-legal verdicts
        "directed": {
            "enabled": bool(director and director.get("enabled")),
            "commands": (director or {}).get("commands"),
            "log": (director or {}).get("log"),
            "episodes_directed": [r["ep"] for r in ep_rows if r["directed"]],
            "note": (
                "directed successes are fixture validations, not " "policy capability"
                if any(r["directed"] for r in ep_rows)
                else None
            ),
        },
        "entities": sorted(ents, key=_ent_order),
        "episodes": ep_rows,
        "pairwise": [
            {"pair": p["pair"], "global_min": float(p["d"].min()), "at_tick": int(p["d"].argmin())}
            for p in pairs
        ],
        "contacts": timeline,
        # per-articulation truth (all declared names, no primary)
        "articulations": articulation_report,
        # legacy single-articulation projection (cardinality fact only)
        "joint_travel": joints[:TOP_JOINTS],
        "joint_travel_truncated": max(0, len(joints) - TOP_JOINTS),
        "prop_joints": prop_joints,
        "limitations": [
            "entities are shard-visible only (robot/base + "
            "prop/<name>); body-part positions need FK",
            f"single-env view (env {env}) of " f"{meta.get('num_envs')} envs",
        ]
        if (meta.get("num_envs") or 1) > 1
        else ["entities are shard-visible only (robot/base + prop/<name>)"],
    }
    if gate:
        from . import gate as _gate

        try:
            doc["gate"] = _gate.gate_doc(run)
        except _gate.GateError as e:
            doc["gate"] = {
                "available": False,
                "status": "na(results 残缺)",
                "columns": [],
                "rows": [],
                "episodes": [],
                "summary": None,
                "gate": None,
                "error": str(e),
                "hint": e.hint,
            }

    pngs: list[str] = []
    if charts:
        for p in pairs:
            slug = p["pair"].replace(" <-> ", "__").replace("/", "_")
            f = line_chart(
                out / f"dist__{slug}.png", f"min distance: {p['pair']} (m)", [(p["pair"], p["d"])]
            )
            pngs.append(str(f))
        if base_z is not None:
            series = [(f"ep{e['ep']}", base_z[e["start"] : e["end"]]) for e in eps[:8]]
            f = line_chart(out / "base_z.png", f"base z (fall threshold {z_thresh:.2f} m)", series)
            pngs.append(str(f))
        for name, c in contacts.items():
            slug = name.replace("/", "_")
            f = line_chart(
                out / f"contact__{slug}.png", f"contact: {name}", [(name, c.astype(np.float32))]
            )
            pngs.append(str(f))
    doc["charts"] = pngs

    (out / "behavior_report.json").write_text(json.dumps(doc, indent=1))
    (out / "behavior_report.md").write_text(render_md(doc))
    return doc


def _joint_rows_md(rows: list[dict], truncated: int) -> list[str]:
    """One joint-travel table (+ its truncation note)."""
    L = ["joint | min | max | range | argmax_tick", "---|---|---|---|---"]
    L += [
        f"{j['joint']} | {j['min']:.4f} | {j['max']:.4f} | "
        f"{j['range']:.4f} | t={j['argmax_tick']}"
        for j in rows
    ] or ["- | - | - | - | -"]
    if truncated:
        L.append(f"(... +{truncated} more joints in report.json)")
    return L


def _articulations_md(arts: dict) -> list[str]:
    """Per-articulation orders + joint travel (multi-articulation runs)."""
    if len(arts) <= 1:
        return []
    L = [
        "",
        "## joint travel per articulation",
        "articulation | joints | foot_bodies",
        "---|---|---",
    ]
    L += [
        f"{name} | {len(a['joint_names'])} | " f"{', '.join(a['foot_bodies']) or '-'}"
        for name, a in arts.items()
    ]
    for name, a in arts.items():
        L += ["", f"### {name} (top {len(a['joint_travel'])} by range)"]
        L += _joint_rows_md(a["joint_travel"], a["joint_travel_truncated"])
    return L


def render_md(doc: dict) -> str:
    """The markdown face (design §4.3 table shape)."""
    r, fa = doc["run"], doc["fall_axis"]
    L = [
        f"=== behavior report: {r['name'] or r['dir']} "
        f"({r['ticks']} ticks, {len(doc['episodes'])} episodes, "
        f"env {r['env']}) ==="
    ]
    dd = doc.get("directed") or {}
    if dd.get("enabled"):
        L.append(
            f"**DIRECTED RUN -- director commands: "
            f"{dd.get('commands', '?')} | directed episodes: "
            f"{dd['episodes_directed'] or 'none'} | directed "
            f"successes are fixture validations, not policy "
            f"capability**"
        )
    if doc.get("gate") is not None:
        from .gate import render_gate_md

        L += render_gate_md(doc["gate"])
    L.append(
        f"fall axis: base_z < {fa['threshold']:.2f} "
        f"({fa['source']}); episodes fell = {fa['episodes_fell']}"
    )
    L.append(f"entities: {', '.join(doc['entities']) or '-'}")
    L += [
        "",
        "## episodes",
        "ep | ticks | fall_tick | min_dist | first_contact | " "leaf_travel | directed | verdict",
        "---|---|---|---|---|---|---|---",
    ]
    for row in doc["episodes"]:
        md = (
            "; ".join(f"{k}={v['min']:.3f}m@t{v['tick']}" for k, v in row["min_dist"].items())
            or "-"
        )
        fc = (
            "; ".join(
                f"{k.replace('contact/', '')}@t{t}"
                for k, t in row["first_contact"].items()
                if t is not None
            )
            or "none"
        )
        lt = "; ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in row["leaf_travel"].items()) or "-"
        L.append(
            f"{row['ep']} | {row['ticks'][0]}..{row['ticks'][1]} | "
            f"{row['fall_tick'] if row['fall_tick'] is not None else 'none'}"
            f" | {md} | {fc} | {lt} | {str(row.get('directed', False)).lower()} "
            f"| {row['verdict']}"
        )
    L += ["", "## pairwise min distance (world xyz)", "pair | global_min | at tick", "---|---|---"]
    L += [f"{p['pair']} | {p['global_min']:.4f} | t={p['at_tick']}" for p in doc["pairwise"]] or [
        "- | - | -"
    ]
    L += [
        "",
        "## contact events",
        "group | first_tick | contact_ticks | " "enter/exit",
        "---|---|---|---",
    ]
    L += [
        f"{k} | {v['first_tick'] if v['first_tick'] is not None else 'none'}"
        f" | {v['contact_ticks']} | {v['enter_exit'][0]}/"
        f"{v['enter_exit'][1]}"
        for k, v in doc["contacts"].items()
    ] or ["(no contact/<group> keys in shards) | - | - | -"]
    L += ["", f"## joint travel (top {len(doc['joint_travel'])} by range)"]
    L += _joint_rows_md(doc["joint_travel"], doc["joint_travel_truncated"])
    L += _articulations_md(doc.get("articulations") or {})
    if doc["prop_joints"]:
        L += [
            "",
            "## prop joints (|leaf|max = manual convention)",
            "key | min | max | |leaf|max | argmax_tick",
            "---|---|---|---|---",
        ]
        L += [
            f"{j['key']} | {j['min']:.4f} | {j['max']:.4f} | "
            f"{j['abs_max']:.4f} | t={j['argmax_tick']}"
            for j in doc["prop_joints"]
        ]
    if doc["charts"]:
        L += ["", "## charts"] + [f"- {c}" for c in doc["charts"]]
    if doc["limitations"]:
        L += ["", "## limitations"] + [f"- {x}" for x in doc["limitations"]]
    return "\n".join(L) + "\n"
