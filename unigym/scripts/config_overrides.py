"""CLI config overrides: `-s key=val` dot-path rewrites.

`unigym run/debug/probe CONFIG -s manual.control_dt=0.02` applies
point overrides to the loaded run config BEFORE the engine is built --
root of the sed-surgery friction (campaign x30/x31/x32 were all sed
generated).  Values are type-inferred (int -> float -> bool -> JSON ->
string); the effective diff is printed one line per override and lands
in meta.json (reproducibility).  With NO overrides the config object is
returned untouched (BITWISE gate: byte-identical runs, tests/test_config_overrides.py).
"""
from __future__ import annotations

import json


def parse_value(raw: str):
    """Type inference: int -> float -> bool -> JSON list/object -> str."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw[:1] in ("[", "{"):
        try:
            return json.loads(raw)
        except ValueError:
            pass
    return raw


def parse_set_args(items: list[str] | None) -> list[tuple[str, object]]:
    """`["a.b=1", "c=[1,2]"]` -> `[("a.b", 1), ("c", [1, 2])]` (ordered)."""
    out: list[tuple[str, object]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"-s/--set expects key=value, got {item!r} "
                "(quote JSON values: -s key='[1,2]')")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not key:
            raise ValueError(f"-s/--set: empty key in {item!r}")
        out.append((key, parse_value(raw)))
    return out


def apply_overrides(cfg: dict, overrides: list[tuple[str, object]]) -> tuple:
    """Apply dot-path overrides; returns (cfg, diffs).

    ``diffs`` = [{path, old, new}] (old None = key was absent).  cfg is
    mutated in place ONLY when overrides exist -- an empty list returns
    the same object untouched (BITWISE: no --set -> config bitwise-identical).
    Unknown INTERMEDIATE paths are created as dicts (a missing
    ``benchmark.min_episodes`` is a legitimate new leaf); descending
    through a NON-dict is a hard error (typo guard, fail-fast).
    """
    if not overrides:
        return cfg, []
    diffs: list[dict] = []
    for path, value in overrides:
        node = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = {}
                node[p] = nxt
            if not isinstance(nxt, dict):
                raise ValueError(
                    f"-s {path}: intermediate '{p}' is "
                    f"{type(nxt).__name__}, not a mapping (typo?)")
            node = nxt
        leaf = parts[-1]
        diffs.append({"path": path, "old": node.get(leaf, None),
                      "new": value})
        node[leaf] = value
    return cfg, diffs


def diff_lines(diffs: list[dict]) -> list[str]:
    """One line per override: `override: <path> <old> -> <new>`."""
    def fmt(v):
        return json.dumps(v) if isinstance(v, (dict, list)) else repr(v)
    return [f"override: {d['path']} {fmt(d['old'])} -> {fmt(d['new'])}"
            for d in diffs]
