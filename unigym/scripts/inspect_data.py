"""Data quality inspection: fields/shapes/dtype/NaN/distribution stats.

Usage:
    python inspect_data.py <run_dir>          # e.g. data_out/<run>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _ns_keys(data: dict, leaf: str) -> list[str]:
    """All ``articulation/<name>/<leaf>`` keys of a loaded run (sorted)."""
    suffix = "/" + leaf
    return sorted(k for k in data if k.startswith("articulation/") and k.endswith(suffix))


def _single_key(data: dict, leaf: str, required: bool = False):
    keys = _ns_keys(data, leaf)
    if len(keys) == 1:
        return keys[0]
    if required:
        raise SystemExit(
            f"inspect_data: expected one articulation/<name>/{leaf} column, "
            f"found {keys or 'none'} (multi-articulation inspection is not "
            f"supported by this single-vector report)"
        )
    return keys[0] if keys else None


def _articulation_orders(meta: dict) -> dict:
    """``{name: {joint_names, foot_bodies, ...}}`` from the v4 meta record."""
    arts = meta.get("articulations")
    if isinstance(arts, dict):
        return arts
    return {}


def _state_keys_by_articulation(data: dict) -> dict[str, dict[str, str]]:
    """``{name: {leaf: full recorded key}}`` for every articulation column."""
    out: dict[str, dict[str, str]] = {}
    for key in data:
        if not (isinstance(key, str) and key.startswith("articulation/") and key.count("/") >= 2):
            continue
        _prefix, name, leaf = key.split("/", 2)
        out.setdefault(name, {})[leaf] = key
    return out


def load_run(run_dir: str):
    shards = sorted(glob.glob(os.path.join(run_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz under {run_dir}")
    data: dict[str, list[np.ndarray]] = {}
    for p in shards:
        with np.load(p) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in data.items()}, shards


def ascii_hist(x: np.ndarray, bins: np.ndarray, width: int = 48) -> str:
    h, _ = np.histogram(x, bins=bins)
    hmax = max(1, h.max())
    lines = []
    for i in range(len(h)):
        bar = "#" * int(round(width * h[i] / hmax))
        lines.append(f"  [{bins[i]:+5.2f},{bins[i + 1]:+5.2f}) {h[i]:9d} {bar}")
    return "\n".join(lines)


def _print_base_velocity(data, state_keys, steady, steady_after_s):
    """Base height / velocity stats for every articulation."""
    for name in sorted(state_keys):
        keys = state_keys[name]
        base_pos_key = keys.get("base_pos")
        if base_pos_key is None:
            continue
        print(f"\n== base / velocity [{name}] ==")
        bz = data[base_pos_key][..., 2].reshape(-1)
        print(
            f"base_z    mean={bz.mean():.3f} std={bz.std():.3f} "
            f"min={bz.min():.3f} max={bz.max():.3f}"
        )
        if steady.any():
            print(
                f"base_z    steady(t>{steady_after_s}s) "
                f"mean={bz[steady.reshape(-1)].mean():.3f} "
                f"std={bz[steady.reshape(-1)].std():.3f}"
            )
        print(f"fallen(base_z<0.15) fraction: {(bz < 0.15).mean():.4f}")
        base_lin_key = keys.get("base_lin_vel")
        if base_lin_key is None:
            continue
        vx = data[base_lin_key][..., 0]
        v_flat = vx.reshape(-1)
        print(
            f"vx        p5/p50/p95 = "
            f"{np.percentile(v_flat, 5):+.3f}/"
            f"{np.percentile(v_flat, 50):+.3f}/"
            f"{np.percentile(v_flat, 95):+.3f} "
            f"mean={v_flat.mean():+.3f}"
        )
        if steady.any():
            vs = vx[steady].reshape(-1)
            print(
                f"vx        steady p5/p50/p95 = "
                f"{np.percentile(vs, 5):+.3f}/"
                f"{np.percentile(vs, 50):+.3f}/"
                f"{np.percentile(vs, 95):+.3f} "
                f"mean={vs.mean():+.3f}"
            )


def _print_command_tracking(data, state_keys):
    """Command-channel bins vs each articulation's actual vx."""
    if "command" not in data:
        return
    cvx = data["command"][..., 0].reshape(-1)
    print("\n== command tracking (cmd_vx bin -> actual vx mean/std/count) ==")
    edges = np.arange(-1.4, 1.41, 0.2)
    for name in sorted(state_keys):
        lin = state_keys[name].get("base_lin_vel")
        if lin is None:
            continue
        v_flat = data[lin][..., 0].reshape(-1)
        for i in range(len(edges) - 1):
            m = (cvx >= edges[i]) & (cvx < edges[i + 1])
            if m.sum() > 0:
                print(
                    f"  [{name}] [{edges[i]:+5.2f},"
                    f"{edges[i + 1]:+5.2f}) n={m.sum():9d} "
                    f"actual_vx={v_flat[m].mean():+.3f} "
                    f"+- {v_flat[m].std():.3f}"
                )
    print(
        f"cmd_vx coverage: p5/p50/p95 = "
        f"{np.percentile(cvx, 5):+.3f}/{np.percentile(cvx, 50):+.3f}/"
        f"{np.percentile(cvx, 95):+.3f}"
    )


def _print_joint_ranges(data, state_keys, orders):
    """Per-articulation joint ranges (policy order names)."""
    for name in sorted(state_keys):
        jp_key = state_keys[name].get("joint_pos")
        if jp_key is None:
            continue
        jp = data[jp_key].reshape(-1, data[jp_key].shape[-1])
        lo, hi = jp.min(axis=0), jp.max(axis=0)
        names = orders.get(name, {}).get("joint_names") or [f"j{i}" for i in range(jp.shape[1])]
        print(f"\n== {jp_key} range per joint (policy order) ==")
        for i, n in enumerate(names):
            print(f"  {n:18s} [{lo[i]:+7.3f}, {hi[i]:+7.3f}]")


def _print_episodes(data, T, N, tie):
    """Episode counts, durations, and per-env monotonicity."""
    if "episode_id" not in data:
        return
    eid = data["episode_id"].reshape(T, N)
    print(f"\nepisodes : {int(eid.max()) + 1} distinct ids " "(per-env counter)")
    if "time_in_episode" not in data:
        return
    n_ep = int(eid.max()) + 1
    dur = np.zeros((N, n_ep), dtype=np.float64)
    np.maximum.at(dur, (np.broadcast_to(np.arange(N), eid.shape), eid), tie)
    dur = dur[dur > 0]
    print(
        f"completed episodes: {dur.size}; duration min/mean/max = "
        f"{dur.min():.2f}/{dur.mean():.2f}/{dur.max():.2f} s"
    )
    lo = np.floor(dur.min())
    hi = np.ceil(dur.max())
    nb = min(24, max(8, int(hi - lo) * 2))
    print("episode duration histogram (s):")
    print(ascii_hist(dur, np.linspace(lo, hi, nb + 1)))
    mono = all((np.diff(eid[:, n]) >= 0).all() for n in range(N))
    print(f"episode_id monotonic per env: {mono}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--steady_after_s", type=float, default=2.0)
    args = parser.parse_args()

    data, shards = load_run(args.run_dir)
    meta_path = os.path.join(args.run_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    print(f"run dir : {args.run_dir}")
    print(f"shards  : {len(shards)}")
    orders = _articulation_orders(meta)
    state_keys = _state_keys_by_articulation(data)
    if meta:
        print(
            f"meta    : num_envs={meta.get('num_envs')} steps={meta.get('steps')} "
            f"policy_dt={meta.get('policy_dt')} episode_len={meta.get('episode_length_s')}s "
            f"schema_version={meta.get('schema_version')}"
        )
        names = sorted(set(orders) | set(state_keys))
        for name in names:
            record = orders.get(name) or {}
            print(f"articulation {name}:")
            print(f"  joint_names = {record.get('joint_names')}")
            print(f"  foot_bodies = {record.get('foot_bodies')}")
            print(f"  state_keys  = {sorted(state_keys.get(name, {}))}")

    # ---- field table + NaN/Inf check ----
    print("\n== fields ==")
    bad = []
    for k in sorted(data):
        a = data[k]
        nan = inf = 0
        if a.dtype.kind == "f":
            nan, inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
        flag = " <-- NaN!" if nan else ("" if inf == 0 else " <-- Inf!")
        if nan or inf:
            bad.append(k)
        print(f"  {k:24s} {str(a.shape):22s} {str(a.dtype):8s} nan={nan} inf={inf}{flag}")

    # per-articulation state keys (no single/_unique projection)
    if not state_keys:
        base_keys = _ns_keys(data, "base_pos")
        if not base_keys:
            raise SystemExit(
                "inspect_data: expected articulation/<name>/base_pos " "column(s), found none"
            )
    first_base = _single_key(data, "base_pos")
    if first_base is None:
        raise SystemExit("inspect_data: no articulation/<name>/base_pos column")
    T, N = data[first_base].shape[:2]
    print(f"\nrecords : T={T} steps x N={N} envs = {T * N}")
    tie = data["time_in_episode"].reshape(T, N)
    steady = tie > args.steady_after_s

    _print_base_velocity(data, state_keys, steady, args.steady_after_s)
    _print_command_tracking(data, state_keys)
    _print_joint_ranges(data, state_keys, orders)

    _print_episodes(data, T, N, tie)

    if bad:
        print(f"\n[FAIL] NaN/Inf in fields: {bad}")
        raise SystemExit(1)
    print("\n[OK] no NaN/Inf detected")


if __name__ == "__main__":
    main()
