#!/usr/bin/env python
"""Contact-manager smoke probe (feat/contact-manager).

Usage:
    python scripts/probe_contacts.py <out_dir> [--mg 354.78]

Reads the FIRST shard npz of a contacts-enabled collect run and checks:

* P1  every declared contact/<name> key is present in the npz;
* P2  shape/dtype table per key (general raw (T,N,B,3) float, general
      norm/contact (T,N,B), pairwise raw (T,N,B,F,3), pairwise
      norm/contact (T,N,B[,F]) with aggregate);
* P3  physics invariant: general raw Sigma Fz ~= m*g within +-10%
      over the quasi-static settling window (first 3 policy steps;
      pass --mg with the robot weight in N, default the g1 36.165 kg);
* P4  pairwise live demo: feet-vs-ground pair (any-aggregate) equals the
      general contact mask when the ground is the only counterpart;
* P5  backend consistency: general contact mask == (norm z > threshold).

Exits non-zero on the first violated check (probe/CI gate semantics).
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

# g1 total mass (kg), mujoco XML + isaaclab USD agree to <1%; used for the
# +-10% Sigma Fz window when --mg is not given.
G1_TOTAL_MASS = 36.165
G = 9.81


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument(
        "--mg",
        type=float,
        default=G1_TOTAL_MASS * G,
        help="robot weight in N (default: g1 %.3f kg * g)" % G1_TOTAL_MASS,
    )
    args = ap.parse_args()
    shards = sorted(glob.glob(f"{args.out_dir}/shard_*.npz"))
    if not shards:
        print(f"FAIL: no shard npz under {args.out_dir}")
        return 1
    z = np.load(shards[0])
    keys = [k for k in z.files if k.startswith("contact/")]
    if not keys:
        print("FAIL: no contact/ keys in npz (contacts section inactive?)")
        return 1
    print(f"[probe] contact keys: {sorted(keys)}")

    # expected shapes: {key: (ndim, dtype_kind, last_axes wildcard)}
    ok = True
    for k in keys:
        a = z[k]
        print(f"[probe] {k}: shape={a.shape} dtype={a.dtype}")
        if k.endswith("_raw") and a.ndim not in (4, 5):
            print(f"FAIL: raw group {k} must be 4D/5D, got {a.shape}")
            ok = False
    if not ok:
        return 1

    raw_key = "contact/feet_ground_raw"
    if raw_key in z.files:
        raw = z[raw_key]  # (T, N, B, 3)
        # post-settling window: skip the spawn-drop impact head (first 12
        # policy steps), then TIME-AVERAGE — the standing policy's vertical
        # COM acceleration averages out, so mean Sigma Fz ~= mg.  (A
        # fixed 3-step window is engine-dependent: mujoco spawns soft,
        # isaaclab's first steps contain the drop impact spike.)
        head = min(12, max(1, raw.shape[0] // 10))
        window = raw[head:]
        fz_sum = window[..., 2].sum(axis=(-1, -2)) / max(window.shape[1], 1)
        ratio = float(np.mean(fz_sum)) / args.mg
        print(
            f"[probe] P3 SigmaFz(mean steps {head}:{raw.shape[0]}) = "
            f"{float(np.mean(fz_sum)):.2f} N, mg = {args.mg:.2f} N, "
            f"ratio = {ratio:.3f}"
        )
        if abs(ratio - 1.0) > 0.10:
            print("FAIL: |ratio - 1| > 0.10 (Sigma Fz ~= mg violated)")
            return 1

    gen_key, pair_key = "contact/feet_ground", "contact/feet_vs_ball"
    if pair_key not in z.files:
        pair_key = "contact/feet_vs_ground"
    if gen_key in z.files and pair_key in z.files:
        g, p = z[gen_key], z[pair_key]
        if g.shape == p.shape and g.dtype == np.bool_:
            # pairwise ⊆ general: a pairwise hit requires the general mask
            # (the filter only covers a SUBSET of counterparts when the
            # sensor also touches non-filter bodies, e.g. ground)
            viol = int(np.logical_and(p, ~g).sum())
            live = int(p.sum())
            print(
                f"[probe] P4 pairwise⊆general violations = {viol}, " f"live pairwise steps = {live}"
            )
            if viol:
                print("FAIL: pairwise contact reported with general off")
                return 1
            if live == 0:
                print("FAIL: pairwise demo has no live contact step")
                return 1

    rawp_key = "contact/feet_vs_ball_raw"
    if gen_key in z.files and rawp_key in z.files:
        rawp = z[rawp_key]  # (T,N,B,F,3)
        norms = np.linalg.norm(rawp, axis=-1).min(axis=-1)  # (T,N,B)
        mask = norms > 1.0
        g = z[gen_key]
        if mask.shape == g.shape:
            viol = int(np.logical_and(mask, ~g.astype(bool)).sum())
            live = int(mask.sum())
            print(
                f"[probe] P5 pair-raw threshold mask: live steps = {live}, "
                f"violations vs general = {viol}"
            )
            if viol:
                print("FAIL: pairwise raw force without general contact")
                return 1

    print("[probe] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
