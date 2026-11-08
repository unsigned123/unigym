"""Per-shard sha256 of rollout arrays; bit-exact comparison between two runs.

Acceptance protocol (deterministic backends, e.g. isaaclab): a refactor must
reproduce identical arrays under the same seed.  meta/summary are exempt
(timestamps/wall-time); this tool hashes the rollout arrays themselves.

Usage:
    python scripts/shard_sha256.py <run_dir>                 # print hashes
    python scripts/shard_sha256.py <dir_a> <dir_b>           # + diff verdict
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

# v103.5 data schema v3: robot state keys are namespaced
# (``articulation/<name>/<leaf>``) and every action face is
# ``action_target/<face>``.  The tool therefore hashes EVERY numeric/string
# column of each shard in sorted-key order (object columns -- language
# strings -- are skipped: their in-memory pointer bytes are not stable).
_SKIP_DTYPE = "O"


def run_hashes(run_dir: Path) -> list[str]:
    shards = sorted(run_dir.glob("shard_*.npz"))
    assert shards, f"no shard_*.npz in {run_dir}"
    lines = []
    for sh in shards:
        data = np.load(sh, allow_pickle=True)
        for key in sorted(data.files):
            arr = np.asarray(data[key])
            if arr.dtype.kind == _SKIP_DTYPE:
                continue
            arr = np.ascontiguousarray(arr)
            lines.append(
                f"{sh.name} {key} {hashlib.sha256(arr.tobytes()).hexdigest()}")
    return lines


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    lines = run_hashes(Path(sys.argv[1]))
    print("\n".join(lines))
    if len(sys.argv) == 3:
        other = run_hashes(Path(sys.argv[2]))
        if lines == other:
            print(f"=== BIT-EXACT MATCH: {len(lines)}/{len(lines)} "
                  f"({Path(sys.argv[1]).name} vs {Path(sys.argv[2]).name}) ===")
        else:
            diff = [f"  A: {a}\n  B: {b}" for a, b in zip(lines, other) if a != b]
            raise SystemExit("=== MISMATCH ===\n" + "\n".join(diff))


if __name__ == "__main__":
    main()
