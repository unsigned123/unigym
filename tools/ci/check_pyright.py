"""Run the pyright gates and ratchets used by CI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path("tools/ci/pyright-baseline.json")


def _pyright_executable() -> str:
    candidate = Path(sys.executable).with_name("pyright")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("pyright")
    if found is None:
        raise SystemExit("pyright is not installed in this environment")
    return found


def _run(config: str) -> dict[str, int]:
    env = os.environ.copy()
    env.setdefault("PYRIGHT_PYTHON_GLOBAL_NODE", "false")
    proc = subprocess.run(
        [_pyright_executable(), "--outputjson", "-p", config],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pyright did not return JSON for {config}")
    summary = payload.get("summary", {})
    return {
        "errors": int(summary.get("errorCount", 0)),
        "warnings": int(summary.get("warningCount", 0)),
        "files": int(summary.get("filesAnalyzed", 0)),
    }


def _format(result: dict[str, int]) -> str:
    counts = f"{result['errors']} errors / {result['warnings']} warnings"
    return f"{counts} ({result['files']} files)"


def main() -> int:
    """Run all configured pyright gates."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard",
        nargs="+",
        default=[],
        help="configs that must have zero errors and zero warnings",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="JSON baseline for ratchet configs",
    )
    args = parser.parse_args()

    failed = False

    for config in args.hard:
        result = _run(config)
        ok = result["errors"] == 0 and result["warnings"] == 0
        print(f"{'PASS' if ok else 'FAIL'} hard {config}: {_format(result)}")
        failed |= not ok

    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        for name, expected in baseline.items():
            config = expected["config"]
            result = _run(config)
            errors_ok = result["errors"] <= int(expected["errors"])
            warnings_ok = result["warnings"] <= int(expected["warnings"])
            ok = errors_ok and warnings_ok
            print(
                f"{'PASS' if ok else 'FAIL'} ratchet {name} ({config}): "
                f"{_format(result)} <= "
                f"{expected['errors']} errors / {expected['warnings']} warnings"
            )
            failed |= not ok
    else:
        print(f"FAIL baseline missing: {args.baseline}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
