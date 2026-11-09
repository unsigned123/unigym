"""Unified logging and statistics management for unigym runs.

Replaces the scattered ``print`` calls and hand-written ``summary.json`` in
``collect.py`` with a single :class:`DataCollectLogger` that:

* accumulates per-step metrics via :class:`StatsCollector` (sliding window +
  full-run percentiles),
* prints progress at a configurable interval using window means,
* tracks per-episode durations and filter accept/reject counts,
* writes a rich ``summary.json`` at the end (config snapshot + git hash +
  full statistics + filter stats + throughput).

Yaml usage (indirectly, via ``collect.py``)::

    logger:
      log_interval: 500
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from .stats import StatsCollector

__all__ = ["DataCollectLogger", "StatsCollector"]

_GIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_hash(cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:  # best effort: git hash is cosmetic ("unknown" fallback)
        pass
    return "unknown"


class DataCollectLogger:
    """Unified logging and statistics manager.

    Parameters
    ----------
    config
        Full run config dict (written into summary as a snapshot).
    output_dir
        Directory where ``summary.json`` will be written.
    num_envs
        Number of parallel environments.
    total_steps
        Total planned policy steps (for progress formatting).
    policy_dt
        Policy control timestep (seconds).
    log_interval
        Print progress every *log_interval* steps.
    dist_keys
        Keys for which p5/p50/p95 distributions are computed. ``terminated``
        (per-step fraction of envs firing the configured termination group,
        computed by collect.py) is the P10 primary and feeds
        ``fallen_fraction``; the bare ``fallen`` key is the fallback, and when
        neither is present the env-mean-height fallback applies.
    """

    def __init__(
        self,
        config: dict,
        output_dir: str,
        num_envs: int,
        total_steps: int = 0,
        policy_dt: float = 0.02,
        log_interval: int = 500,
        dist_keys: tuple[str, ...] = (
            "vx",
            "base_z",
            "alive",
            "fallen",
            "terminated",
            "in_recovery",
        ),
    ):
        self.config = config
        self.output_dir = output_dir
        self.num_envs = num_envs
        self._total_steps = total_steps
        self._policy_dt = policy_dt
        self._log_interval = max(1, int(log_interval))
        self._stats = StatsCollector(window_size=log_interval, dist_keys=dist_keys)
        self._episode_durations: list[float] = []
        self._episode_snapshots: list[dict] = []
        self._filter_stats: dict = {
            "total_episodes": 0,
            "kept_episodes": 0,
            "discarded_episodes": 0,
            "discards_by_filter": {},
        }
        self._shard_count = 0
        self._t0 = time.time()
        self._git_hash = _git_hash(_GIT_DIR)

    @property
    def stats(self) -> dict:
        """Current accumulated statistics (for live inspection)."""
        return {
            "window_means": self._stats.window_means,
            "overall_means": self._stats.overall_means,
            "distributions": self._stats.distributions,
        }

    @property
    def filter_stats(self) -> dict:
        return dict(self._filter_stats)

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    # ------------------------------------------------------------------ #
    # per-step logging
    # ------------------------------------------------------------------ #
    def log_step(self, step: int, extra_info: dict) -> None:
        """Record one step's metrics and print progress at *log_interval*.

        ``extra_info`` should contain scalar values (already meaned over envs)
        for keys such as ``vx``, ``base_z``, ``alive``, ``duty_FL`` ...
        plus optional ``shards`` (current shard count).
        """
        metrics = {}
        for k, v in extra_info.items():
            if k == "shards":
                self._shard_count = int(v)
                continue
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                continue
        self._stats.update(metrics)
        if step % self._log_interval == 0 or step == self._total_steps - 1:
            self._print_progress(step)

    def _print_progress(self, step: int) -> None:
        w = self._stats.window_means
        vx = w.get("vx", 0.0)
        bz = w.get("base_z", 0.0)
        alive = w.get("alive", 0.0)
        ep_len = self._mean_recent_episode_steps()
        shards = self._shard_count
        elapsed = time.time() - self._t0
        line = (
            f"[step {step:>6d}/{self._total_steps}] "
            f"vx={vx:+.3f} base_z={bz:.3f} "
            f"alive={alive * 100:.0f}% ep_len={ep_len} "
            f"shards={shards} elapsed={elapsed:.0f}s"
        )
        if "obs_disp_m" in w:
            line += (
                f" obs_disp_m={w['obs_disp_m']:.2f} "
                f"obs_disp_max={w['obs_disp_max']:.2f} "
                f"obs_z_max={w['obs_z_max']:.2f}"
            )
        print(line, flush=True)

    def _mean_recent_episode_steps(self, n: int = 10) -> int:
        if not self._episode_durations:
            return 0
        recent = self._episode_durations[-n:]
        mean_s = float(np.mean(recent))
        return int(round(mean_s / self._policy_dt)) if self._policy_dt > 0 else 0

    # ------------------------------------------------------------------ #
    # per-episode logging
    # ------------------------------------------------------------------ #
    def log_episode(self, episode_info: dict, kept: bool, filter_reason: str | None = None) -> None:
        """Record episode-level statistics at episode end.

        ``episode_info`` should contain at least ``duration_s`` (float).
        """
        dur = float(episode_info.get("duration_s", 0.0))
        self._episode_durations.append(dur)
        self._filter_stats["total_episodes"] += 1
        if kept:
            self._filter_stats["kept_episodes"] += 1
        else:
            self._filter_stats["discarded_episodes"] += 1
            if filter_reason:
                self._filter_stats["discards_by_filter"][filter_reason] = (
                    self._filter_stats["discards_by_filter"].get(filter_reason, 0) + 1
                )
        self._episode_snapshots.append(
            {
                "duration_s": dur,
                "kept": kept,
                "filter_reason": filter_reason,
            }
        )

    # ------------------------------------------------------------------ #
    # per-shard logging
    # ------------------------------------------------------------------ #
    def log_shard(self, shard_idx: int, shard_size: int) -> None:
        self._shard_count = shard_idx + 1

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def write_summary(
        self, total_records: int | None = None, shards: int | None = None, extra: dict | None = None
    ) -> dict:
        """Write ``summary.json`` and return the summary dict."""
        wall = time.time() - self._t0
        dists = self._stats.distributions
        overall = self._stats.overall_means
        bz_raw = self._stats.raw_values("base_z")
        term_raw = self._stats.raw_values("terminated")
        fallen_raw = self._stats.raw_values("fallen")
        # primary (P10): termination-group firing ratio; fallbacks:
        # per-step "fallen" metric, then env-mean base_z below 0.15 m
        fallen_fraction = (
            float(term_raw.mean())
            if term_raw.size
            else (float(fallen_raw.mean()) if fallen_raw.size else None)
        )
        # fallback / reference: env-mean base_z below 0.15 m
        fallen_fraction_height = float((bz_raw < 0.15).mean()) if bz_raw.size else 0.0
        if fallen_fraction is None:
            fallen_fraction = fallen_fraction_height

        # per-termination firing fractions (collect.py feeds "term_<name>"
        # step metrics when the group has >1 terminations)
        term_fractions = {
            k[len("term_") :]: float(v) for k, v in overall.items() if k.startswith("term_")
        }

        # duty metrics are generated per profile foot (collect.py); label sets
        # vary by robot — scan the overall means instead of hardcoding them.
        duty = {k[len("duty_") :]: float(v) for k, v in overall.items() if k.startswith("duty_")}

        durs = (
            np.array(self._episode_durations, dtype=np.float64)
            if self._episode_durations
            else np.array([])
        )
        summary: dict = {
            "framework": "unigym",
            "version": _get_version(),
            "git_hash": self._git_hash,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": self.config.get("run_name"),
            "config": self.config,
            "num_envs": self.num_envs,
            "steps": self._total_steps,
            "duration_s": self._total_steps * self._policy_dt,
            "wall_time_s": wall,
            "steps_per_sec": self._total_steps / wall if wall > 0 else 0,
            "env_steps_per_sec": (self._total_steps * self.num_envs / wall) if wall > 0 else 0,
            "total_records": total_records
            if total_records is not None
            else self._total_steps * self.num_envs,
            "shards": shards if shards is not None else self._shard_count,
            "stats": {
                "vx": dists.get("vx", {}),
                "base_z": {
                    **(dists.get("base_z", {})),
                    "fallen_fraction": fallen_fraction,
                    "fallen_fraction_height_only": fallen_fraction_height,
                },
                "alive": dists.get("alive", {}),
                "fallen": dists.get("fallen", {}),
                "terminated": dists.get("terminated", {}),
                "in_recovery": dists.get("in_recovery", {}),
                "duty_factor": duty,
            },
            "terminations": term_fractions,
            "episodes": {
                "count": len(self._episode_durations),
                "duration_s": {
                    "min": float(durs.min()) if durs.size else None,
                    "max": float(durs.max()) if durs.size else None,
                    "mean": float(durs.mean()) if durs.size else None,
                    "p50": (float(np.percentile(durs, 50)) if durs.size else None),
                },
            },
            "filter_stats": dict(self._filter_stats),
        }
        if extra:
            summary.update(extra)
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary


def _get_version() -> str:
    try:
        from unigym import __version__

        return __version__
    except Exception:  # best effort: __version__ absent outside an install
        return "unknown"
