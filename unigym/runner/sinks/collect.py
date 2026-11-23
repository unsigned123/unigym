"""CollectSink: consumer-side behavior moved verbatim from collect.py (B0).

Owns the recording / logging / video side of a collection run:

* per-step rec dict assembly (``_KEY_MAP`` renaming, obs / action_raw /
  action_target / in_recovery / command / language),
* language generation with ``interval_steps`` throttling,
* shard writing (direct ``writer.append`` per policy step),
* recorder frame capture and per-episode video writing,
* DataCollectLogger feeding (log_step / log_episode / write_summary) and
  the n_episodes / step_metrics statistics (duty, term).

Callback contract (docs/contracts/runner_contract.md):

* ``on_step(t, bundle)`` — pre-reset data side; the computation order inside
  matches the original collect.py loop exactly (rec build -> language ->
  writer.append -> camera capture).
* ``on_episode_end(env_ids, bundle)`` — the due branch's data-side part,
  called by the engine BEFORE ``env.reset_envs``: log durations, write the
  episode video, count the episode.  The reset chain itself
  (env.reset_envs / adapter.on_reset / sampler.reset) is driven by the engine.
* ``on_step_end(t, bundle)`` — per-step metrics dict + ``logger.log_step``,
  kept AFTER the reset chain exactly like the original code.
* ``on_run_end()`` — ``writer.finalize``, summary.json; returns the summary
  dict.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence

import numpy as np
import torch

from unigym.runtime.actions import (
    ActionGroupValue,
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
)
from unigym.runtime.controllers import articulation_facts

# extra key -> standardized shard-column key (for the writer).  The
# namespaced form (``articulation/<name>/base_pos_w``) maps by SUFFIX so the
# recorded column keeps the historical leaf name under its namespace.
_KEY_MAP = {
    "base_pos_w": "base_pos",
    "base_lin_vel_b": "base_lin_vel",
    "base_ang_vel_b": "base_ang_vel",
}


def _record_key(key: str) -> str:
    """Apply :data:`_KEY_MAP` to a bare or namespaced extra key."""
    mapped = _KEY_MAP.get(key)
    if mapped is not None:
        return mapped
    head, sep, leaf = key.rpartition("/")
    if sep and leaf in _KEY_MAP:
        return f"{head}/{_KEY_MAP[leaf]}"
    return key


def _fallback_entry_key(entry: NamedAction) -> str:
    """Deterministic per-entry action key (direct-construction fallback).

    The normal path is spec-derived: ``effective_action_groups`` (authored
    groups + backend-declared prop groups) yields the historical
    ``action_target/<group>`` columns.  Only a sink built without that table
    (test doubles / direct construction) falls back to
    ``<articulation>/<comma-joined joint_names>``.  Joint names are identifiers,
    so the comma is a documented separator, not a free-form payload.
    """
    return f"{entry.articulation}/{','.join(entry.joint_names)}"


def _fallback_action_contract(targets: Sequence[NamedAction]) -> list:
    """The per-entry fallback contract (no authoring table available).

    Entries are ordered by ``(articulation, joint_names, kind)`` so sequence
    order is not semantic.  The key is :func:`_fallback_entry_key`; one entry
    key addressed twice is ambiguous and fails loud.
    """
    ordered = sorted(targets, key=lambda na: (na.articulation, na.joint_names, na.kind))
    seen: set[str] = set()
    out: list = []
    for na in ordered:
        key = _fallback_entry_key(na)
        if key in seen:
            raise ValueError(
                f"action entry {key!r} is addressed by more than one NamedAction entry"
            )
        seen.add(key)
        out.append(
            {
                "key": key,
                "column": f"action_target/{key}",
                "articulation": na.articulation,
                "group": key,
                "joint_names": list(na.joint_names),
                "kind": na.kind,
            }
        )
    return out


def action_group_contract(groups: Sequence[OrderedActionGroup]) -> list:
    """The spec+prop-derived action contract (playback v3 meta shape).

    One ``{key, column, articulation, group, joint_names, kind}`` row per
    effective group, in declaration order (authored groups first, then
    backend-declared prop groups).  A group name unique across articulations
    keys/columns by ``<group>``; a shared name is disambiguated as
    ``<articulation>/<group>`` so record columns stay unique.  Duplicate
    ``(articulation, name)`` identities are ambiguous and fail loud.

    The sink writes this into ``meta.json`` (``action_groups``) once at
    construction; it is authoring metadata, not a runtime-entry snapshot.
    """
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.name] = counts.get(group.name, 0) + 1
    seen: set[tuple[str, str]] = set()
    out: list = []
    for group in groups:
        identity = (group.articulation, group.name)
        if identity in seen:
            raise ValueError(f"action group {identity!r} is declared more than once")
        seen.add(identity)
        key = group.name if counts[group.name] == 1 else f"{group.articulation}/{group.name}"
        out.append(
            {
                "key": key,
                "column": f"action_target/{key}",
                "articulation": group.articulation,
                "group": group.name,
                "joint_names": list(group.joints),
                "kind": group.target,
            }
        )
    return out


def _policy_runtimes(env) -> list:
    """Every body control runtime (declaration order, no primary)."""
    return [
        c
        for c in (getattr(env, "controls", {}) or {}).values()
        if getattr(c, "role", None) == "body"
    ]


def _policy_runtime(env):
    """The unique policy runtime (a cardinality fact, not a primary)."""
    runtimes = _policy_runtimes(env)
    if len(runtimes) != 1:
        raise NotImplementedError(
            f"this consumer needs exactly one policy articulation, found "
            f"{len(runtimes)} (per-articulation metrics are V103.6)"
        )
    return runtimes[0]


def _state_key(rt, leaf: str) -> str:
    """Recorded per-articulation state key (``articulation/<articulation>/<leaf>``).

    The namespace is the articulation, never the control-runtime (action-group)
    name: state extra keys are published as ``articulation/<articulation>/...``.
    """
    return f"articulation/{rt.articulation}/{leaf}"


class CollectSink:
    """Shard-writing collection sink (the pre-split collect.py behavior)."""

    def __init__(
        self,
        *,
        writer,
        logger,
        language_gen,
        lang_interval,
        recorder,
        adapter,
        env,
        num_envs,
        total_steps,
        device,
        policy_dt,
        out_dir,
        seed,
        episode_length_s,
        dur_lo,
        dur_hi,
        command_dim,
        action_groups: Sequence[OrderedActionGroup] | None = None,
    ):
        self.writer = writer
        self.logger = logger
        self.language_gen = language_gen
        self.lang_interval = int(lang_interval)
        self.recorder = recorder
        self.adapter = adapter
        self.env = env
        self.num_envs = int(num_envs)
        self.total_steps = int(total_steps)
        self.device = device
        self.policy_dt = policy_dt
        self.out_dir = out_dir
        self.seed = seed
        self.episode_length_s = episode_length_s
        self.dur_lo, self.dur_hi = dur_lo, dur_hi
        self.command_dim = command_dim
        # playback: the spec+prop-derived action contract, fixed at
        # construction.  ``None`` keeps the direct-construction fallback for
        # test doubles; the build path always supplies the effective groups.
        self._effective_groups = None if not action_groups else tuple(action_groups)
        self._action_spec = (
            None if self._effective_groups is None else OrderedActionSpec(self._effective_groups)
        )
        self._action_groups: list | None = (
            None
            if self._effective_groups is None
            else action_group_contract(self._effective_groups)
        )
        if self._action_groups is not None:
            self.writer.meta["action_groups"] = self._action_groups
        # recorder frame buffer (per-episode, for env record_env only)
        self.rec_frames: list[np.ndarray] = []
        self.rec_ep_id = 0
        self.n_episodes = self.num_envs  # initial batch
        self.last_language = [""] * self.num_envs  # cached between lang_interval steps
        self._t0 = time.time()

    # ------------------------------------------------------------------ #
    # per-step recording (pre-reset data side)
    # ------------------------------------------------------------------ #
    def _adopt_projection_groups(self, projection: Sequence[ActionGroupValue]) -> None:
        """Freeze the contract from a runner-supplied projection (fallback path)."""
        groups = tuple(value.group for value in projection)
        self._effective_groups = groups
        self._action_spec = OrderedActionSpec(groups) if groups else None
        self._action_groups = None if not groups else action_group_contract(groups)
        if self._action_groups is not None:
            self.writer.meta["action_groups"] = self._action_groups

    def _record_group_columns(self, rec: dict, projection: Sequence[ActionGroupValue]) -> None:
        """Write one authored column per effective group, in group-joint order."""
        values = {
            (value.group.articulation, value.group.name): value.values for value in projection
        }
        for entry in self._action_groups or ():
            rec[entry["column"]] = values[(entry["articulation"], entry["group"])]

    def _record_fallback_columns(self, rec: dict, targets: Sequence[NamedAction]) -> None:
        """Write deterministic per-entry columns when no authoring table exists."""
        if self._action_groups is None:
            self._action_groups = _fallback_action_contract(targets)
            self.writer.meta["action_groups"] = self._action_groups
        for na in targets:
            rec[f"action_target/{_fallback_entry_key(na)}"] = na.values

    def on_step(self, t: int, bundle: dict) -> None:
        raw = bundle["raw"]
        targets = bundle["targets"]
        obs_collect = bundle["obs_collect"]
        record = bundle["record"]
        eff_cmd = bundle["cmd"]
        # record (data invariant: record before reset)
        rec = {}
        for k, v in record.items():
            if k in ("should_reset", "reset_env_ids"):
                continue
            rec[_record_key(k)] = v
        rec["obs"] = obs_collect
        rec["action_raw"] = raw
        # canonical EPISODE-LOCAL timeline columns (benchmark contract): the first
        # post-step frame of each episode is tick 1 at time policy_dt -- never
        # the global run step, duration_s, or arange*dt starting at 0.
        step_index = bundle.get("step_index")
        time_s = bundle.get("time_s")
        if step_index is not None:
            rec["step_index"] = step_index
        if time_s is not None:
            rec["time_s"] = time_s
        reset_state = bundle.get("reset_state") or {}
        if reset_state:
            rec["reset_index"] = torch.full(
                (self.num_envs,), 0, dtype=torch.int64, device=self.device
            )
            rec["reset_valid"] = torch.tensor(
                [
                    -1
                    if reset_state[i]["reset_valid"] is None
                    else int(reset_state[i]["reset_valid"])
                    for i in range(self.num_envs)
                ],
                dtype=torch.int8,
                device=self.device,
            )
            rec["reset_reason"] = [
                str(reset_state[i]["reset_reason"] or "") for i in range(self.num_envs)
            ]
        # one column per effective group (authored groups + backend-declared
        # prop groups).  Values come from the runner's authoring projection
        # when present, otherwise from the tolerant spec projection of this
        # tick's routed targets.  Sequence order is never semantic.
        projection = bundle.get("action_group_values")
        if self._action_spec is None and projection is not None:
            self._adopt_projection_groups(projection)
        if self._action_spec is not None:
            if projection is None:
                projection = self._action_spec.project_groups(targets)
            self._record_group_columns(rec, projection)
        else:
            self._record_fallback_columns(rec, targets)
        # P11: recovery-switching adapters expose their per-env switch state
        in_recovery = getattr(self.adapter, "in_recovery", None)
        if in_recovery is not None:
            rec["in_recovery"] = in_recovery.detach().clone()
        if eff_cmd is not None:
            rec["command"] = eff_cmd
        # language generation (every step, cached for lang_interval > 1)
        if self.language_gen is not None:
            if t % self.lang_interval == 0 or t == self.total_steps - 1:
                rt = _policy_runtime(self.env)
                lin = record[_state_key(rt, "base_lin_vel_b")]
                ang = record[_state_key(rt, "base_ang_vel_b")]
                pos = record[_state_key(rt, "base_pos_w")]
                lang_extra = {
                    "vx": lin[:, 0],
                    "vy": lin[:, 1],
                    "wz": ang[:, 2],
                    "base_z": pos[:, 2],
                }
                lang_cmd = (
                    eff_cmd
                    if eff_cmd is not None
                    else torch.zeros(
                        self.num_envs,
                        getattr(self.adapter, "self_command_dim", 1),
                        device=self.device,
                    )
                )
                states = self.adapter.build_language_states(lang_extra, lang_cmd)
                self.last_language = self.language_gen.describe(states, lang_cmd)
            rec["language"] = self.last_language
        # determinism-audit hook (shard playback): adapters that carry an
        # expected stream may audit the just-assembled rec -- read-only,
        # raised errors propagate (strict mode fail-fasts the run)
        audit = getattr(self.adapter, "on_recorded_step", None)
        if callable(audit):
            audit(t, rec)
        self.writer.append(rec)
        # camera capture (recorder reads its referenced camera's env slice)
        if self.recorder.enabled and t % self.recorder.record_interval == 0:
            frame = self.recorder.capture_frame()
            if frame is not None:
                self.rec_frames.append(frame)

    # ------------------------------------------------------------------ #
    # episode end (before-reset data side; reset chain is engine-driven)
    # ------------------------------------------------------------------ #
    def on_episode_end(self, env_ids, bundle: dict) -> None:
        record = bundle["record"]
        due_list = env_ids.tolist()
        for ei in due_list:
            dur = float(record["time_in_episode"][ei])
            self.logger.log_episode({"duration_s": dur})
        # recorder: write video if record_env's episode ended
        if self.recorder.enabled and self.recorder.record_env in due_list:
            if self.rec_frames:
                path = os.path.join(
                    self.out_dir, f"video_env{self.recorder.record_env}_ep{self.rec_ep_id}.mp4"
                )
                self.recorder.write_video(
                    self.rec_frames, path, fps=int(round(1.0 / self.policy_dt))
                )
                print(f"[collect] wrote {path} " f"({len(self.rec_frames)} frames)", flush=True)
                self.rec_frames = []
            self.rec_ep_id += 1
        self.n_episodes += env_ids.numel()

    # ------------------------------------------------------------------ #
    # per-step metrics (kept after the reset chain, like the original loop)
    # ------------------------------------------------------------------ #
    def on_step_end(self, t: int, bundle: dict) -> None:
        record = bundle["record"]
        alive = bundle["alive"]
        runtimes = _policy_runtimes(self.env)
        step_metrics = {
            "alive": float(alive.float().mean()),
            **bundle["term_metrics"],
            "shards": self.writer.shard_count,
        }
        for rt in runtimes:
            suffix = "" if len(runtimes) == 1 else f"/{rt.name}"
            step_metrics[f"vx{suffix}"] = float(
                record[_state_key(rt, "base_lin_vel_b")][:, 0].mean()
            )
            step_metrics[f"base_z{suffix}"] = float(
                record[_state_key(rt, "base_pos_w")][:, 2].mean()
            )
        in_recovery = getattr(self.adapter, "in_recovery", None)
        if in_recovery is not None:  # P11: switch-state distribution
            step_metrics["in_recovery"] = float(in_recovery.float().mean())
        self.logger.log_step(t, step_metrics)

    # ------------------------------------------------------------------ #
    # run end: finalize + summary
    # ------------------------------------------------------------------ #
    def on_run_end(self) -> dict:
        # flush remaining recorder frames (incomplete last episode)
        if self.recorder.enabled and self.rec_frames:
            path = os.path.join(
                self.out_dir, f"video_env{self.recorder.record_env}_ep{self.rec_ep_id}.mp4"
            )
            self.recorder.write_video(self.rec_frames, path, fps=int(round(1.0 / self.policy_dt)))
            print(f"[collect] wrote {path} ({len(self.rec_frames)} frames)", flush=True)

        self.writer.finalize()
        wall = time.time() - self._t0
        total_records = self.writer.total_steps * self.num_envs
        summary_extra = {
            "wall_time_s": wall,
            "num_episodes": self.n_episodes,
            "seed": self.seed,
            "policy_dt": self.policy_dt,
            "episode_length_s": self.episode_length_s,
            "episode_length_range": (None if self.dur_lo is None else [self.dur_lo, self.dur_hi]),
            # per-articulation neutral facts (v5; no flat primary order)
            "articulations": {
                rt.articulation: articulation_facts(self.env.articulations[rt.articulation], rt)
                for rt in _policy_runtimes(self.env)
            },
            "command_dim": self.command_dim,
            "language_generator": (
                type(self.language_gen).__name__ if self.language_gen is not None else None
            ),
            "language_interval_steps": self.lang_interval,
            # NOTE: terminations config lives in summary["config"] snapshot;
            # summary["terminations"] is reserved for per-term firing fractions
            "camera_enabled": self.recorder.enabled,
        }
        # P11: recovery-switching statistics (present only for such adapters)
        recovery_stats = getattr(self.adapter, "recovery_stats", None)
        if recovery_stats is not None:
            summary_extra["recovery_stats"] = recovery_stats
        # adapter-provided run report (shard playback's determinism
        # audit lives here): duck-typed dict attribute, absent otherwise
        run_report = getattr(self.adapter, "run_report", None)
        if isinstance(run_report, dict):
            summary_extra["adapter_report"] = run_report
        summary = self.logger.write_summary(
            total_records=total_records,
            shards=self.writer.shard_count,
            extra=summary_extra,
        )
        print(
            "[collect] summary:\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True
        )
        return summary
