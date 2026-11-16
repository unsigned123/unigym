"""CollectSink: consumer-side behavior moved verbatim from collect.py (B0).

Owns the recording / logging / video side of a collection run:

* per-step rec dict assembly (``_KEY_MAP`` renaming, obs / action_raw /
  action_target / in_recovery / command / language),
* language generation with ``interval_steps`` throttling,
* shard writing (direct ``writer.append``) or per-env episode buffering
  + filter-chain flush when filters are active,
* recorder frame capture and per-episode video writing,
* DataCollectLogger feeding (log_step / log_episode / write_summary) and
  the n_episodes / step_metrics statistics (duty, term).

Callback contract (docs/contracts/runner_contract.md):

* ``on_step(t, bundle)`` — pre-reset data side; the computation order inside
  matches the original collect.py loop exactly (rec build -> language ->
  filter buffer / writer.append -> camera capture).
* ``on_episode_end(env_ids, bundle)`` — the due branch's data-side part,
  called by the engine BEFORE ``env.reset_envs``: flush episodes (filter
  mode) or log durations, write the episode video, count the episode.
  The reset chain itself (env.reset_envs / adapter.on_reset / sampler.reset)
  is driven by the engine.
* ``on_step_end(t, bundle)`` — per-step metrics dict + ``logger.log_step``,
  kept AFTER the reset chain exactly like the original code.
* ``on_run_end()`` — remaining flushes, ``writer.finalize``, summary.json;
  returns the summary dict.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence

import numpy as np
import torch

from unigym.runtime.actions import NamedAction
from unigym.runtime.controllers import articulation_facts

from ..episodes import _flush_episode, slice_step_for_env

# extra key -> standardized episode_data key (for filters + writer).  The
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


def _canonical_action_entries(targets: Sequence[NamedAction]) -> list[tuple[NamedAction, str, str]]:
    """Canonical ``(entry, record_column, face_key)`` triples, in identity order.

    The action-group contract is addressed by ``(articulation, group)``, so
    the incoming sequence order is NOT semantic: entries are sorted by that
    identity before comparison or recording (a reordered sequence produces
    the identical contract).  A group name shared by two articulations is
    disambiguated with the articulation path so record keys stay unique; a
    duplicate identity is ambiguous and fails loud.
    """
    ordered = sorted(targets, key=lambda na: (na.articulation, na.group))
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for na in ordered:
        identity = (na.articulation, na.group)
        if identity in seen:
            raise ValueError(
                f"action group {identity!r} is addressed by more than one " f"NamedAction entry"
            )
        seen.add(identity)
        counts[na.group] = counts.get(na.group, 0) + 1
    out: list[tuple[NamedAction, str, str]] = []
    for na in ordered:
        if counts[na.group] == 1:
            face_key = na.group
        else:
            face_key = f"{na.articulation}/{na.group}"
        out.append((na, f"action_target/{face_key}", face_key))
    return out


def action_face_contract(targets: Sequence[NamedAction]) -> list:
    """The action-face contract (playback v3) as data.

    One ``{key, column, articulation, group, joint_names, kind}`` entry per
    NamedAction group, canonicalized by ``(articulation, group)`` -- captured
    from the actual per-step targets (zero inference: names and widths come
    from the entries themselves, never from family constants).  Sequence
    order is not semantic and reordering must not change the contract.  Every
    column is ``action_target/<group>`` (``action_target/<articulation>/<group>``
    when the group name is not unique across articulations); there is no
    primary entry.

    The sink writes this into ``meta.json`` (``action_faces``) on its
    first recorded step; shard playback validates every face against
    it and constructs the NamedActions from it.  Old shards without the
    key are rejected explicitly (pre-playback re-record).
    """
    return [
        {
            "key": face_key,
            "column": column,
            "articulation": na.articulation,
            "group": na.group,
            "joint_names": list(na.joint_names),
            "kind": na.kind,
        }
        for na, column, face_key in _canonical_action_entries(targets)
    ]


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
    """Recorded per-articulation state key (``articulation/<name>/<leaf>``)."""
    return f"articulation/{rt.name}/{leaf}"


class CollectSink:
    """Shard-writing collection sink (the pre-split collect.py behavior)."""

    def __init__(
        self,
        *,
        writer,
        logger,
        filter_chain,
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
    ):
        self.writer = writer
        self.logger = logger
        self.filter_chain = filter_chain
        self.filter_active = len(filter_chain) > 0
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
        # playback: the action-face contract, captured ONCE from the
        # actual first-step routing and written into the writer's meta
        # (lands in meta.json with the first flush); None until then
        self._action_faces: list | None = None
        # per-env episode buffer (only when filters active)
        self.ep_bufs: list[list[dict]] = (
            [[] for _ in range(self.num_envs)] if self.filter_active else []
        )
        # recorder frame buffer (per-episode, for env record_env only)
        self.rec_frames: list[np.ndarray] = []
        self.rec_ep_id = 0
        self.n_episodes = self.num_envs  # initial batch
        self.last_language = [""] * self.num_envs  # cached between lang_interval steps
        self._t0 = time.time()

    # ------------------------------------------------------------------ #
    # per-step recording (pre-reset data side)
    # ------------------------------------------------------------------ #
    def on_step(self, t: int, bundle: dict) -> None:
        raw = bundle["raw"]
        targets = bundle["targets"]
        obs_collect = bundle["obs_collect"]
        extra = bundle["extra"]
        eff_cmd = bundle["cmd"]
        # record (data invariant: record before reset)
        rec = {}
        for k, v in extra.items():
            if k in ("should_reset", "reset_env_ids"):
                continue
            rec[_record_key(k)] = v
        rec["obs"] = obs_collect
        rec["action_raw"] = raw
        # one column per NamedAction group: action_target/<group> (v103.5;
        # no primary, no action_target_<group> suffix form).  The canonical
        # (articulation, group) order makes sequence order irrelevant.
        for na, column, _face_key in _canonical_action_entries(targets):
            rec[column] = na.values
        # playback contract (single source of truth on disk): capture
        # the faces from THIS step's actual routing once, then hold every
        # later step to it -- a mid-run face change would make the
        # recorded contract a lie, so it fail-fasts instead
        faces = action_face_contract(targets)
        if self._action_faces is None:
            self._action_faces = faces
            self.writer.meta["action_faces"] = faces
        elif faces != self._action_faces:
            raise ValueError(
                f"action faces changed mid-run: {faces} != "
                f"{self._action_faces} -- the recorded meta contract "
                f"(action_faces) would not describe the shard stream"
            )
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
                lin = extra[_state_key(rt, "base_lin_vel_b")]
                ang = extra[_state_key(rt, "base_ang_vel_b")]
                pos = extra[_state_key(rt, "base_pos_w")]
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
        if self.filter_active:
            # buffer per-env step slices for episode-level filtering
            for ei in range(self.num_envs):
                self.ep_bufs[ei].append(slice_step_for_env(rec, ei))
        else:
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
        extra = bundle["extra"]
        due_list = env_ids.tolist()
        if self.filter_active:
            for ei in due_list:
                _flush_episode(self.ep_bufs[ei], self.filter_chain, self.writer, self.logger)
                self.ep_bufs[ei] = []
        else:
            for ei in due_list:
                dur = float(extra["time_in_episode"][ei])
                self.logger.log_episode({"duration_s": dur}, kept=True)
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
        extra = bundle["extra"]
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
                extra[_state_key(rt, "base_lin_vel_b")][:, 0].mean()
            )
            step_metrics[f"base_z{suffix}"] = float(
                extra[_state_key(rt, "base_pos_w")][:, 2].mean()
            )
        in_recovery = getattr(self.adapter, "in_recovery", None)
        if in_recovery is not None:  # P11: switch-state distribution
            step_metrics["in_recovery"] = float(in_recovery.float().mean())
        self.logger.log_step(t, step_metrics)

    # ------------------------------------------------------------------ #
    # run end: remaining flushes + finalize + summary
    # ------------------------------------------------------------------ #
    def on_run_end(self) -> dict:
        # flush remaining incomplete episodes (filter mode)
        if self.filter_active:
            for ei in range(self.num_envs):
                if self.ep_bufs[ei]:
                    _flush_episode(self.ep_bufs[ei], self.filter_chain, self.writer, self.logger)
                    self.ep_bufs[ei] = []

        # flush remaining recorder frames (incomplete last episode)
        if self.recorder.enabled and self.rec_frames:
            path = os.path.join(
                self.out_dir, f"video_env{self.recorder.record_env}_ep{self.rec_ep_id}.mp4"
            )
            self.recorder.write_video(self.rec_frames, path, fps=int(round(1.0 / self.policy_dt)))
            print(f"[collect] wrote {path} ({len(self.rec_frames)} frames)", flush=True)

        self.writer.finalize()
        wall = time.time() - self._t0
        total_records = (
            self.writer.total_steps
            if self.filter_active
            else self.writer.total_steps * self.num_envs
        )
        summary_extra = {
            "wall_time_s": wall,
            "num_episodes": self.n_episodes,
            "seed": self.seed,
            "policy_dt": self.policy_dt,
            "episode_length_s": self.episode_length_s,
            "episode_length_range": [self.dur_lo, self.dur_hi],
            # per-articulation neutral facts (v4; no flat primary order)
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
