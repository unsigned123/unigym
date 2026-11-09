"""SyncRunner: backend-agnostic, synchronous rollout skeleton (B0/R2).

Extracted from collect.py's main loop.  Per policy step the runner runs
``act -> postprocess -> env.step -> sampler.update -> compute_obs`` and the
unified termination check, then hands a *bundle* to the consumer-side sink:

    sink.on_step(t, bundle)                       # record (pre-reset data side)
    [due mask = clock | reset policy]             #
    sink.on_episode_end(due, bundle)              # BEFORE env.reset_envs
    if auto_reset:                                # Gym Env sets this False
        env.reset_envs(due); adapter.on_reset(due)  # reset chain (fixed order)
        sampler.reset(due)                          #
    sink.on_step_end(t, bundle)                   # post-reset per-step metrics
    ... loop ...
    sink.on_run_end() -> dict                     # final flush + summary

The runner owns control flow only: recording, filtering, video, logging and
statistics all live in the sink (docs/contracts/runner_contract.md).  terminated/alive
are computed here (``TerminationState.from_extra`` + ``term_group.check``) —
pure tensor ops, no RNG consumption, so moving them relative to collect.py's
inline computation cannot perturb bit-exact rollouts.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from unigym.runner.base import StepResult
from unigym.runtime.bridge import RewindBoundary
from unigym.terminations import TerminationState


def _adapter_groups(instance):
    """The control groups the instance's adapter owns (build-time partition).

    The run's build step derives ``env.adapter_groups`` from
    ``RunSpec.controls``; this helper reads that partition for the run's
    adapter.  A run with no controls
    (test doubles) keeps the historical no-validation path returning
    ``None``.  No joint-order / action-face inference is performed.
    """
    env = getattr(instance, "env", None)
    controls = getattr(env, "controls", None)
    if not controls:
        return None
    adapter = getattr(instance, "adapter", None)
    name = (getattr(adapter, "_unigym_name", None)
            or getattr(getattr(instance, "spec", None), "adapter_name", None))
    groups = getattr(env, "adapter_groups", None)
    if groups is not None:
        if name in groups:
            return tuple(groups[name])
        raise ValueError(
            f"adapter {name!r} has no entry in env.adapter_groups "
            f"(declared adapters: {sorted(groups)})")
    if not name:
        # test doubles / direct construction: the body control faces are a
        # cardinality fact, not a guess.
        owned = tuple(c.action_face for c in controls.values()
                      if c.action_face == c.articulation)
        if owned:
            return owned
        raise ValueError(
            "cannot resolve the adapter's control groups: no registry "
            "name on the adapter and no env.adapter_groups partition")
    owned = tuple(c.name for c in controls.values() if c.adapter == name)
    if not owned:
        raise ValueError(
            f"adapter {name!r} owns no control group in env.controls; "
            f"declared groups: {[(c.name, c.adapter) for c in controls.values()]}")
    return owned


def _joint_limits_by_articulation(env):
    """``{articulation: (lower, upper)}`` for joint-limit terminations.

    Prefers the backend's per-articulation mapping (v103.6; no first
    binding).  A flat ``policy_joint_pos_limits`` is accepted ONLY when the
    env proves exactly one policy controller -- the documented cardinality
    convenience, never a primary fallback.
    """
    mapping = getattr(env, "joint_pos_limits_by_articulation", None)
    if mapping is not None:
        return dict(mapping)
    return None


class SyncRunner:
    """Drives the act/step/record/reset loop against any EnvBackend.

    Args:
        cfg: full run config dict (``collect.steps`` supplies the rollout
            length).
        profile: robot profile; unused by the collect loop itself, reserved
            for task-layer extensions (benchmarks, B1+).
        adapter: Adapter (act / postprocess_action / compute_obs /
            on_reset; ``generates_command``/``last_command`` honoured).
        sampler: command sampler or None (self-commanding adapters).
        term_group: TerminationGroup used for the unified terminated/alive
            computation (detection + optional reset policy below).
        env: EnvBackend (env/backends/base.py contract).
        sink: consumer object (see runner/sinks/ and docs/contracts/runner_contract.md).
        reset_on_termination: when True, envs whose termination fired are
            merged into the due mask and reset the same step.  Default False
            reproduces collection mode (episodes end by duration only).
        reset_mask_fn: optional callable ``bundle -> LongTensor(env_ids)``
            fully replacing the terminated-rows term of the reset policy
            (future reset strategies, e.g. graded failure).  Default None.
        replay: optional trace replay source (``unigym.trace.replay.
            TraceReplay``, contract docs/contracts/trace_contract.md §7).
            Two injection points per tick: obs mode feeds the recorded
            policy frame BEFORE ``adapter.act``; action mode replaces the
            adapter's raw output with the recorded raw AFTER ``act`` and
            BEFORE ``postprocess_action`` (design §3.2).  When the trace
            is exhausted the loop STOPS (declared semantics, §7.3).
            Default None = plain rollout, bit-exact unchanged.
        auto_reset: when False the tick still reports ``reset_env_ids`` (the
            due mask) and still calls ``sink.on_episode_end`` for recording,
            but skips the ``env.reset_envs``/``adapter.on_reset``/
            ``sampler.reset`` chain -- the classic single-env ``gym.Env``
            leaves the terminal state in place until the caller resets.
            Default True keeps the three collect/benchmark/both modes
            bit-identical.

    Rewind seam (Round B B1): ``_tick`` is the
    loop counter as INSTANCE state.  The rewind controller sets it back to
    ``restored_tick + 1`` and arms a one-shot :class:`RewindBoundary`
    raise inside ``bridge.tick``; the loop catches it around
    ``sink.on_step``, restores the snapshot's iteration state (obs frame /
    command / torch RNG) and continues from the restored timeline — the
    abandoned branch's episode-end and reset chain never run.  Sinks are
    notified via the duck-typed ``on_rewind(info)`` (episode buffers open
    at the rewind mark themselves ``rewound``).
    """

    @classmethod
    def from_instance(cls, instance) -> SyncRunner:
        """Build a SyncRunner from a live :class:`RunInstance`.

        The constructor arguments are exactly the ones ``collect_cli`` used
        to pass inline: the run cfg, the instance's profile/adapter/sampler/
        term group/env/sink/replay and its reset_on_termination flag.  The
        reserved ``spec.runner.reset_mask_fn`` injection seam is forwarded
        as-is (absent -> None).
        """
        spec = instance.spec
        runner_cfg = spec.runner or {}
        auto_reset = getattr(instance, "auto_reset", None)
        if auto_reset is None:
            auto_reset = runner_cfg.get("auto_reset", True)
        return cls(
            spec.cfg,
            instance.profile,
            instance.adapter,
            instance.sampler,
            instance.term_group,
            instance.env,
            instance.sink,
            reset_on_termination=instance.reset_on_termination,
            reset_mask_fn=runner_cfg.get("reset_mask_fn"),
            replay=instance.replay,
            auto_reset=auto_reset,
            adapter_groups=_adapter_groups(instance),
        )

    def __init__(self, cfg, profile, adapter, sampler, term_group, env, sink,
                 reset_on_termination: bool = False, reset_mask_fn=None,
                 replay=None, auto_reset: bool = True,
                 adapter_groups=None):
        self.cfg = cfg
        self.profile = profile
        self.adapter = adapter
        self.sampler = sampler
        self.term_group = term_group
        self.env = env
        self.sink = sink
        self.reset_on_termination = bool(reset_on_termination)
        self.reset_mask_fn = reset_mask_fn
        self.replay = replay
        self.auto_reset = bool(auto_reset)
        # v104.3: the build-time controls[] partition names exactly the
        # control groups this adapter owns.  None = no control surface
        # (test doubles / direct construction) -> historical behavior.
        self.adapter_groups = (None if adapter_groups is None
                               else tuple(adapter_groups))
        self.adapter_name = (getattr(adapter, "_unigym_name", None)
                             or type(adapter).__name__)
        self.controls = dict(getattr(env, "controls", None) or {})
        self.total_steps = int(cfg["collect"]["steps"])
        # rewind seam: the resumable loop counter + the snapshot's
        # iteration state, both written only by the rewind controller
        self._tick = 0
        self._rewind_state: dict | None = None
        # per-tick iteration state that must survive across tick() calls
        # (the former run()-local ``cmd`` / ``obs_frame`` pair)
        self._started = False
        self._cmd = None
        self._obs_frame = None
        # joint-limit terminations consume a per-articulation limits mapping
        self._term_joint_limits = None
        if any(getattr(t, "requires_joint_limits", False)
                for t in term_group.terminations):
            self._term_joint_limits = _joint_limits_by_articulation(env)

    def _notify_rewind(self, info: dict) -> None:
        """Fan ``on_rewind`` out to every sink that declares it."""
        fan = getattr(self.sink, "sinks", None)
        targets = list(fan) if fan is not None else [self.sink]
        for s in targets:
            fn = getattr(s, "on_rewind", None)
            if callable(fn):
                fn(info)

    def _validate_targets(self, actions) -> None:
        """Validate one tick's action surface against the control partition.

        Exact group-key equality with the adapter's owned groups (missing
        group or undeclared extra group fails loud), then every entry's
        ``kind`` must equal its control group's configured ``target``.
        No controls (``adapter_groups is None``) keeps the historical
        pass-through for test doubles.
        """
        if self.adapter_groups is None:
            return
        if not isinstance(actions, dict):
            raise TypeError(
                f"adapter {self.adapter_name!r} returned "
                f"{type(actions).__name__}, expected dict[str, NamedAction]")
        owned = set(self.adapter_groups)
        keys = set(actions)
        missing = sorted(owned - keys)
        extra = sorted(keys - owned)
        if missing or extra:
            raise ValueError(
                f"adapter {self.adapter_name!r} action groups mismatch: "
                f"missing={missing}, extra={extra}, "
                f"declared={list(self.adapter_groups)}")
        for face, entry in actions.items():
            control = self.controls.get(face)
            if control is None:
                raise ValueError(
                    f"action face {face!r} is not a declared control "
                    f"(declared controls: {sorted(self.controls)})")
            kind = getattr(entry, "kind", None)
            if kind != control.target:
                from unigym.runtime.actions import LEGAL_ACTION_KINDS

                raise ValueError(
                    f"action face {face!r}: configured target "
                    f"{control.target!r}, actual kind {kind!r}; legal "
                    f"kinds: {list(LEGAL_ACTION_KINDS)}")

    def _start(self) -> None:
        """Reset the run-scoped iteration state (cmd / obs frame)."""
        all_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self._cmd = None if self.sampler is None else self.sampler.reset(all_ids)
        self._obs_frame = None
        self._started = True

    @property
    def ticks_done(self) -> int:
        """Policy ticks completed so far (rewind jumps included)."""
        return int(self._tick)

    def tick(self, actions_override=None) -> StepResult:
        """Advance exactly one policy tick; returns its :class:`StepResult`.

        ``actions_override`` is the additive external-driver hook:
        when it is a ``dict[str, NamedAction]`` the adapter forward and its
        postprocess are skipped and the provided actions go straight to
        ``env.step``.  Every other line (obs refresh, termination, sink
        callbacks, reset chain, RNG order) is shared verbatim with the
        adapter-driven path; ``None`` reproduces that path exactly.
        """
        if not self._started:
            self._start()
        adapter, sampler, env = self.adapter, self.sampler, self.env
        device = env.device
        policy_dt = adapter.control_dt
        generates_command = bool(getattr(adapter, "generates_command", False))
        cmd = self._cmd
        obs_frame = self._obs_frame

        t = int(self._tick)
        if actions_override is None:
            # 1) policy forward (history-managing adapters use their internal
            # history; the first step has zero history)
            # replay obs mode: the recorded policy frame feeds act (§7.2)
            if self.replay is not None:
                injected = self.replay.obs_for(t, obs_frame)
                if injected is not None:
                    obs_frame = injected
            raw = adapter.act(obs_frame)
            # replay action mode: recorded raw replaces the adapter output
            # BEFORE postprocess (§7.2 injection point; None = pre-roll,
            # the live action runs)
            if self.replay is not None:
                injected_raw = self.replay.raw_for(t, raw)
                if injected_raw is not None:
                    raw = injected_raw
            # 2) action -> named surface dict[str, NamedAction] (by name)
            targets = adapter.postprocess_action(raw)
        else:
            # external driver: no policy forward, no postprocess
            raw = None
            targets = actions_override
        # 3) action validation: exactly the adapter's owned control groups
        # (build-time controls[] partition) and each entry's kind must match
        # the group's configured target.
        self._validate_targets(targets)
        # 4) physics (interval DR events applied internally)
        obs, extra = env.step(targets, raw_action=raw, cmd=cmd)
        # 5) command resample (self-commanding adapter: cmd stays None)
        if sampler is not None:
            cmd = sampler.update(policy_dt, cmd)
        # 6) obs frame (embeds this step's action, rolling history)
        obs_frame = adapter.compute_obs(env.obs_cache, cmd, env)
        # dataset-facing frame (default returns obs_frame itself: the
        # recorded "obs" stays the policy frame unless an adapter
        # overrides compute_obs_collect)
        obs_collect = adapter.compute_obs_collect(
            obs_frame, env.obs_cache, cmd, env)
        # effective recorded/language command: sampler output, or the
        # adapter's self-generated cascade command (available after act)
        eff_cmd = cmd
        if eff_cmd is None and generates_command:
            eff_cmd = adapter.last_command
        # unified termination check (pure tensor ops, no RNG consumption)
        term_state = TerminationState.from_extra(
            extra, joint_pos_limits=self._term_joint_limits)
        terminated = self.term_group.check(term_state)
        alive = (~terminated) & torch.isfinite(obs_frame).all(dim=-1)
        term_metrics = {"terminated": float(terminated.float().mean()),
                        "fallen": float(terminated.float().mean())}
        if len(self.term_group) > 1:
            for k, m in self.term_group.check_detail(term_state).items():
                term_metrics[f"term_{k}"] = float(m.float().mean())
        bundle = {
            "t": t,
            "raw": raw,
            "targets": targets,
            "obs_frame": obs_frame,
            "obs_collect": obs_collect,
            "extra": extra,
            "cmd": eff_cmd,
            "terminated": terminated,
            "alive": alive,
            "term_state": term_state,
            "term_metrics": term_metrics,
            "env": env,
        }
        try:
            self.sink.on_step(t, bundle)
        except RewindBoundary as rb:
            # rewind boundary (Round B B1): the rest of this iteration
            # (episode-end + reset chain) belongs to the abandoned
            # timeline branch -- drop it, adopt the snapshot's
            # iteration state and continue from the restored tick.
            self._tick = int(rb.next_tick)
            st, self._rewind_state = self._rewind_state, None
            if st is not None:
                obs_frame = st.get("obs_frame")
                cmd = st.get("cmd")
            self._obs_frame = obs_frame
            self._cmd = cmd
            self._notify_rewind(dict(getattr(rb, "info", {}) or {}))
            return StepResult(
                t=t, obs=obs, extra=extra,
                reset_env_ids=torch.zeros(0, dtype=torch.int64,
                                          device=device),
                terminated=terminated)
        self._tick = t + 1
        # reset policy: episode clock due | (terminated | custom mask)
        due = extra["reset_env_ids"]
        added = None
        if self.reset_mask_fn is not None:
            added = torch.as_tensor(self.reset_mask_fn(bundle),
                                    dtype=torch.int64, device=device)
        elif self.reset_on_termination:
            added = torch.nonzero(terminated).squeeze(-1).to(torch.int64)
        if added is not None and added.numel() > 0:
            due = torch.unique(torch.cat([due.to(torch.int64), added]))
        if due.numel() > 0:
            # record-before-reset: episode-end data side happens BEFORE
            # the reset chain (sink flushes episodes/videos/counts here)
            self.sink.on_episode_end(due, bundle)
            if self.auto_reset:
                env.reset_envs(due)
                adapter.on_reset(due)
                if sampler is not None:
                    cmd = sampler.reset(due)
        self.sink.on_step_end(t, bundle)
        self._obs_frame = obs_frame
        self._cmd = cmd
        return StepResult(t=t, obs=obs, extra=extra,
                          reset_env_ids=due, terminated=terminated)

    def run(self, stop_when: Callable[[], bool] | None = None) -> dict:
        """Run the rollout; returns the sink's summary dict.

        Args:
            stop_when: optional predicate checked at the top of every step;
                when True the loop ends early and the normal ``on_run_end``
                teardown runs (B1: BenchmarkSink's min_episodes gate).
                Default None reproduces the fixed-length rollout bitwise.
        """
        self._start()
        while int(self._tick) < self.total_steps:
            if stop_when is not None and stop_when():
                break
            # declared-exhaustion stop: an adapter may expose
            # actions_exhausted() when it serves a finite action stream
            # (shard playback) -- the run ends HONESTLY when the actions
            # run dry: no hold-last, the normal on_run_end teardown runs
            exhausted = getattr(self.adapter, "actions_exhausted", None)
            if callable(exhausted) and exhausted():
                break
            if self.replay is not None and not self.replay.has_tick(
                    int(self._tick)):
                break                       # trace exhausted (§7.3: stop)
            self.tick()

        return self.sink.on_run_end()

    def reset(self, env_ids=None) -> StepResult:
        """Explicit driver reset through the same fixed reset chain.

        ``env_ids=None`` resets every env.  This is the protocol's external
        reset entry point (R3 drivers); the collect loop never calls it and
        the tick-order/RNG path is untouched.  The returned StepResult has
        empty obs/extra because no policy tick was stepped.
        """
        env = self.env
        fresh = not self._started
        if fresh:
            self._start()
        device = env.device
        if env_ids is None:
            due = torch.arange(env.num_envs, dtype=torch.int64, device=device)
        else:
            due = torch.as_tensor(env_ids, dtype=torch.int64, device=device)
        if due.numel() > 0:
            env.reset_envs(due)
            self.adapter.on_reset(due)
            if self.sampler is not None and not (fresh and env_ids is None):
                self._cmd = self.sampler.reset(due)
        return StepResult(t=int(self._tick), obs={}, extra={},
                          reset_env_ids=due, terminated=None)

    def close(self) -> None:
        """No-op: the synchronous runner owns no loop/thread to stop."""
