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

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.runner.base import StepResult
from unigym.runtime.actions import (
    LEGAL_ACTION_KINDS,
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
    resolve_control_dt,
)
from unigym.runtime.bridge import RewindBoundary
from unigym.terminations import TerminationState


def _joint_target(
    articulation: object, joint: object, value: object, what: str
) -> tuple[tuple[str, str], str]:
    """Validate one adapter-owned ``(articulation, joint) -> kind`` entry."""
    if not isinstance(articulation, str) or not articulation:
        raise ValueError(f"{what} has an empty or non-string articulation: {articulation!r}")
    if not isinstance(joint, str) or not joint:
        raise ValueError(f"{what} has an empty or non-string joint name: {joint!r}")
    if not isinstance(value, str) or value not in LEGAL_ACTION_KINDS:
        raise ValueError(
            f"{what} joint {(articulation, joint)!r} has an illegal target kind "
            f"{value!r}; legal kinds: {list(LEGAL_ACTION_KINDS)}"
        )
    return (articulation, joint), value


def _normalize_joint_targets(targets: object, adapter_name: str) -> dict[tuple[str, str], str]:
    """Normalize one adapter's ``{(articulation, joint): kind}`` partition.

    The partition is authoring/validation metadata: a joint is owned by the
    adapter and expects exactly one target kind.  Duplicate joints would make
    validation ambiguous and fail loud.
    """
    if not isinstance(targets, Mapping):
        raise ValueError(
            f"adapter {adapter_name!r} joint targets must be a mapping of "
            f"(articulation, joint) -> kind, got {type(targets).__name__}"
        )
    out: dict[tuple[str, str], str] = {}
    for raw_key, kind in targets.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ValueError(
                f"adapter {adapter_name!r} joint target key {raw_key!r} must be "
                f"an (articulation, joint) tuple"
            )
        key, target = _joint_target(raw_key[0], raw_key[1], kind, f"adapter {adapter_name!r}")
        if key in out:
            raise ValueError(
                f"adapter {adapter_name!r} declares joint {key!r} more than once; "
                f"adapter joint targets must be unique"
            )
        out[key] = target
    return out


def _controls_joint_targets(controls) -> dict[tuple[str, str], str]:
    """Derive ``{(articulation, joint): kind}`` from physical control rows."""
    targets: dict[tuple[str, str], str] = {}
    for control in controls:
        for joint in control.joint_names:
            key = (str(control.articulation), str(joint))
            if key in targets:
                raise ValueError(
                    f"control rows declare joint {key!r} more than once; joint "
                    f"ownership must be unique"
                )
            targets[key] = str(control.target)
    return targets


def _adapter_joint_targets(instance) -> dict[tuple[str, str], str] | None:
    """The adapter-owned joint -> kind partition (build-time metadata).

    The run's build step derives ``env.adapter_joint_targets`` from
    ``RunSpec.policies`` plus backend-declared prop controls; this helper
    reads that partition for the run's adapter.  A run with no controls (test
    doubles) keeps the historical no-validation path returning ``None``.
    """
    env = getattr(instance, "env", None)
    controls = getattr(env, "controls", None)
    if not controls:
        return None
    adapter = getattr(instance, "adapter", None)
    name = getattr(adapter, "_unigym_name", None) or getattr(
        getattr(instance, "spec", None), "adapter_name", None
    )
    partitions = getattr(env, "adapter_joint_targets", None)
    if partitions is not None:
        if name in partitions:
            return _normalize_joint_targets(partitions[name], str(name))
        raise ValueError(
            f"adapter {name!r} has no entry in env.adapter_joint_targets "
            f"(declared adapters: {sorted(partitions)})"
        )
    if not name:
        # test doubles / direct construction: the body control joints are a
        # cardinality fact, not a guess.
        owned = [c for c in controls.values() if getattr(c, "role", None) == "body"]
        if owned:
            return _controls_joint_targets(owned)
        raise ValueError(
            "cannot resolve the adapter's joint targets: no registry name on "
            "the adapter and no env.adapter_joint_targets partition"
        )
    owned = [c for c in controls.values() if c.adapter == name]
    if not owned:
        raise ValueError(
            f"adapter {name!r} owns no control row in env.controls; declared "
            f"rows: {[(c.name, c.adapter) for c in controls.values()]}"
        )
    return _controls_joint_targets(owned)


def _all_adapter_joint_targets(instance) -> dict[tuple[str, str], str] | None:
    """The union of every adapter's joint -> kind partition.

    :class:`~unigym.runner.external.ExternalRunner` is driven by the caller,
    not by one adapter: its action surface is the whole action spec, so its
    validation partition is the union of every policy-owned joint.
    """
    env = getattr(instance, "env", None)
    partitions = getattr(env, "adapter_joint_targets", None)
    if partitions:
        merged: dict[tuple[str, str], str] = {}
        for adapter_name, entries in partitions.items():
            for key, kind in _normalize_joint_targets(entries, str(adapter_name)).items():
                if key in merged:
                    raise ValueError(
                        f"action joint {key!r} is declared by more than one "
                        f"policy; the external action surface must be unique"
                    )
                merged[key] = kind
        return merged
    controls = getattr(env, "controls", None)
    if not controls:
        return None
    return _controls_joint_targets(controls.values()) or None


def _instance_action_groups(instance) -> tuple[OrderedActionGroup, ...] | None:
    """The run's consumer-side action table (authored groups + props).

    The build step stamps it on the instance and on the env; test doubles may
    omit it entirely (``None`` -> no authoring projection in the bundle).
    """
    groups = getattr(instance, "action_groups", None)
    if groups is None:
        groups = getattr(getattr(instance, "env", None), "effective_action_groups", None)
    if groups is None:
        return None
    return tuple(groups)


def _instance_control_dt(instance) -> float | None:
    """The effective policy period of a live instance (binding beats adapter).

    The run's ``policies`` bindings are the timing authority; the adapter's
    own ``control_dt`` is the explicit fallback when no binding of that
    adapter declares one.  ``None`` means neither source is available (test
    doubles), not a silent default.
    """
    spec = getattr(instance, "spec", None)
    adapter = getattr(instance, "adapter", None)
    fallback = getattr(adapter, "control_dt", None)
    policies = tuple(getattr(spec, "policies", ()) or ())
    name = getattr(adapter, "_unigym_name", None) or getattr(spec, "adapter_name", None)
    if not policies or not name:
        return None if fallback is None else float(fallback)
    return resolve_control_dt(policies, str(name), fallback)


_RESET_SAMPLE_PREFIXES = ("articulation/", "object/", "contact/", "site/", "joints/")


def _reset_sample_arrays(snapshot) -> dict[str, np.ndarray]:
    """Selected pre-policy-step arrays from an obs snapshot.

    Cameras and free-form frames are deliberately excluded (memory): the
    reset sample keeps state-like keys only, never full frames.
    """
    values = getattr(snapshot, "values", None) or {}
    out: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not key.startswith(_RESET_SAMPLE_PREFIXES):
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


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
        control_dt: the effective policy period (s) used to advance the
            command clock.  When set it is authoritative over
            ``adapter.control_dt``; ``None`` falls back to the adapter's own
            value (and a run with a sampler and no period fails loud).  The
            run's ``policies`` bindings supply it through
            :func:`unigym.runtime.actions.resolve_control_dt`.
        action_groups: optional effective action table (authored groups +
            backend-declared prop groups).  When present, every tick exposes
            ``bundle["action_group_values"]`` for sinks/watchers/trace; it is
            consumer metadata, never an action identity or routing input.

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
        declared = getattr(getattr(instance, "env", None), "adapter_groups", None) or {}
        if len(declared) > 1:
            raise ValueError(
                "SyncRunner v0 supports exactly one policy adapter; the run "
                f"declares {sorted(declared)} | hint: bind every action group "
                "to one composite adapter, or drive the run with ExternalRunner"
            )
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
            adapter_joint_targets=_adapter_joint_targets(instance),
            action_groups=_instance_action_groups(instance),
            control_dt=_instance_control_dt(instance),
        )

    def __init__(
        self,
        cfg,
        profile,
        adapter,
        sampler,
        term_group,
        env,
        sink,
        reset_on_termination: bool = False,
        reset_mask_fn=None,
        replay=None,
        auto_reset: bool = True,
        adapter_joint_targets: Mapping[tuple[str, str], str] | None = None,
        action_groups: Sequence[OrderedActionGroup] | None = None,
        control_dt: float | None = None,
    ):
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
        self.adapter_name = getattr(adapter, "_unigym_name", None) or type(adapter).__name__
        resolved_dt = control_dt if control_dt is not None else getattr(adapter, "control_dt", None)
        self.control_dt = None if resolved_dt is None else float(resolved_dt)
        self.controls = dict(getattr(env, "controls", None) or {})
        # v104.3: the build-time policies partition names exactly the
        # (articulation, joint) rows this adapter owns, with the target kind
        # each joint expects.  None = no control surface (test doubles /
        # direct construction) -> historical behavior.
        self.adapter_joint_targets = (
            None
            if adapter_joint_targets is None
            else _normalize_joint_targets(adapter_joint_targets, self.adapter_name)
        )
        # authoring projection for sinks/watchers/trace: the effective action
        # table (authored groups + backend-declared prop groups).  Additive
        # bundle data only -- never an action identity and never backend input.
        self.action_groups = None if not action_groups else tuple(action_groups)
        self._action_spec = (
            None if self.action_groups is None else OrderedActionSpec(self.action_groups)
        )
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
        if any(getattr(t, "requires_joint_limits", False) for t in term_group.terminations):
            self._term_joint_limits = _joint_limits_by_articulation(env)
        # benchmark contract episode facts: per-env reset sample/legality and the
        # run identity (stamped on the env by the run build).
        # test doubles may carry no env surface at all; benchmark facts then
        # stay empty while the historical runner behavior is untouched.
        self._num_envs = int(getattr(env, "num_envs", 0) or 0)
        # per-env EPISODE-LOCAL policy step counter: frame 0 is tick 1 and the
        # tick resets to 0 on that env's reset (never the global run step).
        self._episode_step = [0] * self._num_envs
        self._reset_samples: list[dict[str, np.ndarray] | None] = [None] * self._num_envs
        self._reset_valid: list[bool | None] = [None] * self._num_envs
        self._reset_reason: list[str | None] = [None] * self._num_envs
        self.run_identity = getattr(env, "run_identity", None)

    def _notify_rewind(self, info: dict) -> None:
        """Fan ``on_rewind`` out to every sink that declares it."""
        fan = getattr(self.sink, "sinks", None)
        targets = list(fan) if fan is not None else [self.sink]
        for s in targets:
            fn = getattr(s, "on_rewind", None)
            if callable(fn):
                fn(info)

    def _validate_targets(self, actions) -> None:
        """Validate one tick's action surface against the adapter partition.

        ``actions`` must be a sequence of self-describing
        :class:`~unigym.runtime.actions.NamedAction` entries.  Every addressed
        ``(articulation, joint)`` must be owned by this adapter, carry the
        owned target kind, and be addressed exactly once; every owned joint
        must be covered.  No controls (``adapter_joint_targets is None``)
        keeps the historical pass-through for test doubles.
        """
        partition = self.adapter_joint_targets
        if partition is None:
            return
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise TypeError(
                f"adapter {self.adapter_name!r} returned "
                f"{type(actions).__name__}, expected a sequence of NamedAction"
            )
        addressed: set[tuple[str, str]] = set()
        for entry in actions:
            if not isinstance(entry, NamedAction):
                raise TypeError(
                    f"adapter {self.adapter_name!r} returned a "
                    f"{type(entry).__name__} entry; every action entry must "
                    f"be a NamedAction carrying joint-addressed values"
                )
            for joint in entry.joint_names:
                key = (entry.articulation, joint)
                expected = partition.get(key)
                if expected is None:
                    accepted = sorted(j for a, j in partition if a == entry.articulation)
                    raise ValueError(
                        f"adapter {self.adapter_name!r}: unknown action joint "
                        f"{joint!r} for articulation {entry.articulation!r}; "
                        f"accepted joints: {accepted}"
                    )
                if entry.kind != expected:
                    raise ValueError(
                        f"adapter {self.adapter_name!r}: articulation "
                        f"{entry.articulation!r} joint {joint!r} has action kind "
                        f"{entry.kind!r} but the adapter-owned target kind is "
                        f"{expected!r}; legal kinds: {list(LEGAL_ACTION_KINDS)}"
                    )
                if key in addressed:
                    raise ValueError(
                        f"adapter {self.adapter_name!r}: articulation "
                        f"{entry.articulation!r} joint {joint!r} is addressed "
                        f"more than once in one tick"
                    )
                addressed.add(key)
        missing = sorted(key for key in partition if key not in addressed)
        if missing:
            details = "; ".join(
                f"articulation {articulation!r}: joints "
                f"{[joint for a, joint in missing if a == articulation]}"
                for articulation in sorted({a for a, _ in missing})
            )
            raise ValueError(
                f"adapter {self.adapter_name!r}: adapter-owned joints are "
                f"unaddressed: {details}"
            )

    def _start(self) -> None:
        """Reset the run-scoped iteration state (cmd / obs frame)."""
        all_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self._cmd = None if self.sampler is None else self.sampler.reset(all_ids)
        observations = getattr(self.env, "observations", None)
        if getattr(observations, "command", False):
            self.env.obs.set_command(self._cmd)
        self._obs_frame = None
        self._started = True
        # canonical reset tick 0: the initial spawn state is the reset sample
        # of every env's still-open episode (never recorder frame 0).
        self._record_reset_facts(all_ids.tolist())

    # ------------------------------------------------------------------ #
    # benchmark contract episode facts (reset / termination / timeline)
    # ------------------------------------------------------------------ #
    def _record_reset_facts(self, env_ids) -> None:
        """Capture the post-reset/pre-policy-step sample + legality per env."""
        if self._num_envs <= 0:
            return
        snapshot = self.env.obs.snapshot()
        arrays = _reset_sample_arrays(snapshot)
        outcomes = getattr(self.env, "last_reset_outcomes", None) or {}
        for raw in env_ids:
            ei = int(raw)
            sample = {key: value[ei] for key, value in arrays.items() if ei < len(value)}
            outcome = outcomes.get(ei)
            initializer_sample = getattr(outcome, "sample", None) if outcome is not None else None
            if not sample and initializer_sample is not None:
                sample = {str(k): np.asarray(v) for k, v in initializer_sample.items()}
            self._reset_samples[ei] = sample or None
            self._reset_valid[ei] = None if outcome is None else outcome.valid
            self._reset_reason[ei] = None if outcome is None else outcome.reason

    def _reset_fact(self, ei: int) -> dict:
        # reset_index is the canonical reset tick of the sample (episode
        # relative: always 0, before the first post-step frame tick 1).
        return {
            "reset_index": 0,
            "reset_sample": self._reset_samples[ei],
            "reset_valid": self._reset_valid[ei],
            "reset_reason": self._reset_reason[ei],
        }

    def _termination_facts(self, state, clock_due) -> dict[int, EpisodeTermination]:
        """Per-env core terminations for this tick (term group + deadline).

        Every stamped ``step_index``/``time_s`` is EPISODE-LOCAL (frame 0 is
        tick 1), so it lines up with ``EpisodeData.step_index``.
        """
        by_env: dict[int, list[Termination]] = {}

        def stamp(ei: int) -> dict:
            index = int(self._episode_step[ei])
            time_s = None if self.control_dt is None else float(index) * float(self.control_dt)
            return {"step_index": index, "time_s": time_s}

        detail = self.term_group.check_detail(state) if len(self.term_group) else {}
        for name, mask in detail.items():
            code = f"env.{name}"
            for ei in torch.nonzero(mask).squeeze(-1).tolist():
                by_env.setdefault(int(ei), []).append(
                    Termination(
                        outcome=EpisodeOutcome.FAILURE,
                        code=code,
                        source=TerminationSource.ENVIRONMENT,
                        **stamp(int(ei)),
                    )
                )
        # core engine horizon: the backend's own episode clock fired.
        due = getattr(clock_due, "tolist", lambda: [])()
        for ei in due:
            by_env.setdefault(int(ei), []).append(
                Termination(
                    outcome=EpisodeOutcome.DEADLINE,
                    code="runtime.episode_horizon",
                    source=TerminationSource.RUNTIME,
                    **stamp(int(ei)),
                )
            )
        return {
            ei: EpisodeTermination.from_terminations(terms) for ei, terms in by_env.items() if terms
        }

    @property
    def ticks_done(self) -> int:
        """Policy ticks completed so far (rewind jumps included)."""
        return int(self._tick)

    def tick(self, actions_override=None) -> StepResult:
        """Advance exactly one policy tick; returns its :class:`StepResult`.

        ``actions_override`` is the additive external-driver hook:
        when it is a ``Sequence[NamedAction]`` the adapter forward and its
        postprocess are skipped and the provided actions go straight to
        ``env.step``.  Every other line (obs refresh, termination, sink
        callbacks, reset chain, RNG order) is shared verbatim with the
        adapter-driven path; ``None`` reproduces that path exactly.
        """
        if not self._started:
            self._start()
        adapter, sampler, env = self.adapter, self.sampler, self.env
        device = env.device
        policy_dt = self.control_dt
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
            # 2) action -> self-describing sequence[NamedAction] (by identity)
            targets = adapter.postprocess_action(raw)
        else:
            # external driver: no policy forward, no postprocess
            raw = None
            targets = actions_override
        # 3) action validation: exactly the adapter's owned control groups
        # (build-time controls[] partition) and each entry's kind must match
        # the group's configured target.
        self._validate_targets(targets)
        # 3b) authoring projection for consumers: one value row per effective
        # group, in group-joint order.  Absent table -> None (no projection).
        group_values = (
            None if self._action_spec is None else self._action_spec.project_groups(targets)
        )
        # 4) physics (interval DR events applied internally)
        step = env.step(targets, raw_action=raw, cmd=cmd)
        # episode-local clock: every env advances one policy tick in its OWN
        # episode; a reset below returns that env to tick 0.
        for i in range(self._num_envs):
            self._episode_step[i] += 1
        # 5) command resample (self-commanding adapter: cmd stays None)
        if sampler is not None:
            if policy_dt is None:
                raise RuntimeError(
                    f"adapter {self.adapter_name!r} declares no control_dt and no "
                    f"policy binding sets one; the command clock needs an "
                    f"explicit policy period"
                )
            cmd = sampler.update(policy_dt, cmd)
        # 5b) command/prev_action injection AFTER the sampler update: the
        # observation always carries the post-sampler command, never the
        # pre-step value.
        observations = getattr(env, "observations", None)
        if getattr(observations, "command", False):
            env.obs.set_command(cmd)
        if getattr(observations, "prev_action", False):
            env.obs.set_prev_action(raw)
        obs_snapshot = env.obs.snapshot()
        # 6) obs frame (embeds this step's action, rolling history)
        obs_frame = adapter.compute_obs(obs_snapshot, env)
        # dataset-facing frame (default returns obs_frame itself: the
        # recorded "obs" stays the policy frame unless an adapter
        # overrides compute_obs_collect)
        obs_collect = adapter.compute_obs_collect(obs_frame, obs_snapshot, env)
        # effective recorded/language command: sampler output, or the
        # adapter's self-generated cascade command (available after act)
        eff_cmd = cmd
        if eff_cmd is None and generates_command:
            eff_cmd = adapter.last_command
        # unified termination check (pure tensor ops, no RNG consumption)
        record = step.record
        term_state = TerminationState.from_extra(record, joint_pos_limits=self._term_joint_limits)
        terminated = self.term_group.check(term_state)
        alive = (~terminated) & torch.isfinite(obs_frame).all(dim=-1)
        term_metrics = {
            "terminated": float(terminated.float().mean()),
            "fallen": float(terminated.float().mean()),
        }
        if len(self.term_group) > 1:
            for k, m in self.term_group.check_detail(term_state).items():
                term_metrics[f"term_{k}"] = float(m.float().mean())
        # benchmark contract: structured per-env terminations for this tick
        # (term-group reasons + the core engine horizon).  Core-neutral
        # outcome/source vocabulary; no Task/Gym language here.
        terminations = self._termination_facts(term_state, step.reset_env_ids)
        step_index = torch.tensor(self._episode_step, dtype=torch.int64, device=device)
        time_s = (
            None
            if self.control_dt is None
            else step_index.to(torch.float32) * float(self.control_dt)
        )
        bundle = {
            "t": t,
            "raw": raw,
            "targets": targets,
            "obs": obs_snapshot,
            "obs_frame": obs_frame,
            "obs_collect": obs_collect,
            "record": record,
            "cmd": eff_cmd,
            "action_group_values": group_values,
            "terminated": terminated,
            "alive": alive,
            "term_state": term_state,
            "term_metrics": term_metrics,
            "terminations": terminations,
            "step_index": step_index,
            "time_s": time_s,
            "reset_state": {
                int(ei): {
                    "reset_index": 0,
                    "reset_valid": self._reset_valid[ei],
                    "reset_reason": self._reset_reason[ei],
                }
                for ei in range(self._num_envs)
            },
            "run_identity": self.run_identity,
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
                t=t,
                obs=obs_snapshot,
                record=record,
                reset_env_ids=torch.zeros(0, dtype=torch.int64, device=device),
                terminated=terminated,
            )
        self._tick = t + 1
        # reset policy: episode clock due | (terminated | custom mask)
        due = step.reset_env_ids
        added = None
        if self.reset_mask_fn is not None:
            added = torch.as_tensor(self.reset_mask_fn(bundle), dtype=torch.int64, device=device)
        elif self.reset_on_termination:
            added = torch.nonzero(terminated).squeeze(-1).to(torch.int64)
        if added is not None and added.numel() > 0:
            due = torch.unique(torch.cat([due.to(torch.int64), added]))
        if due.numel() > 0:
            # the episode's START facts (reset sample/legality) ride the
            # episode-end bundle; they were captured at this episode's reset.
            bundle["reset_facts"] = {int(ei): self._reset_fact(int(ei)) for ei in due.tolist()}
            # record-before-reset: episode-end data side happens BEFORE
            # the reset chain (sink flushes episodes/videos/counts here)
            self.sink.on_episode_end(due, bundle)
            if self.auto_reset:
                env.reset_envs(due)
                adapter.on_reset(due)
                if sampler is not None:
                    cmd = sampler.reset(due)
                if getattr(observations, "command", False):
                    env.obs.set_command(cmd)
                if getattr(observations, "prev_action", False):
                    env.obs.set_prev_action(None)
                for raw in due.tolist():
                    self._episode_step[int(raw)] = 0
                # capture the NEXT episode's pre-policy-step reset sample
                self._record_reset_facts(due.tolist())
        self.sink.on_step_end(t, bundle)
        self._obs_frame = obs_frame
        self._cmd = cmd
        return StepResult(
            t=t, obs=obs_snapshot, record=record, reset_env_ids=due, terminated=terminated
        )

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
            if self.replay is not None and not self.replay.has_tick(int(self._tick)):
                break  # trace exhausted (§7.3: stop)
            self.tick()

        return self.sink.on_run_end()

    def reset(self, env_ids=None) -> StepResult:
        """Explicit driver reset through the same fixed reset chain.

        ``env_ids=None`` resets every env.  This is the protocol's external
        reset entry point (R3 drivers); the collect loop never calls it and
        the tick-order/RNG path is untouched.  The returned StepResult has
        empty observation/record because no policy tick was stepped.
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
            observations = getattr(env, "observations", None)
            if getattr(observations, "command", False):
                env.obs.set_command(self._cmd)
            if getattr(observations, "prev_action", False):
                env.obs.set_prev_action(None)
            for raw in due.tolist():
                self._episode_step[int(raw)] = 0
            self._record_reset_facts(due.tolist())
        return StepResult(
            t=int(self._tick),
            obs=env.obs.snapshot(),
            record=env.record(),
            reset_env_ids=due,
            terminated=None,
        )

    def close(self) -> None:
        """No-op: the synchronous runner owns no loop/thread to stop."""
