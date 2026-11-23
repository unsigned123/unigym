"""EpisodeEvaluator owns episode facts, task isolation and final termination."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._support.fakes.benchmark_env import FakeBenchmarkEnv
from unigym.benchmarks.base import EpisodeResult, Task
from unigym.benchmarks.episode import EpisodeLimits
from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.env.obs import ObsSnapshot, ObservationSchema, ObservationSpec
from unigym.evaluation import EvalQueryUnavailableError, EvalViewStaleError
from unigym.runner.evaluation.buffer import TakenEpisode
from unigym.runner.evaluation.scoring import EpisodeEvaluator

_ARTICULATIONS = {"g1": {}}


def _step(episode_id: int, time_s: float) -> dict:
    return {
        "episode_id": torch.tensor(episode_id),
        "time_in_episode": torch.tensor(time_s),
        "articulation/g1/joint_pos": torch.tensor([0.0, 0.0]),
    }


def _taken(*, length: int = 2, rewinds: list[dict] | None = None) -> TakenEpisode:
    steps = [_step(5, 0.02 * (i + 1)) for i in range(length)]
    return TakenEpisode(steps=steps, rewinds=list(rewinds or []), ticks=[{} for _ in steps])


def _bundle(
    *,
    episode_id: int = 5,
    terminated: bool = False,
    reset_facts: dict | None = None,
    terminations: dict | None = None,
    obs: ObsSnapshot | None = None,
    num_envs: int = 1,
) -> dict:
    bundle = {
        "record": {
            "episode_id": torch.tensor([episode_id] * num_envs),
            "time_in_episode": torch.full((num_envs,), 0.02),
            "articulation/g1/joint_pos": torch.zeros(num_envs, 2),
        },
        "terminated": torch.tensor([terminated] * num_envs),
        "reset_facts": reset_facts or {},
        "terminations": terminations or {},
    }
    if obs is not None:
        bundle["obs"] = obs
    return bundle


def _evaluator(
    tasks: list[Task],
    *,
    meta: dict | None = None,
    env=None,
    limits: EpisodeLimits | None = None,
) -> EpisodeEvaluator:
    if meta is None:
        meta = {"num_envs": 1, "policy_dt": 0.02, "sim_dt": 0.005}
    return EpisodeEvaluator(tasks=tasks, meta=meta, env=env, limits=limits)


class _NoopTask(Task):
    name = "noop"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"m": 1.0}, success=True)


class _EnvTermTask(Task):
    name = "envterm"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(
            metrics={"m": 1.0},
            success=True,
            termination=Termination(
                outcome=EpisodeOutcome.SUCCESS,
                code="task.success",
                source=TerminationSource.CUSTOM,
                step_index=2,
                time_s=0.04,
            ),
        )


class _FailingTask(Task):
    name = "broken"

    def score(self, ep, ctx):
        del ep, ctx
        raise RuntimeError("task scoring failed")


class _WrongReturnTask(Task):
    name = "wrong"

    def score(self, ep, ctx):
        del ep, ctx
        return "not an EpisodeResult"


def _env_term() -> tuple[EpisodeTermination, Termination]:
    term = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="env.fall",
        source=TerminationSource.ENVIRONMENT,
        step_index=2,
        time_s=0.04,
    )
    return EpisodeTermination(primary=term), term


def test_deadline_is_stamped_at_the_exact_tick():
    """A deadline terminates only when the episode length equals the tick."""
    evaluation = _evaluator([_NoopTask()], limits=EpisodeLimits(timeout_steps=2)).evaluate(
        _taken(length=2), _bundle(), 0
    )

    assert evaluation.termination is not None
    primary = evaluation.termination.primary
    assert primary.outcome is EpisodeOutcome.DEADLINE
    assert primary.code == "runtime.task_deadline"
    assert primary.step_index == 2
    assert evaluation.episode.terminal_index == 2


def test_deadline_is_not_stamped_after_an_overrun():
    """An episode longer than the deadline did not end through it."""
    evaluation = _evaluator([_NoopTask()], limits=EpisodeLimits(timeout_steps=2)).evaluate(
        _taken(length=3), _bundle(), 0
    )

    assert evaluation.termination is None
    assert evaluation.episode.terminal_index is None


def test_invalid_reset_is_a_core_termination():
    """An initializer-invalid reset is recorded at the episode-local tick 0."""
    reset_facts = {0: {"reset_index": 0, "reset_valid": False, "reset_reason": "blocked"}}

    evaluation = _evaluator([_NoopTask()]).evaluate(_taken(), _bundle(reset_facts=reset_facts), 0)

    assert evaluation.termination is not None
    primary = evaluation.termination.primary
    assert primary.outcome is EpisodeOutcome.INVALID_RESET
    assert primary.code == "initializer.invalid_reset"
    assert primary.step_index == 0
    assert evaluation.episode.reset_valid is False
    assert evaluation.episode.reset_reason == "blocked"


def test_environment_termination_is_a_core_fact():
    """The bundle's environment termination group reaches the evaluation."""
    group, term = _env_term()

    evaluation = _evaluator([_NoopTask()]).evaluate(
        _taken(), _bundle(terminated=True, terminations={0: group}), 0
    )

    assert evaluation.termination is not None
    assert evaluation.termination.primary is term
    assert evaluation.episode.ended_by_termination is True


def test_plain_termination_without_a_fact_has_no_termination():
    """The engine's terminated mask alone does not invent a termination fact."""
    evaluation = _evaluator([_NoopTask()]).evaluate(_taken(), _bundle(terminated=True), 0)

    assert evaluation.termination is None
    assert evaluation.episode.ended_by_termination is True


def test_task_termination_outranks_environment_under_core_precedence():
    """A task SUCCESS wins over environment FAILURE and a deadline."""
    group, _ = _env_term()
    evaluation = _evaluator([_EnvTermTask()], limits=EpisodeLimits(timeout_steps=2)).evaluate(
        _taken(length=2),
        _bundle(terminated=True, terminations={0: group}),
        0,
    )

    assert evaluation.termination is not None
    assert evaluation.termination.primary.outcome is EpisodeOutcome.SUCCESS
    codes = {term.code for term in evaluation.termination.all}
    assert {"env.fall", "runtime.task_deadline", "task.success"} <= codes


def test_failing_and_wrong_return_tasks_are_omitted_in_declaration_order():
    """A raising or wrongly typed task is skipped without reordering the rest."""
    tasks = [_NoopTask(), _FailingTask(), _EnvTermTask(), _WrongReturnTask()]
    evaluation = _evaluator(tasks).evaluate(_taken(), _bundle(), 0)

    assert [score.task for score in evaluation.scores] == ["noop", "envterm"]
    assert [view.task for view in evaluation.views] == ["noop", "envterm"]
    assert [score.result.success for score in evaluation.scores] == [True, True]


def test_skip_set_omits_tasks_but_keeps_episode_facts():
    """A skipped task contributes no score, yet the episode is still built."""
    skip = frozenset({"envterm"})
    evaluation = _evaluator([_NoopTask(), _EnvTermTask()]).evaluate(
        _taken(), _bundle(), 0, skip_tasks=skip
    )

    assert [score.task for score in evaluation.scores] == ["noop"]
    assert evaluation.episode.length == 2
    assert evaluation.episode.episode_id == 5


class _OutOfAxisTask(Task):
    name = "out_of_axis"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(
            metrics={"m": 1.0},
            success=True,
            termination=Termination(
                outcome=EpisodeOutcome.SUCCESS,
                code="task.success",
                source=TerminationSource.CUSTOM,
                step_index=99,
                time_s=1.98,
            ),
        )


def test_task_termination_outside_the_episode_axis_is_rejected():
    """A task termination on the global run tick fails the contract gate."""
    evaluator = _evaluator([_OutOfAxisTask()])

    with pytest.raises(ValueError, match=r"episode-local axis 0\.\.2"):
        evaluator.evaluate(_taken(), _bundle(), 0)


class _ContextTask(Task):
    name = "context"

    def __init__(self, seen: list[object]) -> None:
        super().__init__()
        self._seen = seen

    def score(self, ep, ctx):
        del ep
        self._seen.append(ctx)
        record = ctx.env.record()
        return EpisodeResult(
            metrics={
                "env_id": float(ctx.env_id),
                "rows": float(record["articulation/g1/joint_pos"].shape[0]),
                "query_backend": float(ctx.env.query.backend == "fake"),
            },
            success=True,
        )


def test_context_is_per_env_scoped_and_stale_after_scoring():
    """The task sees one env's rows and the context is closed afterwards."""
    seen: list[object] = []
    env = FakeBenchmarkEnv(num_envs=2)
    bundle = _bundle(num_envs=2)
    evaluator = _evaluator([_ContextTask(seen)], meta={"num_envs": 2, "policy_dt": 0.02}, env=env)

    evaluation = evaluator.evaluate(_taken(), bundle, 1)

    assert evaluation.scores[0].result.metrics == {
        "env_id": 1.0,
        "rows": 1.0,
        "query_backend": 1.0,
    }
    assert env.query_reads == 1
    with pytest.raises(EvalViewStaleError):
        seen[0].env.num_envs


class _ObsTask(Task):
    name = "obs"

    def score(self, ep, ctx):
        del ep
        obs = ctx.env.obs()
        assert obs is not None
        value = obs.value("articulation/g1/joint_pos")
        return EpisodeResult(metrics={"obs_value": float(value[0, 0])}, success=True)


def test_observation_snapshot_is_scoped_to_the_due_env():
    """The context's obs carries only the due env's row."""
    schema = ObservationSchema(
        {"articulation/g1/joint_pos": ObservationSpec("articulation/g1/joint_pos", (2,), "f32")}
    )
    obs = ObsSnapshot(
        {"articulation/g1/joint_pos": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        schema=schema,
        num_envs=2,
        device="cpu",
    )
    evaluator = _evaluator(
        [_ObsTask()],
        meta={"num_envs": 2, "policy_dt": 0.02},
        env=FakeBenchmarkEnv(num_envs=2),
    )

    evaluation = evaluator.evaluate(_taken(), _bundle(num_envs=2, obs=obs), 1)

    assert evaluation.scores[0].result.metrics == {"obs_value": 3.0}


def test_query_is_lazy_for_a_pure_task():
    """A task that ignores the context never opens the runtime query."""
    env = FakeBenchmarkEnv(num_envs=1)
    evaluator = _evaluator([_NoopTask()], env=env)

    evaluator.evaluate(_taken(), _bundle(), 0)

    assert env.query_reads == 0


class _HasQueryTask(Task):
    name = "has_query"

    def score(self, ep, ctx):
        del ep
        return EpisodeResult(metrics={"has_query": float(ctx.env.has_query)}, success=True)


class _QueryCallTask(Task):
    name = "query_call"

    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self._errors = errors

    def score(self, ep, ctx):
        del ep
        try:
            ctx.env.query
        except Exception as exc:
            self._errors.append(exc)
            raise
        return EpisodeResult(metrics={"unexpected": 1.0}, success=True)


def test_record_only_context_has_no_query_and_isolates_query_callers():
    """Without a live env the context is explicitly query-unavailable."""
    errors: list[Exception] = []
    evaluator = _evaluator([_HasQueryTask(), _QueryCallTask(errors)])

    evaluation = evaluator.evaluate(_taken(), _bundle(), 0)

    assert [score.task for score in evaluation.scores] == ["has_query"]
    assert evaluation.scores[0].result.metrics == {"has_query": 0.0}
    assert len(errors) == 1
    assert isinstance(errors[0], EvalQueryUnavailableError)


def test_episode_data_fields_and_values_are_unchanged():
    """The evaluator builds the same EpisodeData fact set as the sink did."""
    group, _ = _env_term()
    sample = {"articulation/g1/joint_pos": torch.zeros(2)}
    reset_facts = {
        0: {
            "reset_index": 0,
            "reset_sample": sample,
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    evaluator = _evaluator([_EnvTermTask()], limits=EpisodeLimits(timeout_steps=2))

    evaluation = evaluator.evaluate(
        _taken(length=2, rewinds=[{"tick": 3}]),
        _bundle(terminated=True, reset_facts=reset_facts, terminations={0: group}),
        0,
    )
    ep = evaluation.episode

    assert set(ep.data) == {
        "episode_id",
        "time_in_episode",
        "articulation/g1/joint_pos",
    }
    assert ep.env_id == 0
    assert ep.episode_id == 5
    assert ep.length == 2
    assert ep.duration_s == pytest.approx(0.04)
    assert ep.ended_by_termination is True
    assert ep.rewound is True
    assert ep.policy_dt == pytest.approx(0.02)
    assert ep.sim_dt == pytest.approx(0.005)
    assert ep.step_index.tolist() == [1, 2]
    assert ep.step_index.dtype == np.int64
    assert ep.time_s.tolist() == [0.02, 0.04]
    assert ep.reset_index == 0
    assert ep.reset_sample is sample
    assert ep.reset_valid is True
    assert ep.reset_reason is None
    assert ep.terminal_index == 2
    assert ep.termination is evaluation.termination
    assert {term.code for term in ep.terminations} == {
        "env.fall",
        "runtime.task_deadline",
        "task.success",
    }
