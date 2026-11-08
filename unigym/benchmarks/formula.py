"""FormulaTask: declarative benchmark tasks from yaml (B1.5).

A formula task is fully described by data -- a metrics list plus a success
predicate tree -- so SIMPLE benchmark tables (metric formulas + threshold
predicates) need no Python.  Third reuse of the framework's declarative-yaml
pattern (events/terminations/filters before it).

yaml form (``benchmark.tasks`` section; ``class: formula`` routes here,
sections without ``class`` keep using the named TASK_REGISTRY)::

    - name: simple_forward
      class: formula            # omitted -> named registry task (stand, ...)
      description: required, goes verbatim into benchmark_report.json
      metrics:
        - {id: alive_fraction, term: frame_fraction, key: gravity_b,
           axis: 2, target: -1.0, tol: 0.2}
        - {id: vel_err_mean, term: tracking_error, state: base_lin_vel,
           cmd: command, dims: [0, 1], reduction: mean}
      success:                  # optional; absent -> success=None
        all:
          - {metric: alive_fraction, gte: 0.9}
          - {metric: vel_err_mean, lte: 0.4}

Predicate tree: combinators ``all``/``any`` (non-empty list) and ``not``
(single child), leaves ``{metric: <id>, <comparator>: <value>}`` with
comparators gte/lte/gt/lt/between([lo, hi])/eq (inclusive bounds).  At most
TWO combinator levels (a leaf may sit at any depth; a third combinator
level is a spec error).  Evaluation is strict-boolean, numpy-free.

Fail-fast discipline (deliberate, differs from BenchmarkSink's per-task
isolation): ONE failing metric raises ValueError inside ``score`` -- the
task never returns partial metrics.  A formula error is a *definition*
error by the task author (bad key/params), silently missing metrics would
poison the report far harder than a loud, named failure; BenchmarkSink
still isolates the whole task per episode.

v1 guardrails: term templates are a whitelist (benchmarks/terms.py), every
op/post/reduction field is an enum, and free-form ``pred_spec`` expression
strings are rejected at construction.  Tasks that need anything beyond the
DSL (cross-episode state, DTW, external judges) must be written as a
Python Task -- the DSL is the fast path, not the universal path.
"""
from __future__ import annotations

import numpy as np

from .base import (EpisodeData, EpisodeResult, Task,
                   robot_state_key_uses)
from .terms import get_term, metric_frames, validate_term_spec

# reserved metric ids (Task.aggregate output keys)
_RESERVED_METRIC_IDS = ("success_rate", "n_episodes")

_COMPARATORS = ("gte", "lte", "gt", "lt", "between", "eq")
_COMBINATORS = ("all", "any", "not")
_MAX_COMBINATOR_DEPTH = 2

# language section (owner ruling: instruction templates live in the
# task spec; K5 expansion modes from the g1pick design §4.2)
_LANGUAGE_EXPANDS = ("seeded", "all", "single")
_LANGUAGE_KEYS = ("pool", "bindings", "expand")


# --------------------------------------------------------------------------- #
# predicate tree: validation + evaluation (separate concerns)
# --------------------------------------------------------------------------- #
def _validate_predicate(node, metric_ids: set[str], path: str,
                        depth: int, errors: list[str]) -> None:
    """Recursively validate one predicate-tree node (depth = number of
    combinator levels from the root, leaves included at any depth)."""
    if not isinstance(node, dict):
        errors.append(f"{path}: predicate node must be a dict (combinator "
                      f"{_COMBINATORS} or leaf {{metric, <comparator>}}), "
                      f"got {type(node).__name__}")
        return
    combo = [k for k in _COMBINATORS if k in node]
    if combo:
        if len(combo) > 1:
            errors.append(
                f"{path}: mixed combinators {combo} (one per node)")
        key = combo[0]
        extra = set(node) - {key}
        if extra:
            errors.append(
                f"{path}: unexpected keys {sorted(extra)} alongside "
                f"combinator '{key}'")
        if depth > _MAX_COMBINATOR_DEPTH:
            errors.append(
                f"{path}: combinator nesting exceeds {_MAX_COMBINATOR_DEPTH} "
                "levels (combinator-in-combinator-in-combinator); flatten "
                "the tree or write a Python Task")
            return
        children = node[key]
        if key == "not":
            if isinstance(children, list):
                if len(children) != 1:
                    errors.append(
                        f"{path}.not: exactly one child required, got "
                        f"{len(children)}")
                children = children[:1]
            else:
                children = [children]
            kids = [(f"{path}.not", c) for c in children]
        else:
            if not isinstance(children, list) or not children:
                errors.append(
                    f"{path}.{key}: requires a non-empty list of child "
                    "nodes")
                return
            kids = [(f"{path}.{key}[{i}]", c)
                    for i, c in enumerate(children)]
        for cpath, child in kids:
            _validate_predicate(child, metric_ids, cpath, depth + 1, errors)
        return
    # leaf: {metric: id, <comparator>: value}
    unknown = set(node) - {"metric", *_COMPARATORS}
    if unknown:
        errors.append(
            f"{path}: unknown comparator/keys {sorted(unknown)} (a leaf is "
            "{{metric: <id>, <comparator>: <value>}}; comparators: "
            f"{', '.join(_COMPARATORS)})")
    if "metric" not in node:
        errors.append(
            f"{path}: leaf is missing 'metric' (must reference a "
            "metrics[].id)")
        return
    mid = node["metric"]
    if not isinstance(mid, str):
        errors.append(f"{path}.metric: must be a metric id string, got "
                      f"{mid!r}")
    elif mid not in metric_ids:
        errors.append(
            f"{path}.metric: references undefined metric '{mid}' "
            f"(defined: {sorted(metric_ids)})")
    ops = [k for k in node if k in _COMPARATORS]
    if not ops:
        errors.append(
            f"{path}: leaf is missing a comparator (available: "
            f"{', '.join(_COMPARATORS)})")
    elif len(ops) > 1:
        errors.append(f"{path}: multiple comparators {ops} (exactly one "
                      "per leaf)")
    else:
        arg = node[ops[0]]
        if ops[0] == "between":
            ok = (isinstance(arg, (list, tuple)) and len(arg) == 2
                  and all(isinstance(v, (int, float))
                          and not isinstance(v, bool) for v in arg)
                  and arg[0] <= arg[1])
            if not ok:
                errors.append(
                    f"{path}.{ops[0]}: must be [lo, hi] numbers with "
                    f"lo <= hi (inclusive bounds), got {arg!r}")
        elif isinstance(arg, bool) or not isinstance(arg, (int, float)):
            errors.append(
                f"{path}.{ops[0]}: threshold must be a number, got {arg!r}")


def evaluate_predicate_tree(node, metrics: dict[str, float]) -> bool:
    """Evaluate a validated predicate tree against a metrics dict.

    Strictly boolean; assumes the tree passed ``validate_formula_spec``
    (FormulaTask validates once at construction, never at score time).
    """
    if not isinstance(node, dict):  # pragma: no cover -- validated earlier
        raise ValueError(f"predicate node must be a dict, got {node!r}")
    for key in _COMBINATORS:
        if key not in node:
            continue
        children = node[key]
        if key == "not":
            child = children[0] if isinstance(children, list) else children
            return not evaluate_predicate_tree(child, metrics)
        if key == "all":
            return all(evaluate_predicate_tree(c, metrics)
                       for c in children)
        return any(evaluate_predicate_tree(c, metrics) for c in children)
    value = float(metrics[node["metric"]])
    if "gte" in node:
        return value >= float(node["gte"])
    if "lte" in node:
        return value <= float(node["lte"])
    if "gt" in node:
        return value > float(node["gt"])
    if "lt" in node:
        return value < float(node["lt"])
    if "eq" in node:
        return value == float(node["eq"])
    lo, hi = node["between"]
    return float(lo) <= value <= float(hi)


def _leaf_nodes(node, path: str, out: list) -> None:
    """(path, node) pairs of every predicate-tree leaf (depth-first)."""
    for key in _COMBINATORS:
        if key not in node:
            continue
        children = node[key]
        if key == "not":
            child = children[0] if isinstance(children, list) else children
            _leaf_nodes(child, f"{path}.not", out)
        else:
            for i, c in enumerate(children):
                _leaf_nodes(c, f"{path}.{key}[{i}]", out)
        return
    out.append((path, node))


def leaf_keys(tree) -> dict:
    """Leaf key -> (path, node) (the results.json / gate-table join key).

    Key rule (analyze_contract §6): the referenced metric id when UNIQUE
    in the tree, else ``"<path>.<metric>"`` (disambiguated; e.g. a metric
    compared twice lands as ``root.all[0].any[0].<id>``).
    """
    leaves: list = []
    _leaf_nodes(tree, "root", leaves)
    ids = [str(node["metric"]) for _, node in leaves]
    out: dict = {}
    for path, node in leaves:
        mid = str(node["metric"])
        out[mid if ids.count(mid) == 1 else f"{path}.{mid}"] = (path, node)
    return out


def collect_leaves(tree, metrics: dict, out: dict) -> None:
    """Per-leaf boolean truth into ``out`` (score_detail + analyze gate)."""
    for key, (_path, node) in leaf_keys(tree).items():
        out[key] = bool(evaluate_predicate_tree(node, metrics))


# --------------------------------------------------------------------------- #
# schema validator (construction time; errors point at yaml fields)
# --------------------------------------------------------------------------- #
def _validate_language(lang, errors: list[str]) -> None:
    """Validate the optional ``language:`` instruction section (schema only).

    The section is pack DATA (the pool bodies live in the pack's
    ``language/templates.yaml``); the schema fixes its shape so the report
    channel (``benchmark_report.json::tasks.<name>.params.language``) is
    well-defined:
        pool:      pool id in the pack template pool (non-empty string)
        bindings:  slot -> concrete surface form for this task instance
        expand:    seeded (default) | all | single  (design K5 modes)
    """
    if not isinstance(lang, dict):
        errors.append(f"language: section must be a dict "
                      f"{{pool, bindings, expand}}, got "
                      f"{type(lang).__name__}")
        return
    unknown = set(lang) - set(_LANGUAGE_KEYS)
    if unknown:
        errors.append(f"language: unknown keys {sorted(unknown)} "
                      f"(allowed: {sorted(_LANGUAGE_KEYS)})")
    pool = lang.get("pool")
    if not isinstance(pool, str) or not pool.strip():
        errors.append(
            f"language.pool: required non-empty string (the pool id in the "
            f"pack's language/templates.yaml), got {pool!r}")
    bindings = lang.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        errors.append(
            "language.bindings: required non-empty dict of "
            "{slot: surface-form} string pairs")
    else:
        bad = {k: v for k, v in bindings.items()
               if not isinstance(k, str) or not k
               or not isinstance(v, str) or not v}
        if bad:
            errors.append(
                f"language.bindings: every slot/surface-form pair must be "
                f"non-empty strings, got {bad}")
    expand = lang.get("expand", "seeded")
    if expand not in _LANGUAGE_EXPANDS:
        errors.append(
            f"language.expand: must be one of {_LANGUAGE_EXPANDS} "
            f"(default 'seeded'), got {expand!r}")


def validate_formula_spec(spec) -> list[str]:
    """Validate a formula task section; returns a list of error strings
    (empty = valid).  Checked: description present, metrics list well
    formed (dict items, unique ids, whitelisted terms, term params),
    success predicate tree (comparators, metric references, nesting),
    reserved metric ids.  Every message names the offending field."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"spec: formula task section must be a dict, got "
                f"{type(spec).__name__}"]
    desc = spec.get("description")
    if not isinstance(desc, str) or not desc.strip():
        errors.append(
            "description: required for formula tasks (it documents the "
            "task in benchmark_report.json); add a non-empty string")
    metrics = spec.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append(
            "metrics: required non-empty list of "
            "{id, term, **params} dicts")
        metrics = []
    ids: set[str] = set()
    for i, item in enumerate(metrics):
        path = f"metrics[{i}]"
        if not isinstance(item, dict):
            errors.append(
                f"{path}: metric item must be a dict "
                "{id, term, **params}, got "
                f"{type(item).__name__}")
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            errors.append(
                f"{path}.id: required non-empty string metric id, got "
                f"{mid!r}")
        elif mid in _RESERVED_METRIC_IDS:
            errors.append(
                f"{path}.id: '{mid}' is reserved (Task.aggregate output "
                "key); pick another id")
        elif mid in ids:
            errors.append(f"{path}.id: duplicate metric id '{mid}'")
        else:
            ids.add(mid)
        term = item.get("term")
        if not isinstance(term, str) or not term:
            errors.append(
                f"{path}.term: required template name string, got {term!r} "
                f"(whitelist: benchmarks/terms.py TERM_REGISTRY)")
            continue
        try:
            get_term(term)
        except ValueError as exc:
            errors.append(f"{path}.term: {exc}")
            continue
        params = {k: v for k, v in item.items() if k not in ("id", "term")}
        for err in validate_term_spec(term, params):
            errors.append(f"{path}: {err}")
    success = spec.get("success")
    if success is not None:
        _validate_predicate(success, ids, "success", 1, errors)
    if spec.get("language") is not None:
        _validate_language(spec["language"], errors)
    return errors


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
class FormulaTask(Task):
    """Benchmark task defined entirely by its yaml section.

    Attributes (params, all required unless noted):
        description: str -- emitted verbatim into benchmark_report.json
            (keeps reports self-documenting).
        metrics: list of {id, term, **params} -- one term-template
            invocation each; ids become the flat metric names.
        success: predicate tree over metric ids (optional; absent ->
            EpisodeResult.success is None).

    score(ep): evaluates every metric (term template on the episode),
    then the predicate tree, returning EpisodeResult(metrics, success).
    A metric term raising is wrapped and re-raised as ValueError naming
    the metric id -- fail-fast by design: definition errors are the task
    author's problem and must not silently drop metrics (see module
    docstring).

    The instance ``name`` (report key) defaults to "formula";
    ``build_tasks`` overrides it from the yaml section's ``name`` field.
    """

    name = "formula"

    def __init__(self, params: dict | None = None, meta: dict | None = None):
        super().__init__(params, meta)
        errors = validate_formula_spec(self.params)
        if errors:
            raise ValueError(
                f"invalid formula task spec ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors))
        if self.articulation is None:
            for item in self.params["metrics"]:
                uses = robot_state_key_uses(item)
                if uses:
                    field, key = uses[0]
                    raise ValueError(
                        f"benchmark task '{self.name}': metric "
                        f"'{item.get('id')}' reads per-articulation robot "
                        f"state {key!r} (field {field}) but no articulation "
                        "is declared; add a top-level `articulation: <name>` "
                        f"(available: {self.available_articulations() or 'none'})")
        self._metrics = []
        for item in self.params["metrics"]:
            params_ = {k: v for k, v in item.items()
                       if k not in ("id", "term")}
            self._metrics.append(
                (item["id"], item["term"], get_term(item["term"]), params_))
        self._success = self.params.get("success")

    def score(self, ep: EpisodeData) -> EpisodeResult:
        metrics: dict[str, float] = {}
        for mid, _name, term_fn, spec in self._metrics:
            try:
                metrics[mid] = float(term_fn(
                    ep, spec, articulation=self.articulation))
            except Exception as exc:  # wide on purpose: any metric crash is
                # a spec bug -> typed fail-fast below (no partial metrics)
                raise ValueError(
                    f"formula task '{self.name}': metric '{mid}' failed "
                    f"({exc!r}) -- this is a task-definition error, fix "
                    "the spec (fail-fast: no partial metrics)") from exc
        success = None
        if self._success is not None:
            success = evaluate_predicate_tree(self._success, metrics)
        return EpisodeResult(metrics=metrics, success=success)

    # ------------------------------------------------------------------ #
    # detailed face (results.json; benchmark_contract §results)
    # ------------------------------------------------------------------ #
    def score_detail(self, ep: EpisodeData,
                     retain_frames: bool = False) -> tuple:
        """``score`` plus the declaration-derived detail face.

        Returns ``(EpisodeResult, leaves, frames)`` where

        * ``leaves`` maps every predicate-tree leaf to its boolean truth:
          ``{leaf_key: bool}``; ``leaf_key`` = the referenced metric id when
          unique in the tree, else ``"<path>.<metric>"`` (the same key rule
          the analyze gate renderer uses -- analyze_contract §6);
        * ``frames`` (only when ``retain_frames``) maps metric id -> the
          per-frame series the term reduced (float64 (T,)) -- the
          pre-reduction sequence every whitelisted template already
          computes, retained verbatim (terms without a natural series
          contribute nothing).

        Zero re-derivation: one metric pass feeds metrics + leaves + frames.
        """
        metrics: dict[str, float] = {}
        frames: dict[str, np.ndarray] = {}
        for mid, term_name, term_fn, spec in self._metrics:
            try:
                metrics[mid] = float(term_fn(
                    ep, spec, articulation=self.articulation))
                if retain_frames:
                    series = metric_frames(
                        term_name, ep, spec,
                        articulation=self.articulation)
                    if series is not None:
                        frames[mid] = np.asarray(series, dtype=np.float64)
            except Exception as exc:
                raise ValueError(
                    f"formula task '{self.name}': metric '{mid}' failed "
                    f"({exc!r}) -- this is a task-definition error, fix "
                    "the spec (fail-fast: no partial metrics)") from exc
        leaves: dict[str, bool] = {}
        if self._success is not None:
            collect_leaves(self._success, metrics, leaves)
        success = None
        if self._success is not None:
            success = evaluate_predicate_tree(self._success, metrics)
        return EpisodeResult(metrics=metrics, success=success), leaves, frames
