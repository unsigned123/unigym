"""`unigym describe` -- print a config's assembled ``ModelIR`` in-process.

The command walks the SAME construction path as ``collect_cli`` (yaml load
-> ``build_spec`` -> ``build_instance``) and never spawns a run or talks to
the daemon, so the tables answer the static-model question ("which joints /
bodies / cameras / actuators does this config assemble, with what
addresses?") without stepping physics.

Output faces are pure ``Table`` reprs (no pandas); ``--json`` emits the
canonical IR document (hex floats, stable keys, backend extensions).
The assembled IR is the COMPILED one since v098 M2 (addresses / masses /
ids filled); ``--compiled`` is retained as an explicit synonym of that
live-model description.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace

import yaml

_FACES = ("summary", "joints", "bodies", "cameras", "geoms", "sites",
          "actuators", "all")

_TABLES = ("bodies", "joints", "geoms", "sites", "cameras", "actuators",
           "materials")

_SPEC_DEFAULTS = dict(
    num_envs=None,
    steps=None,
    episode_length_s=None,
    no_artifacts=True,
    ckpt_dir=None,
    instruction=None,
    seed=None,
    set=None,
    adapter_params=None,
    label=None,
    serve_camera=None,
    serve_frame_mb=4.0,
    serve_state_dir=None,
    run_log=None,
    serve_pause=False,
    replay=None,
    replay_mode="action",
    replay_key=None,
    replay_strict=False,
    replay_freeze_rng=False,
    direct=False,
    watch=None,
    enable_cameras=False,
    device="cpu",
)


def _spec_args(config: str, output_dir: str) -> SimpleNamespace:
    """Build the minimal collect-shaped namespace ``build_spec`` reads.

    Args:
        config: The run yaml path (the ``--config`` role).
        output_dir: A throwaway output dir (describe writes no artifacts).

    Returns:
        A namespace carrying the collect parser's defaults.
    """
    fields = dict(_SPEC_DEFAULTS)
    fields.update(config=config, output_dir=output_dir)
    return SimpleNamespace(**fields)


def _summary(model) -> str:
    """Return the human summary block for one IR (metadata + counts)."""
    from unigym.backends.registry import Trait

    lines = [repr(model)]
    source = model.meta.get("source")
    if source:
        lines.append(f"source: {source}")
    for key in _TABLES:
        lines.append(f"{key}: {len(getattr(model, key))}")
    present = [trait.name for trait in Trait
               if trait.name.startswith("MODEL_") and model.supports(trait)]
    lines.append("supports: " + (", ".join(present) or "-"))
    return "\n".join(lines)


def _render(model, face: str) -> str:
    """Return the requested face as text (summary or one/ all tables)."""
    if face == "summary":
        return _summary(model)
    if face == "all":
        blocks = [_summary(model)]
        blocks.extend(f"[{key}]\n{getattr(model, key)}" for key in _TABLES)
        return "\n\n".join(blocks)
    return str(getattr(model, face))


def _render_set(model_set, face: str) -> str:
    """Render a ModelSet: one section per articulation (+ merged if distinct)."""
    blocks = []
    for name in model_set.names():
        blocks.append(f"[articulation {name}]\n{_render(model_set[name], face)}")
    merged = model_set.merged
    distinct = merged is not None and not (
        len(model_set) == 1 and merged is model_set[next(iter(model_set))])
    if distinct:
        blocks.append(f"[merged]\n{_render(merged, face)}")
    return "\n\n".join(blocks)


def _set_json(model_set) -> dict:
    """Return the canonical per-articulation + merged JSON document."""
    document = {
        "articulations": {
            name: model_set[name].to_dict(include_ext=True,
                                          include_meta=False)
            for name in model_set.names()
        },
        "merged": (model_set.merged.to_dict(include_ext=True,
                                            include_meta=False)
                   if model_set.merged is not None else None),
    }
    return document


def _compiled_model(instance):
    """Return the compiled-model description for the live env.

    Resolves the backend's ``model_handle`` face from the registry (v101
    A1.5) instead of sniffing env attributes.

    Raises:
        SystemExit: The backend exposes no compiled model handle.
    """
    from unigym.backends.registry import backend_info, build_name

    info = backend_info(build_name(instance.spec.backend))
    if info.model_handle is None:
        raise SystemExit(
            f"[describe] error: --compiled is unavailable for backend "
            f"{instance.spec.backend!r} | hint: drop --compiled for the "
            f"declared IR")
    model = info.model_handle()(instance.env)
    if model is None:
        raise SystemExit(
            f"[describe] error: backend {instance.spec.backend!r} exposed no "
            f"compiled model | hint: drop --compiled for the declared IR")
    factory = info.model_provider
    if factory is None:
        raise SystemExit(
            f"[describe] error: backend {instance.spec.backend!r} declares "
            f"no model provider")
    return factory()().describe(model)


def cmd_describe(a: argparse.Namespace, extra: list[str]) -> int:
    """Build the run's IR in-process and print the requested face."""
    from unigym.packs.load import load_mounted_code

    load_mounted_code()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tmp_dir = tempfile.mkdtemp(prefix="unigym_describe_")
    instance = None
    try:
        from unigym.run.build import build_instance
        from unigym.run.spec import build_spec

        with contextlib.redirect_stdout(sys.stderr):
            spec = build_spec(cfg, _spec_args(a.config, tmp_dir))
            instance = build_instance(spec)
        if a.compiled:
            model = _compiled_model(instance)
        else:
            model = instance.model
    finally:
        if instance is not None:
            with contextlib.redirect_stdout(sys.stderr):
                instance.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if a.json:
        if a.compiled:
            print(json.dumps(model.to_dict(include_ext=True,
                                           include_meta=False), indent=2))
        else:
            print(json.dumps(_set_json(model), indent=2))
    elif a.compiled:
        print(_render(model, a.face))
    else:
        print(_render_set(model, a.face))
    return 0


def _add_args(parser: argparse.ArgumentParser) -> None:
    """Attach the describe arguments to *parser* (shared by both entries)."""
    parser.add_argument("config", help="path to run yaml")
    parser.add_argument("--json", action="store_true",
                        help="canonical IR JSON (hex floats, extensions)")
    parser.add_argument("--compiled", action="store_true",
                        help="describe the live compiled MjModel (the "
                             "assembled IR is already compiled since M2)")
    parser.add_argument("--face", choices=_FACES, default="summary",
                        help="which table to print (default: summary)")


def add_describe_subparser(sub) -> None:
    """Register the ``describe`` command (lazy CLI dispatch)."""
    parser = sub.add_parser(
        "describe", help="print a config's assembled ModelIR in-process "
                         "(summary / tables / --json / --compiled)")
    _add_args(parser)
    parser.set_defaults(fn=cmd_describe)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry (module use); the launcher dispatches via ``fn``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "describe":
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="unigym describe")
    _add_args(parser)
    return cmd_describe(parser.parse_args(argv), [])


if __name__ == "__main__":
    sys.exit(main())
