"""watchers.spec -- the watcher DECLARATION schema (the convergence point
of §8.6's two entry points).

Both entry points (yaml born-armed / CLI live add) converge on
:class:`WatchSpec`: schema validation (unknown keys rejected fail-fast,
one predicate tier exactly), action/trigger field validation, and the
tier-2 budget key (:meth:`effective_tier`).  Also home of the expr
compilation helpers (console-eval pass-through semantics, incl. the
exec-block trailing-expression fallback) and the callable path checker
(explicit ``package.module:Class`` importlib -- user workspace only,
unigym never ships user functions).

Pure stdlib (+ lazy importlib): safe to import from the CLI/daemon side.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .base import ACTIONS, DEFAULT_ON, DIRECT, WatchError

__all__ = ["WatchSpec", "TIER2_CAP", "DEFAULT_COOLDOWN", "compile_expr",
           "exec_block", "check_callable_path"]

TIER2_CAP = 8
DEFAULT_COOLDOWN = 50

_SPEC_KEYS = {"name", "kind", "template", "params", "expr", "callable",
              "on", "mode", "hold", "once", "cooldown", "every", "tier",
              "guard", "note", "allow_direct"}


# --------------------------------------------------------------------------- #
# spec (declaration; both entry points converge here -- §8.6)
# --------------------------------------------------------------------------- #
@dataclass
class WatchSpec:
    """One watcher declaration (name + predicate + trigger + actions)."""
    name: str
    kind: str = "template"          # template | expr | callable
    template: str | None = None
    params: dict = field(default_factory=dict)
    expr: str | None = None
    callable_path: str | None = None
    on: tuple = DEFAULT_ON
    mode: str = "edge"              # edge | level
    hold: int = 1
    once: bool = False
    cooldown: int = DEFAULT_COOLDOWN
    every: int | None = 1
    tier: int | None = None         # default from template/kind
    guard: str | None = None        # tier-1 armed-window expr
    note: str = ""
    allow_direct: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "template": self.template, "params": self.params,
                "expr": self.expr, "callable": self.callable_path,
                "on": list(self.on), "mode": self.mode, "hold": self.hold,
                "once": self.once, "cooldown": self.cooldown,
                "every": self.every, "tier": self.tier, "guard": self.guard,
                "note": self.note, "allow_direct": self.allow_direct}

    @classmethod
    def from_dict(cls, d: dict) -> WatchSpec:
        if not isinstance(d, dict):
            raise WatchError(f"watch spec must be a mapping, got "
                             f"{type(d).__name__}")
        # YAML 1.1 parses a bare `on:` key as boolean True (and `off:`/
        # `no:` as False) -- normalize back to the schema names.
        d = {("on" if k is True else "off" if k is False else k): v
             for k, v in d.items()}
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise WatchError("watch spec: 'name' (non-empty string) is "
                             f"required, got {name!r}")
        unknown = set(d) - _SPEC_KEYS
        if unknown:
            raise WatchError(
                f"watch '{name}': unknown key(s) {sorted(unknown)} "
                f"(allowed: {sorted(_SPEC_KEYS)})")
        given = [k for k, has in (("template", d.get("template")),
                                  ("expr", d.get("expr")),
                                  ("callable", d.get("callable"))) if has]
        if len(given) != 1:
            raise WatchError(
                f"watch '{name}': exactly one of template|expr|callable "
                f"is required, got {given or 'none'}")
        on = d.get("on", DEFAULT_ON)
        if isinstance(on, str):
            on = [s.strip() for s in on.split(",") if s.strip()]
        on = tuple(on)
        bad = [a for a in on if a not in ACTIONS]
        if bad:
            raise WatchError(
                f"watch '{name}': unknown action(s) {bad} (available: "
                f"{list(ACTIONS)})")
        mode = d.get("mode", "edge")
        if mode not in ("edge", "level"):
            raise WatchError(
                f"watch '{name}': mode must be edge|level, got {mode!r}")
        spec = cls(
            name=name.strip(), kind=given[0],
            template=d.get("template"), params=d.get("params") or {},
            expr=d.get("expr"), callable_path=d.get("callable"),
            on=on, mode=mode, hold=int(d.get("hold", 1)),
            once=bool(d.get("once", False)),
            cooldown=int(d.get("cooldown", DEFAULT_COOLDOWN)),
            every=int(d.get("every") or 0) or None,
            tier=d.get("tier"), guard=d.get("guard"), note=d.get("note", ""),
            allow_direct=bool(d.get("allow_direct", False)))
        if spec.hold < 1:
            raise WatchError(
                f"watch '{name}': hold must be >= 1, got {spec.hold}")
        if spec.cooldown < 0:
            raise WatchError(
                f"watch '{name}': cooldown must be >= 0, got "
                f"{spec.cooldown}")
        if spec.tier is not None and spec.tier not in (1, 2):
            raise WatchError(
                f"watch '{name}': tier must be 1|2, got {spec.tier!r}")
        if DIRECT in spec.on and not spec.allow_direct:
            raise WatchError(
                f"watch '{name}': action 'direct' requires "
                f"allow_direct: true (P2 explicit switch)")
        return spec

    @property
    def template_name(self) -> str:
        """The declared template name (kind='template' invariant)."""
        if self.template is None:
            raise WatchError(
                f"watch '{self.name}': kind 'template' requires 'template'")
        return self.template

    @property
    def callable_target(self) -> str:
        """The declared callable path (kind='callable' invariant)."""
        if self.callable_path is None:
            raise WatchError(
                f"watch '{self.name}': kind 'callable' requires 'callable'")
        return self.callable_path

    def effective_tier(self) -> int:
        """The budget tier (§2.2): declared, template-inherent, or the
        expr/callable default 2."""
        if self.tier is not None:
            return int(self.tier)
        if self.kind == "template":
            from .templates import get_template
            return int(get_template(self.template_name)["tier"])
        return 2


# --------------------------------------------------------------------------- #
# expr compilation (console eval pass-through semantics)
# --------------------------------------------------------------------------- #
def compile_expr(expr) -> tuple:
    """-> (eval-code | source, is_exec).  The exec path keeps the SOURCE
    (the trailing-expression rewrite needs the AST, not the code)."""
    if not isinstance(expr, str) or not expr.strip():
        raise WatchError("expr: a non-empty string is required")
    try:
        return compile(expr, "<watcher>", "eval"), False
    except SyntaxError:
        try:
            ast.parse(expr, "<watcher>")
        except SyntaxError as e:
            raise WatchError(f"expr syntax error: {e}") from None
        return expr, True


def exec_block(src: str, ns: dict):
    """Statement path: full exec; the trailing expression statement's
    value is returned (Inspector._exec_block semantics)."""
    tree = ast.parse(src, "<watcher>")
    ret = "__watcher_ret__"
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id=ret, ctx=ast.Store())],
            value=tree.body[-1].value)
        ast.fix_missing_locations(tree)
    exec(compile(tree, "<watcher>", "exec"), ns)       # noqa: S102
    return ns.pop(ret, None)


def load_callable(path: str):
    """'package.module:Class' or '/abs/path/file.py:Class' -> the object.

    Both forms are EXPLICIT (zero intrusion): the dotted form imports
    from the run's sys.path, the file form loads by location -- the run
    process never needs the author's PYTHONPATH.
    """
    import importlib
    import importlib.util
    module, _, attr = path.partition(":")
    if not module or not attr:
        raise WatchError(
            f"callable: expected 'package.module:Class' or "
            f"'/abs/path/file.py:Class', got {path!r}")
    if module.endswith(".py") and ("/" in module or module.endswith(".py")):
        if not module.startswith("/"):
            raise WatchError(
                f"callable: file form must be an absolute path, got "
                f"{module!r} (relative imports are ambiguous across the "
                f"CLI/daemon/run process boundary)")
        if not Path(module).is_file():
            raise WatchError(f"callable: no such file: {module!r}")
        name = "unigym_user_watcher_" + \
            "".join(c for c in Path(module).stem if c.isalnum())
        spec_ = importlib.util.spec_from_file_location(name, module)
        if spec_ is None or spec_.loader is None:
            raise WatchError(f"callable: cannot load {module!r}")
        mod = importlib.util.module_from_spec(spec_)
        sys.modules[name] = mod
        spec_.loader.exec_module(mod)
    else:
        if not all(p.isidentifier() for p in attr.split(".")):
            raise WatchError(
                f"callable: malformed path {path!r} "
                f"(expected 'package.module:Class[.attr]')")
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            raise WatchError(
                f"callable: cannot import {module!r}: {e} (hint: the RUN "
                f"process resolves module paths -- use the absolute file "
                f"form '/abs/path/file.py:Class' for out-of-tree modules)"
            ) from None
    obj = mod
    try:
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError as e:
        raise WatchError(f"callable: {e}") from None
    if not callable(obj):
        raise WatchError(f"callable: {path!r} is not callable")
    return obj


def check_callable_path(path) -> str:
    """Validate the explicit path NOW (add-time fail-fast)."""
    if not isinstance(path, str):
        raise WatchError(
            f"callable: expected a string path, got {path!r}")
    load_callable(path)
    return path
