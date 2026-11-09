"""Benchmark packs: namespaced task-definition sources.

A pack is a directory of declarative benchmark task definitions that lives
OUTSIDE the package -- multi-source benchmark tables (SIMPLE, per-project
task sets) stay isolated from each other and users never edit package code
to add a task.  Layout (§7 of docs/contracts/benchmark_contract.md):

    <pack_dir>/pack.yaml     # {name: required, version?, description?}
                tasks/*.yaml # FormulaTask declarations (schema as in §9)
                tasks/*.py   # hand-written Task plugins (see guardrail below)
                references/  # pack-owned reference npz files; a relative
                             # ``reference:`` in reference_mpjpe resolves here
                templates/*.yaml  # task templates (v071 §2): a FormulaTask
                                  # skeleton with ``{slot}`` references; the
                                  # ``slots:`` head declares every slot and
                                  # its type (scalar/enum/ref/node)
    + pack.yaml may declare ``templates:`` + ``task_rows:`` (a parameter-row
      table, one row per task) instead of -- or beside -- materialized
      tasks/*.yaml; rows instantiate templates at LOAD time and the resolved
      spec crosses the exact same validation gate as a materialized task.

Discovery order (later sources see earlier ones for ambiguity reporting;
nothing is ever silently overwritten):

    1. builtin   -- the first-party TASK_REGISTRY as namespace "builtin"
    2. benchmark.packs: [paths...]   from the run yaml's benchmark section
    3. packs_root -- unigym.yaml ``packs_root`` key, default <repo>/benchmark_packs
                     (each immediate subdir owning a pack.yaml is a pack)

Canonical task name is ``<pack>/<task>`` (the report key).  A bare name
("stand") resolves only when unique across ALL namespaces; an ambiguous
bare name raises ValueError listing the ``<pack>/<task>`` candidates.
Within a pack, task names must be unique (yaml file stems, explicit yaml
``name`` keys and plugin task names all share one namespace); canonical
collisions are impossible by construction, pack-name collisions across
discovery sources are a loud ValueError.

Aggregated errors: ALL yaml/schema problems of one pack are collected and
raised in a single ValueError (file + field per message) -- a broken pack
is fixed in one edit-test round, not one error per run.

Python plugin guardrail (trusted-source-only): ``tasks/*.py`` is loaded
with importlib (module name ``unigym_pack_<pack>_<stem>``, so two packs
with the same file stem cannot collide in sys.modules).  Loading EXECUTES
arbitrary Python -- only ever point discovery at directories you control,
and every plugin module must say so: its docstring must contain
"trusted-source" (case-insensitive) or the loader rejects it.  The loader
looks for, in order:

* ``TASK`` -- a Task subclass or a preconfigured Task instance
* ``TASKS`` -- a list of Task subclasses and/or instances

A subclass is instantiated per run as ``cls(params, meta)`` (an instance's
baked-in params become defaults; section params override them).  Anything
beyond the FormulaTask DSL + this plugin seam (cross-episode state, DTW,
external judges) is exactly what plugins are for -- see the escape-hatch
rule in docs/contracts/benchmark_contract.md §9.5.

numpy/yaml only at module level; plugin loading is importlib + the
benchmarks package (no torch/isaaclab anywhere).
"""
from __future__ import annotations

import copy
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .base import TASK_REGISTRY, EpisodeData, Task  # noqa: F401 (EpisodeData: docs)
from .formula import validate_formula_spec
from .pack_meta import (load_meta_package, resolve_row,  # noqa: F401
                        rewrite_pack_references)

_BUILTIN_NS = "builtin"
_MODULE_PREFIX = "unigym_pack_"
_PLUGIN_DOCSTRING_MARKER = "trusted-source"
_MODULE_NAME_SAFE = re.compile(r"[^0-9a-zA-Z_]")


# --------------------------------------------------------------------------- #
# definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PackTaskDef:
    """One resolvable task definition inside a pack.

    kind == "formula": ``spec`` holds the validated FormulaTask section
    (description/metrics/success; ``name``/``class`` stripped, pack-relative
    ``reference:`` paths rewritten absolute).  kind == "class": ``cls`` is a
    Task subclass and ``default_params`` the params baked into a plugin
    instance (empty for plain classes).  ``source`` names the defining file
    for diagnostics; ``description`` is the first doc line for CLI listings.
    """

    pack: str
    task: str
    kind: str                       # "formula" | "class"
    spec: dict | None = None
    cls: type[Task] | None = None
    default_params: dict = field(default_factory=dict)
    source: str = ""
    description: str = ""           # first doc line, for CLI listings

    @property
    def canonical(self) -> str:
        return f"{self.pack}/{self.task}"

    def instantiate(self, params: dict, meta: dict) -> Task:
        """Build the run instance; the report key is set by the caller."""
        if self.kind == "class":
            if self.cls is None:
                raise ValueError(
                    f"task {self.canonical!r}: class definition has no cls")
            merged = {**self.default_params, **(params or {})}
            return self.cls(params=merged, meta=meta)
        if self.spec is None:
            raise ValueError(
                f"task {self.canonical!r}: formula definition has no spec")
        merged = {**self.spec, **(params or {})}
        from .formula import FormulaTask  # lazy: avoids the import cycle
        return FormulaTask(params=merged, meta=meta)


@dataclass
class Pack:
    """A loaded pack: identity + its task definitions."""

    name: str
    root: Path | None               # None for the builtin namespace
    description: str | None
    version: str | None
    tasks: dict[str, PackTaskDef]   # local task name -> definition

    def __len__(self) -> int:
        return len(self.tasks)

    def dump_resolved(self) -> dict[str, dict]:
        """Introspection (v071 §2.2 rule 4): ``{task name: resolved spec}``
        for every formula task of the pack -- materialized and
        template-resolved alike.  The dump is a deep copy: one-way stream
        (load-time resolve -> consumers), the runtime never writes back."""
        return {task: copy.deepcopy(d.spec)
                for task, d in sorted(self.tasks.items())
                if d.kind == "formula" and d.spec is not None}


class PackIndex:
    """All known packs (builtin first), with the resolution API.

    * ``resolve_task(name) -> (PackTaskDef, pack_name)`` -- canonical
      ``<pack>/<task>`` direct lookup, or a bare name that must be unique
      across every namespace (ambiguity lists the candidates).
    * ``expand_wildcards(sections)`` -- a section named ``<pack>/*`` becomes
      one ``{name: <pack>/<task>}`` section per task of the pack (sorted);
      wildcard sections must not carry extra keys.
    """

    def __init__(self, packs: list[Pack]):
        self.packs: dict[str, Pack] = {}
        for pack in packs:
            if pack.name in self.packs:
                raise ValueError(
                    f"pack name '{pack.name}' declared twice "
                    f"({self.packs[pack.name].root} and {pack.root}); "
                    "rename one pack.yaml -- collisions are never silently "
                    "merged")
            self.packs[pack.name] = pack

    # -- resolution ------------------------------------------------------- #
    def resolve_task(self, name) -> tuple[PackTaskDef, str]:
        """Canonical-or-unique-bare lookup; errors list the alternatives."""
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"benchmark task name must be a non-empty string, got {name!r}")
        if "/" in name:
            pack_name, _, task = name.partition("/")
            pack = self.packs.get(pack_name)
            if pack is None:
                raise ValueError(
                    f"unknown pack '{pack_name}' in task name {name!r} "
                    f"(available packs: {sorted(self.packs)})")
            if task not in pack.tasks:
                raise ValueError(
                    f"pack '{pack_name}' has no task '{task}' "
                    f"(available: {sorted(pack.tasks)})")
            return pack.tasks[task], pack_name
        hits = [(p.name, t) for p in self.packs.values()
                for t in p.tasks if t == name]
        if not hits:
            raise ValueError(
                f"unknown benchmark task '{name}' (available: "
                f"{self.canonical_names()}; use the '<pack>/<task>' form)")
        if len(hits) > 1:
            raise ValueError(
                f"ambiguous bare task name '{name}' matches "
                f"{sorted(f'{p}/{t}' for p, t in hits)}; qualify it as "
                "'<pack>/<task>'")
        p, t = hits[0]
        return self.packs[p].tasks[t], p

    def expand_wildcards(self, sections: list) -> list:
        """Expand ``{name: <pack>/*}`` sections into the pack's tasks."""
        out = []
        for section in sections:
            if not isinstance(section, dict):
                out.append(section)
                continue
            name = section.get("name")
            if not (isinstance(name, str) and name.endswith("/*")):
                out.append(section)
                continue
            extra = set(section) - {"name"}
            if extra:
                raise ValueError(
                    f"wildcard section {name!r} must be exactly "
                    f"{{name: '<pack>/*'}} (per-task params would be "
                    f"ambiguous); got extra keys {sorted(extra)}")
            pack_name = name[:-2]
            pack = self.packs.get(pack_name)
            if pack is None:
                raise ValueError(
                    f"unknown pack '{pack_name}' in wildcard {name!r} "
                    f"(available packs: {sorted(self.packs)})")
            for task in sorted(pack.tasks):
                out.append({"name": f"{pack_name}/{task}"})
        return out

    # -- introspection ------------------------------------------------------ #
    def canonical_names(self) -> list[str]:
        return [f"{p.name}/{t}" for p in self.packs.values()
                for t in sorted(p.tasks)]


# pack loading
# --------------------------------------------------------------------------- #
def _load_pack_yaml(root: Path) -> dict:
    """pack.yaml -> dict; raises ValueError with the file path on problems."""
    path = root / "pack.yaml"
    if not path.is_file():
        raise ValueError(f"not a benchmark pack (missing {path})")
    import yaml  # lazy: keeps `import unigym.benchmarks.packs` light
    try:
        meta = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid yaml ({exc})") from exc
    if not isinstance(meta, dict):
        raise ValueError(
            f"{path}: must be a dict {{name, version?, description?, "
            f"templates?+task_rows?}}, got {type(meta).__name__}")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: 'name' is required (non-empty string)")
    if "/" in name or name == _BUILTIN_NS:
        raise ValueError(
            f"{path}: pack name {name!r} is invalid (no '/' and not "
            f"'{_BUILTIN_NS}' -- that namespace is the first-party registry)")
    return meta


def _formula_task_from_yaml(path: Path) -> tuple[str, dict]:
    """One tasks/*.yaml -> (local task name, formula spec).

    The document IS the task section; the local name is the explicit
    ``name`` key when present, else the file stem.  ``class`` must be
    absent or "formula" (pack yamls are formula declarations by
    definition -- hand-written tasks use tasks/*.py plugins).
    """
    import yaml  # lazy, same reason as _load_pack_yaml
    try:
        spec = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid yaml ({exc})") from exc
    if not isinstance(spec, dict):
        raise ValueError(
            f"{path}: task file must be a dict (a formula task section), "
            f"got {type(spec).__name__}")
    cls_key = spec.pop("class", None)
    if cls_key is not None and cls_key != "formula":
        raise ValueError(
            f"{path}: pack yaml tasks are FormulaTask declarations; "
            f"class {cls_key!r} is invalid (omit the key or use "
            "'formula'; Python tasks go in tasks/*.py)")
    name = spec.pop("name", None) or path.stem
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: 'name' must be a non-empty string")
    return name, spec


def _plugin_defs(path: Path, pack: str) -> list[tuple[str, PackTaskDef]]:
    """Load one tasks/*.py plugin (trusted-source-only, see module doc).

    Returns [(local task name, PackTaskDef)]; raises ValueError on any
    contract violation (docstring marker, no TASK/TASKS, wrong types,
    duplicate names inside the module).
    """
    stem = path.stem
    modname = _MODULE_PREFIX + _MODULE_NAME_SAFE.sub("_", pack) \
        + "_" + _MODULE_NAME_SAFE.sub("_", stem)
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path}: cannot load plugin (no module loader)")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module  # register BEFORE exec (dataclasses etc.)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # wide on purpose: plugin import errors are
        # re-raised with context; registry entry rolled back first
        sys.modules.pop(modname, None)
        raise ValueError(f"{path}: plugin import failed ({exc!r})") from exc
    doc = (module.__doc__ or "")
    if _PLUGIN_DOCSTRING_MARKER not in doc.lower():
        raise ValueError(
            f"{path}: plugin module docstring must declare "
            f"'{_PLUGIN_DOCSTRING_MARKER}' -- loading a plugin EXECUTES "
            "arbitrary Python; only point packs at sources you control")
    raw: list = []
    if hasattr(module, "TASK"):
        raw = [module.TASK]
    elif hasattr(module, "TASKS"):
        raw = module.TASKS
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                f"{path}: TASKS must be a non-empty list of Task "
                "subclasses/instances")
    else:
        raise ValueError(
            f"{path}: no plugin entry point (define TASK -- a Task "
            "subclass or instance -- or TASKS, a list of them)")
    defs: list[tuple[str, PackTaskDef]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if isinstance(item, Task):
            cls, default_params, name = type(item), dict(item.params), \
                item.name
        elif isinstance(item, type) and issubclass(item, Task):
            cls, default_params, name = item, {}, item.name
        else:
            raise ValueError(
                f"{path}: TASKS[{i}] must be a Task instance or Task "
                f"subclass, got {item!r}")
        if not name or name in seen:
            raise ValueError(
                f"{path}: plugin task name {name!r} empty or duplicated "
                "inside the module (Task.name is the local pack task name)")
        seen.add(name)
        defs.append((name, PackTaskDef(
            pack=pack, task=name, kind="class", cls=cls,
            default_params=default_params, source=str(path))))
    return defs


def load_pack(root: Path) -> Pack:
    """Load + validate one pack directory; aggregated ValueError on any
    problem (every message prefixed with the offending file).

    Three task sources share one namespace: materialized tasks/*.yaml,
    Python plugins, and (when pack.yaml declares ``templates:`` +
    ``task_rows:``) template x parameter-row resolutions -- v071 §2."""
    root = Path(root)
    meta = _load_pack_yaml(root)
    name = meta["name"].strip()
    errors: list[str] = []
    tasks: dict[str, PackTaskDef] = {}
    task_files = sorted((root / "tasks").glob("*.yaml")) \
        if (root / "tasks").is_dir() else []
    py_files = sorted((root / "tasks").glob("*.py")) \
        if (root / "tasks").is_dir() else []
    templates: dict[str, tuple[dict, dict]] = {}
    rows: dict[str, dict] = {}
    rows_path: Path | None = None
    tables: dict = {}
    if meta.get("templates") is not None or meta.get("task_rows") is not None:
        try:
            templates, rows, rows_path, tables = load_meta_package(
                root, meta)
        except ValueError as exc:
            errors.append(str(exc))
    if not task_files and not py_files and rows_path is None:
        errors.append(f"{root / 'tasks'}: no tasks found (tasks/*.yaml, "
                      "tasks/*.py, or templates + task_rows required -- "
                      "an empty pack is a typo, not a pack)")
    references_dir = root / "references"
    for path in task_files:
        try:
            local, spec = _formula_task_from_yaml(path)
            rewrite_pack_references(spec, references_dir, str(path))
            errs = validate_formula_spec(spec)
            if errs:
                errors.extend(f"{path}: {e}" for e in errs)
                continue
            if local in tasks:
                errors.append(
                    f"{path}: duplicate task name '{local}' in pack "
                    f"'{name}' (already defined by "
                    f"{tasks[local].source})")
                continue
            desc = " ".join(str(spec.get("description", "")).split())
            tasks[local] = PackTaskDef(
                pack=name, task=local, kind="formula", spec=spec,
                source=str(path), description=desc)
        except ValueError as exc:
            errors.append(str(exc))
    if rows_path is not None:
        for local in sorted(rows):
            spec, row_errors = resolve_row(
                local, rows[local], templates, references_dir, rows_path,
                tables)
            errors.extend(row_errors)
            if spec is None:
                continue
            if local in tasks:
                errors.append(
                    f"{rows_path}#{local}: duplicate task name '{local}' in "
                    f"pack '{name}' (already defined by "
                    f"{tasks[local].source})")
                continue
            desc = " ".join(str(spec.get("description", "")).split())
            tasks[local] = PackTaskDef(
                pack=name, task=local, kind="formula", spec=spec,
                source=f"{rows_path}#{local}", description=desc)
    for path in py_files:
        try:
            for local, d in _plugin_defs(path, name):
                if local in tasks:
                    errors.append(
                        f"{path}: duplicate task name '{local}' in pack "
                        f"'{name}' (already defined by "
                        f"{tasks[local].source})")
                    continue
                tasks[local] = d
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(
            f"pack '{name}' ({root}) has {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors))
    return Pack(
        name=name, root=root,
        description=(meta.get("description") or None),
        version=meta.get("version"), tasks=tasks)


def _description_of(def_: PackTaskDef) -> str:
    if def_.description:
        return def_.description
    if def_.cls is not None:
        return " ".join((def_.cls.__doc__ or "").split())
    return ""


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def builtin_pack() -> Pack:
    """The first-party TASK_REGISTRY as namespace "builtin".

    Canonical names are ``builtin/<task>``; bare registry names keep their
    stable behavior through resolve_task (unique bare -> the registry
    class, report key = the class ``name``, never rewritten).
    """
    tasks = {}
    for name, cls in TASK_REGISTRY.items():
        tasks[name] = PackTaskDef(
            pack=_BUILTIN_NS, task=name, kind="class", cls=cls,
            source="unigym.benchmarks (TASK_REGISTRY)",
            description=" ".join((cls.__doc__ or "").split()))
    return Pack(name=_BUILTIN_NS, root=None,
                description="first-party tasks (benchmarks/tasks.py)",
                version=None, tasks=tasks)


def default_packs_root() -> Path:
    """unigym.yaml ``packs_root`` if set, else the repo ``benchmarks`` face."""
    from unigym.core.asset_root import repo_root
    from unigym.packs import repo_pack

    repo = Path(repo_root())
    dc = repo / "unigym.yaml"
    if dc.is_file():
        import yaml  # lazy
        try:
            cfg = yaml.safe_load(dc.read_text()) or {}
        except yaml.YAMLError:
            cfg = {}
        root = cfg.get("packs_root")
        if root:
            return Path(root)
    face = repo_pack().faces.get("benchmarks")
    return face if face is not None else repo / "benchmark_packs"


def discover_packs(packs_cfg: list | None = None,
                   packs_root=None) -> PackIndex:
    """Full discovery: builtin -> explicit benchmark.packs paths ->
    packs_root subdirs (sorted; a subdir counts when it owns pack.yaml).

    ``packs_cfg`` is the run yaml's ``benchmark.packs`` list -- every entry
    must BE a pack directory (missing pack.yaml is an error there, unlike
    the packs_root scan where non-pack subdirs are simply not packs).
    Each pack is loaded eagerly, so definition errors surface at discovery
    (aggregated per pack), not mid-run.
    """
    packs = [builtin_pack()]
    for raw in (packs_cfg or []):
        root = Path(raw)
        if not root.is_dir():
            raise ValueError(f"benchmark.packs entry not found: {root}")
        packs.append(load_pack(root))
    if packs_root is not None:
        root = Path(packs_root)
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (child / "pack.yaml").is_file():
                    packs.append(load_pack(child))
        return PackIndex(packs)
    # v081: span the mounted packs' benchmarks faces, external first;
    # a pack name is shadowed by the earlier mount (the repo copy is the
    # last one, exactly like the runs/scenes faces).
    from unigym.packs import pack_roots
    seen = {pack.name for pack in packs}
    for mounted in pack_roots():
        face = mounted.faces.get("benchmarks")
        if face is None or not face.is_dir():
            continue
        for child in sorted(face.iterdir()):
            if not (child / "pack.yaml").is_file():
                continue
            if child.name in seen:
                continue
            packs.append(load_pack(child))
            seen.add(child.name)
    return PackIndex(packs)
