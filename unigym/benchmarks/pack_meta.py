"""Task meta-package engine: templates x parameter rows.

A pack can declare ``templates:`` + ``task_rows:`` in its pack.yaml: the
templates are FormulaTask skeletons whose strings carry ``{slot}``
references ONLY (any other brace fails the load -- the no-logic rule,
enforced mechanically: conditions/loops/functions
cannot even be spelled); the rows are parameter lines, one per task,
instantiating one template each.  Resolution happens at pack-LOAD time
and the resolved spec crosses the exact same gate as a materialized task
(rewrite_pack_references + validate_formula_spec, §2.2 rule 2 -- the
gate's position does not move).

Slot types (the ``slots:`` head of every template):
    scalar -- str/int/float/bool
    enum   -- the value must be one of the declared candidates
    ref    -- ``'<file>:<dotted.keys>'`` (pack-root relative); the value
              must be an existing row KEY of the pointed mapping
    node   -- whole-node substitution (mapping/list/scalar) for
              structured values
The implicit ``task`` slot is the row's task name (an explicit mismatch
is a loud error).  Row values are DATA: they are never re-scanned for
slot references (no recursive templating).

The task-rows table carries its own load-time schema gate (§2.3):
schema key, row types, required template + sign-off per row -- every
error named ``<table>/<row>/<field>`` (load_task_rows).  This module is
the engine; the pack integration (load_pack) and the materialized
tasks/*.yaml path live in packs.py.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import cast

from .formula import validate_formula_spec
from .task_providers import build_provider_spec

# --------------------------------------------------------------------------- #
# task meta-package: templates x parameter rows
# --------------------------------------------------------------------------- #
TASK_ROWS_SCHEMA = "unigym.benchmarks.task_rows/1"
"""The only accepted ``schema:`` value of a ``task_rows`` table."""

_SLOT_KINDS = ("scalar", "enum", "ref", "node")
_SLOT_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_PROVIDER_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_ROW_KEYS = ("template", "provider", "signoff", "slots")
_SCALARS = (str, int, float, bool)


def _parse_yaml_file(path: Path):
    """One yaml file -> the parsed document.

    ValueError carrying the path on any parse problem.
    """
    import yaml  # lazy: keeps `import unigym.benchmarks.packs` light
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid yaml ({exc})") from exc


def _slot_decl(name: str, raw, where: str) -> dict:
    """Normalize + validate one ``slots:`` declaration.

    A declaration is either a kind string (``scalar`` / ``node``) or a
    mapping carrying exactly one payload key: ``{enum: [candidates...]}``
    or ``{ref: '<file>:<dotted.keys>'}`` (the file is pack-root relative
    and the pointer must land on a mapping -- its KEYS are the legal
    values, so a ref slot can only name rows that exist).
    """
    if not _SLOT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{where}: slot name {name!r} invalid (lowercase snake_case)")
    if isinstance(raw, str):
        kind, payload = raw, {}
    elif isinstance(raw, dict) and raw:
        extra = set(raw) - {"enum", "ref"}
        if extra or len(raw) != 1:
            raise ValueError(
                f"{where}: slot '{name}' declaration must be a kind string "
                "or a dict with exactly one of {enum: [...]}, {ref: ...}, "
                f"got keys {sorted(raw)}")
        if "enum" in raw:
            kind, payload = "enum", {"values": raw["enum"]}
        else:
            kind, payload = "ref", {"ref": raw["ref"]}
    else:
        raise ValueError(
            f"{where}: slot '{name}' declaration must be a string or a "
            f"non-empty dict, got {type(raw).__name__}")
    if kind not in _SLOT_KINDS:
        raise ValueError(
            f"{where}: slot '{name}' kind {kind!r} unknown (kinds: "
            f"{', '.join(_SLOT_KINDS)})")
    if kind == "enum":
        values = payload.get("values")
        if (not isinstance(values, list) or not values
                or not all(isinstance(v, _SCALARS) for v in values)):
            raise ValueError(
                f"{where}: slot '{name}' enum candidates must be a "
                "non-empty list of scalars")
    if kind == "ref" and (not isinstance(payload.get("ref"), str)
                          or payload["ref"].count(":") != 1):
        raise ValueError(
            f"{where}: slot '{name}' ref must be '<file>:<dotted.keys>' "
            f"(pack-root relative), got {payload.get('ref')!r}")
    return {"kind": kind, **payload}


def _resolve_ref_pointer(ref: str, root: Path, where: str) -> dict:
    """``<file>:<dotted.keys>`` -> the mapping it points at.

    The file resolves against the pack root; every dotted key must walk
    down a mapping and the landing node must be one (its keys are the
    legal values of the ref slot).
    """
    file_part, _, keys_part = ref.partition(":")
    path = root / file_part
    if not path.is_file():
        raise ValueError(f"{where}: ref target file not found: {path}")
    node = _parse_yaml_file(path)
    for key in keys_part.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(
                f"{where}: ref pointer {ref!r} does not resolve "
                f"(stuck at key {key!r})")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(
            f"{where}: ref pointer {ref!r} must land on a mapping "
            f"(the legal values are its keys), got {type(node).__name__}")
    return node


def _string_slot_names(s: str, where: str) -> list[str]:
    """The ``{slot}`` references in one template string.

    Every brace in a template string must belong to a well-formed slot
    reference -- literal braces, ``{% ... %}``, ``${...}``, ``{{ }}`` all
    fail the load.  That is the no-logic rule made
    mechanical: a template is data references and nothing else.
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "{":
            j = s.find("}", i + 1)
            if j < 0:
                raise ValueError(
                    f"{where}: unclosed '{{' in template string {s!r}")
            name = s[i + 1:j]
            if not _SLOT_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"{where}: brace span {{{name}}} in {s!r} is not a "
                    "slot reference (templates carry {slot} references "
                    "only -- no logic, no literal braces)")
            out.append(name)
            i = j + 1
        elif s[i] == "}":
            raise ValueError(
                f"{where}: unmatched '}}' in template string {s!r}")
        else:
            i += 1
    return out


def _load_template(path: Path) -> tuple[str, dict, dict]:
    """One ``templates/<name>.yaml`` -> (name, skeleton, slot decls).

    The top-level ``slots:`` key is loader metadata (stripped here): it
    declares every slot the skeleton may reference and its type.  The
    implicit ``task`` slot (the row's task name) is declared scalar
    unless the template overrides it.  The skeleton must not carry
    ``name``/``class`` keys -- the row key names the task; templates are
    pure skeletons.
    """
    name = path.stem
    doc = _parse_yaml_file(path)
    if not isinstance(doc, dict):
        raise ValueError(
            f"{path}: template must be a dict (slots head + task "
            f"skeleton), got {type(doc).__name__}")
    raw_slots = doc.pop("slots", None)
    if not isinstance(raw_slots, dict) or not raw_slots:
        raise ValueError(
            f"{path}: template must declare a non-empty 'slots:' mapping "
            "(a template without slots is a materialized task -- write "
            "it as tasks/*.yaml)")
    decls: dict[str, dict] = {}
    for sname, raw in raw_slots.items():
        decls[sname] = _slot_decl(sname, raw, str(path))
    for reserved in ("name", "class"):
        if reserved in doc:
            raise ValueError(
                f"{path}: template must not carry '{reserved}' (the row "
                "key names the task; templates are pure skeletons)")
    decls.setdefault("task", {"kind": "scalar"})
    for sname, decl in decls.items():
        if decl["kind"] == "ref":
            decl["table"] = _resolve_ref_pointer(
                decl["ref"], path.parents[1], str(path))
    return name, doc, decls


def load_meta_package(root: Path, meta: dict) -> tuple[dict, dict, Path, dict]:
    """The pack.yaml ``templates:`` + ``task_rows:`` declarations.

    Returns ({name: (skeleton, decls)}, {task: row}, rows path, tables
    registry).  Both keys are declared together or not at all (one
    without the other is a typo).  Every problem -- declaration shape, template file,
    table schema, per-row §2.3 checks -- aggregates into one ValueError.
    """
    errors: list[str] = []
    t_decl = meta.get("templates")
    r_decl = meta.get("task_rows")
    if t_decl is not None:
        if (not isinstance(t_decl, list) or not t_decl
                or not all(isinstance(t, str) and t and "/" not in t
                           for t in t_decl)):
            errors.append(
                f"{root / 'pack.yaml'}: 'templates' must be a non-empty "
                "list of template names (templates/<name>.yaml stems)")
            t_decl = []
        dupes = sorted({t for t in t_decl if t_decl.count(t) > 1})
        if dupes:
            errors.append(
                f"{root / 'pack.yaml'}: 'templates' lists {dupes} twice")
        if not isinstance(r_decl, str) or not r_decl:
            errors.append(
                f"{root / 'pack.yaml'}: 'task_rows' must be a non-empty "
                "path when 'templates' is declared (the parameter-row "
                "table, pack-root relative)")
            r_decl = ""
    if not isinstance(r_decl, str) or not r_decl:
        errors.append(
            f"{root / 'pack.yaml'}: 'task_rows' must be a non-empty path "
            "(the parameter-row table, pack-root relative)")
        r_decl = ""
    if errors:
        raise ValueError("\n".join(errors))
    templates: dict[str, tuple[dict, dict]] = {}
    tdir = root / "templates"
    if t_decl:
        if not tdir.is_dir():
            raise ValueError(
                f"{tdir}: templates declared but the directory is "
                "missing")
        for tname in t_decl:
            tpath = tdir / f"{tname}.yaml"
            if not tpath.is_file():
                errors.append(f"{tpath}: declared template not found")
                continue
            try:
                loaded = _load_template(tpath)
                if loaded[0] != tname:
                    errors.append(
                        f"{tpath}: file stem {loaded[0]!r} != the "
                        f"declared template name {tname!r}")
                    continue
                templates[tname] = (loaded[1], loaded[2])
            except ValueError as exc:
                errors.append(str(exc))
    rows: dict[str, dict] = {}
    rows_path = Path(os.path.normpath(root / r_decl))
    try:
        rows = load_task_rows(rows_path)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("\n".join(errors))
    return templates, rows, rows_path, {"task_rows": rows}


def load_task_rows(path: Path) -> dict[str, dict]:
    """One task-rows table -> ``{task name: normalized row}``.

    The load-time schema gate: row types, required fields,
    sign-off presence.  All errors aggregate into one ValueError, every
    message named ``<table>/<row>/<field>`` -- this is the correctness
    insurance for hand-maintained rows (a new task is one table line,
    and a broken line is named to its field).
    """
    if not path.is_file():
        raise ValueError(f"{path}: task-rows table not found")
    doc = _parse_yaml_file(path)
    errors: list[str] = []
    rows: dict[str, dict] = {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"{path}: table must be a dict (schema + rows), "
            f"got {type(doc).__name__}")
    schema = doc.get("schema")
    if schema != TASK_ROWS_SCHEMA:
        errors.append(
            f"{path}: schema must be {TASK_ROWS_SCHEMA!r}, got {schema!r}")
    extra = sorted(set(doc) - {"schema", "rows"})
    if extra:
        errors.append(f"{path}: unknown table keys {extra}")
    raw_rows = doc.get("rows")
    if not isinstance(raw_rows, dict) or not raw_rows:
        errors.append(
            f"{path}: rows: required non-empty mapping "
            "{task name: row}")
        raw_rows = {}
    for name, row in raw_rows.items():
        if not isinstance(name, str) or not name or "/" in name:
            errors.append(
                f"{path}: rows: task name {name!r} invalid (non-empty "
                "string, no '/')")
            continue
        where = f"{path.name}/{name}"
        if not isinstance(row, dict):
            errors.append(
                f"{where}: row must be a mapping, got "
                f"{type(row).__name__}")
            continue
        unknown = sorted(set(row) - set(_ROW_KEYS))
        if unknown:
            errors.append(
                f"{where}: unknown row keys {unknown} (allowed: "
                f"{sorted(_ROW_KEYS)})")
        template = row.get("template")
        provider = row.get("provider")
        # exactly one instantiation source; providers are
        # referenced BY NAME, never by path (a path-shaped value is
        # rejected here, before the registry even sees it).
        if bool(template) == bool(provider):
            errors.append(
                f"{where}/template: exactly one of 'template' or "
                "'provider' required (a row instantiates one template "
                f"OR calls one provider), got template={template!r} "
                f"provider={provider!r}")
        if template and (not isinstance(template, str) or not template):
            errors.append(
                f"{where}/template: must be a non-empty string (the "
                "template this row instantiates)")
        if provider and (not isinstance(provider, str) or not provider):
            errors.append(
                f"{where}/provider: must be a non-empty string (the "
                "registered provider name)")
        elif provider and not _PROVIDER_NAME_RE.fullmatch(provider):
            errors.append(
                f"{where}/provider: {provider!r} is not a provider NAME "
                "(lowercase snake_case -- data cites names, never "
                "paths; a path-shaped provider reference is rejected "
                "at load)")
        signoff = row.get("signoff")
        if not isinstance(signoff, str) or not signoff.strip():
            errors.append(
                f"{where}/signoff: required non-empty string (who "
                "verified the row and against what -- the sign-off "
                "travels with the row)")
        slots = row.get("slots")
        if slots is None:
            slots = {}
        if not isinstance(slots, dict) or not all(
                isinstance(k, str) for k in slots):
            errors.append(
                f"{where}/slots: must be a mapping of slot name -> value")
            slots = {}
        rows[name] = {"template": template, "provider": provider,
                      "signoff": signoff, "slots": slots}
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def resolve_row(name: str, row: dict, templates: dict,
                 references_dir: Path, rows_path: Path, tables: dict
                 ) -> tuple[dict | None, list[str]]:
    """One row -> (resolved spec, errors).

    Slot presence + value types check first (naming
    ``<table>/<row>/<field>``), then the instantiation -- a template
    substitution, or a provider call (pure spec-builder, the
    loader owns the gate) -- then the resolved spec crosses the EXACT
    same gate as a materialized task: the shared reference rewrite +
    ``validate_formula_spec`` (the gate's position
    does not move).
    """
    errors: list[str] = []
    where = f"{rows_path.name}/{name}"
    if row.get("provider"):
        try:
            spec = build_provider_spec(row["provider"], row["slots"],
                                       tables)
        except ValueError as exc:
            return None, [f"{where}/provider: {exc}"]
        rewrite_pack_references(spec, references_dir, str(rows_path))
        errors.extend(f"{rows_path}#{name}: {e}"
                      for e in validate_formula_spec(spec))
        return (spec if not errors else None), errors
    tname = row["template"]
    if tname not in templates:
        return None, [
            f"{where}/template: unknown template {tname!r} "
            f"(declared: {sorted(templates)})"]
    skeleton, decls = templates[tname]
    values = dict(row["slots"])
    if "task" in values and values["task"] != name:
        errors.append(
            f"{where}/slots.task: {values['task']!r} != the row key "
            f"{name!r} (copy-paste drift)")
    values["task"] = name
    missing = sorted(set(decls) - set(values))
    if missing:
        errors.append(
            f"{where}/slots: missing slot(s) {missing} required by "
            f"template '{tname}'")
    extra = sorted(set(values) - set(decls))
    if extra:
        errors.append(
            f"{where}/slots: unknown slot(s) {extra} (template "
            f"'{tname}' declares {sorted(decls)})")
    if missing or extra:
        return None, errors
    for sname, decl in decls.items():
        value = values[sname]
        kind = decl["kind"]
        if kind == "scalar" and not isinstance(value, _SCALARS):
            errors.append(
                f"{where}/slots.{sname}: scalar slot got "
                f"{type(value).__name__}")
        elif kind == "enum" and value not in decl["values"]:
            errors.append(
                f"{where}/slots.{sname}: {value!r} not in enum "
                f"candidates {decl['values']}")
        elif kind == "ref":
            if not isinstance(value, str):
                errors.append(
                    f"{where}/slots.{sname}: ref slot must name a row "
                    f"key (string), got {type(value).__name__}")
            elif value not in decl["table"]:
                errors.append(
                    f"{where}/slots.{sname}: {value!r} is not a row key "
                    f"of {decl['ref']}")
        elif kind == "node" and not isinstance(
                value, (dict, list, *_SCALARS)):
            errors.append(
                f"{where}/slots.{sname}: node slot must be a mapping, "
                f"list or scalar, got {value!r}")
    if errors:
        return None, errors
    spec = cast(dict, _substitute(skeleton, values, where, errors))
    if errors or spec is None:
        return None, errors
    rewrite_pack_references(spec, references_dir, str(rows_path))
    errors.extend(f"{rows_path}#{name}: {e}"
                  for e in validate_formula_spec(spec))
    return spec, errors


def _substitute(skeleton, values: dict, where: str,
                errors: list[str]):
    """Walk the skeleton, substituting ``{slot}`` references.

    A string that IS exactly one reference takes the value whole (any
    type -- the whole-node slot); references embedded in longer strings
    stringify (the value must be a string).  Mapping keys must be
    literal.  Row values are DATA: they are never re-scanned for slot
    references (no recursive templating).
    """
    if isinstance(skeleton, dict):
        out = {}
        for key, val in skeleton.items():
            if isinstance(key, str) and _string_slot_names(key, where):
                errors.append(
                    f"{where}: mapping keys must be literal -- slot "
                    f"reference in key {key!r}")
                continue
            out[key] = _substitute(val, values, where, errors)
        return out
    if isinstance(skeleton, list):
        return [_substitute(item, values, where, errors)
                for item in skeleton]
    if not isinstance(skeleton, str):
        return skeleton
    try:
        names = _string_slot_names(skeleton, where)
    except ValueError as exc:
        errors.append(str(exc))
        return skeleton
    if not names:
        return skeleton
    unknown = [n for n in names if n not in values]
    if unknown:
        errors.append(
            f"{where}: unknown slot reference(s) {unknown} in "
            f"{skeleton!r} (declared: {sorted(values)})")
        return skeleton
    if len(names) == 1 and skeleton == f"{{{names[0]}}}":
        return values[names[0]]
    bad = [n for n in names if not isinstance(values[n], str)]
    if bad:
        errors.append(
            f"{where}: slot(s) {bad} referenced inside the string "
            f"{skeleton!r} must be string-valued (use a whole-node "
            "reference for structured values)")
        return skeleton
    for n in names:
        skeleton = skeleton.replace(f"{{{n}}}", values[n])
    return skeleton


# --------------------------------------------------------------------------- #


def rewrite_pack_references(spec: dict, references_dir: Path,
                             where: str) -> None:
    """Anchor relative ``reference:`` paths of reference_mpjpe metrics.

    Anchored at the pack's references/ (rewritten in place, BEFORE schema
    validation, so existence checks and score-time loads see the same
    file).
    """
    for i, item in enumerate(spec.get("metrics") or []):
        if not isinstance(item, dict) or item.get("term") != "reference_mpjpe":
            continue
        ref = item.get("reference")
        if isinstance(ref, str) and ref and not Path(ref).is_absolute():
            item["reference"] = str(references_dir / ref)
