"""``unigym asset``: the L0 asset primitive's query and pin faces.

The launcher owns the argparse wiring (``unigym/scripts/launch.py``); this
module owns four commands over the derived catalog (v086 §5) and the
selection lock (v086 §6):

    unigym asset catalog build [--out DIR]
    unigym asset catalog search [--kind K] [--tag T ...] [--pack P]
                                [--uses ID ...] [--json]
    unigym asset file <id>
    unigym asset selection check [--dir DIR]

Collection is delegated to ``tools/build_asset_catalog.py`` -- the single
pack scan.  ``catalog build`` runs it into ``--out``; the query commands
run it into a throwaway directory and answer from that fresh result, so a
stale cache can never answer a query and the CLI never re-implements
manifest discovery.

Fail closed and teach: every error names the offending id/dir/path and the
next command that fixes it (``docs/CONVENTIONS.md`` §9).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from difflib import get_close_matches
from pathlib import Path

import yaml

from unigym.assets import catalog as asset_catalog
from unigym.assets.manifest import (
    ManifestError,
    is_asset_id,
    load_manifest,
    split_asset_id,
)
from unigym.core.asset_root import deps_root, repo_root

REPO_ROOT = Path(repo_root())
"""The checkout that owns ``tools/``; the collector script lives there."""

COLLECTOR = REPO_ROOT / "tools" / "build_asset_catalog.py"
"""The one catalog collector (manifest scan + generator metadata)."""

DEFAULT_OUT_NAME = "unigym_asset_catalog"
"""Default catalog directory name under ``${UNIGYM_DEPS_ROOT}``."""

PAYLOAD_ROLES = {"model": "model", "texture_set": "texture"}
"""The manifest role holding a file asset's single primary payload."""

NEAR_MATCH_LIMIT = 5
"""How many near ids an unknown-id error lists."""


class AssetCliError(ValueError):
    """A command cannot answer; the message names the input and the fix."""


def _out_dir(value: str | None) -> Path:
    """The catalog output directory: ``--out`` or the deps-root cache."""
    if value:
        return Path(value).expanduser().resolve()
    return (Path(deps_root()) / DEFAULT_OUT_NAME).resolve()


def _collector_output(result: subprocess.CompletedProcess) -> str:
    """The collector's diagnostic text for one failed run."""
    text = (result.stdout or "") + (result.stderr or "")
    return text.strip() or f"the collector exited {result.returncode}"


def _run_collector(out_dir: Path) -> subprocess.CompletedProcess:
    """Run the shared collector into ``out_dir`` (stdout captured).

    The collector refuses an ``--out`` inside a pack (packs are read-only)
    and reports every broken manifest; both arrive through the return
    code and the captured text rather than a traceback.
    """
    if not COLLECTOR.is_file():
        raise AssetCliError(
            f"{COLLECTOR} is missing; the asset catalog is collected by "
            "tools/build_asset_catalog.py -- run from a complete checkout")
    return subprocess.run(
        [sys.executable, str(COLLECTOR), "--out", str(out_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)


def _fresh_catalog() -> dict:
    """Rebuild the catalog in a throwaway directory and return the document.

    The catalog is derived, so answering a query from a cached copy would
    silently answer an older pack.

    Raises:
        AssetCliError: The collector cannot finish; the message carries the
            collector's own failure lines.
    """
    with tempfile.TemporaryDirectory(prefix="unigym_asset_catalog_") as tmp:
        result = _run_collector(Path(tmp))
        if result.returncode != 0:
            raise AssetCliError(
                "the asset catalog cannot be rebuilt:\n"
                + _collector_output(result))
        return asset_catalog.load_catalog(tmp)


def _counts(entries: list[dict]) -> tuple[dict, dict]:
    """The entry counts by kind and by carrier (both sorted dicts)."""
    by_kind = Counter(entry["kind"] for entry in entries)
    by_carrier = Counter(entry["carrier"] for entry in entries)
    return dict(sorted(by_kind.items())), dict(sorted(by_carrier.items()))


def cmd_catalog_build(a, _extra=None) -> int:
    """``unigym asset catalog build`` -- write catalog.json + catalog.digest."""
    out_dir = _out_dir(getattr(a, "out", None))
    try:
        result = _run_collector(out_dir)
        if result.returncode != 0:
            print(f"[unigym asset] catalog build failed for {out_dir}:")
            print(_collector_output(result))
            return 1
        document = asset_catalog.load_catalog(out_dir)
    except (AssetCliError, asset_catalog.CatalogError) as exc:
        print(f"[unigym asset] {exc}")
        return 1
    entries = document["entries"]
    by_kind, by_carrier = _counts(entries)
    print(f"[unigym asset] catalog: {out_dir / asset_catalog.CATALOG_NAME}")
    print(f"  entries: {len(entries)}")
    print(f"  by_kind: {by_kind}")
    print(f"  by_carrier: {by_carrier}")
    print(f"  digest: {asset_catalog.catalog_digest(document)}")
    return 0


def _search_filters(a) -> dict:
    """Normalize the ``catalog search`` filters into ``find`` arguments."""
    return {
        "kind": getattr(a, "kind", None),
        "tags": list(getattr(a, "tag", None) or []),
        "pack": getattr(a, "pack", None),
        "uses": list(getattr(a, "uses", None) or []),
    }


def cmd_catalog_search(a, _extra=None) -> int:
    """``unigym asset catalog search`` -- the sorted ids of one ``find``."""
    json_mode = bool(getattr(a, "json", False))
    filters = _search_filters(a)
    try:
        document = _fresh_catalog()
        ids = asset_catalog.find(
            document, kind=filters["kind"], tags=filters["tags"],
            pack=filters["pack"], uses=filters["uses"])
    except (AssetCliError, asset_catalog.CatalogError) as exc:
        if json_mode:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"[unigym asset] {exc}")
        return 1
    if json_mode:
        print(json.dumps({"ok": True, "count": len(ids), "ids": ids,
                          "filters": filters}))
        return 0
    for asset_id in ids:
        print(asset_id)
    print(f"{len(ids)} asset(s)")
    return 0


def _known_ids(document: dict) -> list[str]:
    """Every catalog id (sorted by the builder)."""
    return [entry["id"] for entry in document.get("entries", ())]


def _find_entry(document: dict, asset_id: str) -> dict:
    """Return one catalog entry; an unknown id fails closed with near ids."""
    if not is_asset_id(asset_id):
        raise AssetCliError(
            f"{asset_id!r} is not an asset id; write 'pack:<ns>/<path>' "
            "(e.g. pack:robocasa/assets/generators/objects/x/y) or "
            "'pack:<ns>/generator/<name>'")
    for entry in document.get("entries", ()):
        if entry["id"] == asset_id:
            return entry
    close = get_close_matches(asset_id, _known_ids(document),
                              n=NEAR_MATCH_LIMIT, cutoff=0.6)
    lines = [f"unknown asset id {asset_id!r}"]
    if close:
        lines.append("  near matches: " + ", ".join(close))
    lines.append("  hint: `unigym asset catalog search --json` lists the "
                 "catalog ids; a renamed asset answers under its current id "
                 "(an alias is a tombstone, v086 R8)")
    raise AssetCliError("\n".join(lines))


def _pack_root(namespace: str, asset_id: str) -> Path:
    """The mounted pack's root directory for one id namespace.

    An asset id path is relative to the pack root (``pack:<ns>/assets/...``
    for the ``assets`` face; the collector checks id against directory the
    same way), so the manifest sits at ``<root>/<id path>/asset.yaml``.
    """
    from unigym.packs import pack_roots

    mounts = {pack.name: pack for pack in pack_roots()}
    pack = mounts.get(namespace)
    if pack is None:
        raise AssetCliError(
            f"{asset_id}: no mounted pack owns namespace {namespace!r}; "
            f"mounted packs: {sorted(mounts)} -- mount the pack "
            "(UNIGYM_PACKS or <deps_root>/unigym_packs) or fix the id")
    if pack.faces.get("assets") is None:
        present = sorted(name for name, path in pack.faces.items() if path)
        raise AssetCliError(
            f"{asset_id}: pack {namespace!r} declares no 'assets' face; "
            f"faces present: {present}")
    return Path(pack.root) if pack.root is not None else Path(repo_root())


def _payload_role(document: Mapping, asset_id: str) -> str:
    """The ``files`` role that holds this file asset's primary payload."""
    kind = document.get("kind")
    files = document.get("files") or {}
    role = PAYLOAD_ROLES.get(kind)
    if role is not None:
        if role not in files:
            raise AssetCliError(
                f"{asset_id}: kind {kind!r} declares no files.{role}; the "
                f"manifest ships roles {sorted(files)} -- a {kind} asset "
                f"carries its payload under files.{role}")
        return role
    if len(files) == 1:
        return next(iter(files))
    raise AssetCliError(
        f"{asset_id}: kind {kind!r} has no single payload role; the "
        f"manifest ships roles {sorted(files)} -- `asset file` answers "
        f"file assets of kind {sorted(PAYLOAD_ROLES)}")


def _payload_base(document: Mapping, asset_dir: Path) -> Path:
    """The directory a payload path resolves against.

    A ``texture_set`` is a stem-named index directory *beside* its image:
    ``tools/emit_texture_sets.py`` writes manifests only and never moves a
    texture, so its ``files.texture`` basename resolves one level up.  Every
    other asset keeps the manifest rule -- paths are relative to the asset
    directory.
    """
    if document.get("kind") == "texture_set":
        return asset_dir.parent
    return asset_dir


def _payload_path(entry: dict) -> Path:
    """Resolve one file asset's primary payload to an absolute path.

    The id path is relative to the owning pack root (v086 R0), so the
    manifest sits at ``<root>/<id path>/asset.yaml`` and the payload path is
    relative to that directory.
    """
    asset_id = entry["id"]
    namespace, path = split_asset_id(asset_id)
    asset_dir = _pack_root(namespace, asset_id) / path
    manifest_path = asset_dir / "asset.yaml"
    try:
        document = load_manifest(manifest_path)
    except ManifestError as exc:
        raise AssetCliError(
            f"{exc} (the catalog lists {asset_id!r} but its manifest is "
            "unreadable; check the pack mount and rebuild the catalog)") from exc
    role = _payload_role(document, asset_id)
    value = document["files"][role]
    if not isinstance(value, str):
        raise AssetCliError(
            f"{asset_id}: files.{role} declares {len(value)} paths; the "
            f"primary payload must be one path -- fix the manifest")
    payload = (_payload_base(document, asset_dir) / value).resolve()
    if not payload.is_file():
        raise AssetCliError(
            f"{asset_id}: declared payload {payload} is missing; the pack "
            "payload is incomplete -- reprovision "
            "(python tools/pack_asset_closure.py) or run `unigym pack doctor`")
    return payload


def _generator_template(name: str) -> tuple[str, list[str]]:
    """Resolve a generator name to ``(qualified template, param keys)``.

    The template registry is the single truth for a computed payload: the
    registered name keys the builder and the callable's module names the
    pack that owns it.  The mounted code face is imported first (the closed
    registry fills as an import side effect).
    """
    from unigym.packs.load import load_mounted_code
    from unigym.props import templates

    load_mounted_code()
    if name not in templates._TEMPLATES:
        raise AssetCliError(
            f"generator {name!r} is catalogued but no template is "
            f"registered under that name; registered templates: "
            f"{sorted(templates._TEMPLATES)} -- rebuild the catalog after "
            "fixing the registration")
    schema, builder = templates._TEMPLATES[name]
    module = getattr(builder, "__module__", "?")
    qualname = getattr(builder, "__qualname__",
                       getattr(builder, "__name__", "?"))
    return f"{module}:{qualname}", sorted(schema)


def _print_generator(asset_id: str) -> int:
    """Print the registered template behind a generator asset id."""
    _namespace, path = split_asset_id(asset_id)
    name = path[len("generator/"):]
    try:
        template, params = _generator_template(name)
    except AssetCliError as exc:
        print(f"[unigym asset] {exc}")
        return 1
    print(f"generator: {asset_id}")
    print(f"  template: {template}")
    print(f"  params: {', '.join(params) if params else '(none)'}")
    return 0


def cmd_file(a, _extra=None) -> int:
    """``unigym asset file <id>`` -- the absolute payload path of one asset."""
    asset_id = a.id
    try:
        document = _fresh_catalog()
        entry = _find_entry(document, asset_id)
    except (AssetCliError, asset_catalog.CatalogError) as exc:
        print(f"[unigym asset] {exc}")
        return 1
    if entry.get("carrier") == "generator":
        return _print_generator(asset_id)
    try:
        payload = _payload_path(entry)
    except AssetCliError as exc:
        print(f"[unigym asset] {exc}")
        return 1
    print(payload)
    return 0


def _load_mapping(path: Path, label: str) -> dict:
    """Read one yaml mapping, failing closed with the path and the fix."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AssetCliError(f"{path}: cannot read the {label} ({exc})") from exc
    except yaml.YAMLError as exc:
        raise AssetCliError(
            f"{path}: the {label} is not valid yaml ({exc}); fix the file") from exc
    if not isinstance(document, Mapping):
        raise AssetCliError(
            f"{path}: the {label} must be a yaml mapping, got "
            f"{type(document).__name__}")
    return dict(document)


def _check_one(document: dict, spec_path: Path) -> tuple[list[str], list[str]]:
    """Resolve one spec and verify its lock through the decision layer.

    The resolution rule lives in :mod:`unigym.assets.selection` (the L0
    decision layer): this command loads the spec and the lock and asks
    :func:`~unigym.assets.selection.check_lock` to re-derive the ids, so the
    CLI can never disagree with the writer of the lock.

    Returns:
        ``(resolved ids, problems)``; an empty problem list means the lock's
        catalog digest and resolved ids both match a fresh resolution.
    """
    from unigym.assets import selection

    try:
        spec = selection.load_spec(spec_path)
    except selection.SelectionError as exc:
        return [], [str(exc)]
    lock_path = spec_path.with_suffix(".lock")
    if not lock_path.is_file():
        return [], [
            f"{lock_path} is missing; every selection pins its resolved ids "
            "in a lock beside the spec (v086 §6)"]
    try:
        lock = _load_mapping(lock_path, "selection lock")
        return selection.check_lock(lock, document, spec), []
    except (AssetCliError, selection.SelectionError) as exc:
        return [], [str(exc)]


def _orphan_locks(directory: Path, specs: list[Path]) -> list[str]:
    """Locks with no spec beside them: a pin nothing re-derives."""
    stems = {path.stem for path in specs}
    return [
        f"{lock}: lock has no spec beside it -- delete it or restore "
        f"{lock.with_suffix('.yaml')}"
        for lock in sorted(directory.glob("*.lock"))
        if lock.stem not in stems
    ]


def _selection_dirs(explicit: str | None) -> list[Path]:
    """The selection faces to check: ``--dir`` or every mounted pack's."""
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        if not directory.is_dir():
            raise AssetCliError(
                f"{directory}: no selection directory; point --dir at the "
                "pack's selection face (every *.yaml spec sits beside its "
                "*.lock)")
        return [directory]
    from unigym.packs import pack_roots

    found: list[Path] = []
    for pack in pack_roots():
        root = Path(pack.root) if pack.root is not None else Path(repo_root())
        candidate = root / "selection"
        if candidate.is_dir():
            found.append(candidate)
    return found


def cmd_selection_check(a, _extra=None) -> int:
    """``unigym asset selection check`` -- resolve specs, verify the locks."""
    try:
        directories = _selection_dirs(getattr(a, "dir", None))
    except AssetCliError as exc:
        print(f"[unigym asset] {exc}")
        return 1
    if not directories:
        print("[unigym asset] selection: no mounted pack carries a 'selection' "
              "face; nothing to verify (pass --dir DIR to check one)")
        return 0
    checked = 0
    failing = 0
    try:
        document: dict | None = None
        for directory in directories:
            specs = sorted(directory.glob("*.yaml"))
            for spec_path in specs:
                if document is None:
                    document = _fresh_catalog()
                checked += 1
                resolved, problems = _check_one(document, spec_path)
                if problems:
                    failing += 1
                    print(f"DRIFT {spec_path}")
                    for problem in problems:
                        print(f"      {problem}")
                else:
                    print(f"ok    {spec_path.name}: {len(resolved)} id(s)")
            for line in _orphan_locks(directory, specs):
                failing += 1
                print(f"DRIFT {line}")
    except (AssetCliError, asset_catalog.CatalogError) as exc:
        print(f"[unigym asset] {exc}")
        return 1
    if failing:
        print(f"[unigym asset] selection: FAIL ({failing} problem(s) over "
              f"{checked} spec(s))")
        return 1
    print(f"[unigym asset] selection: OK ({checked} spec(s) pinned)")
    return 0


def add_asset_subparser(sub) -> None:
    """Register the ``asset`` command group on the launcher parser."""
    asset = sub.add_parser("asset", help="L0 asset primitive: the derived "
                                        "catalog, payload paths and the "
                                        "selection locks")
    asset_sub = asset.add_subparsers(dest="asset_cmd", required=True)

    catalog = asset_sub.add_parser("catalog", help="the derived fact index "
                                                  "(build / search)")
    catalog_sub = catalog.add_subparsers(dest="catalog_cmd", required=True)
    build = catalog_sub.add_parser(
        "build", help="scan every mounted pack, write catalog.json + "
                      "catalog.digest, print counts and digest")
    build.add_argument("--out", default=None,
                       help="output directory (default: "
                            "${UNIGYM_DEPS_ROOT}/unigym_asset_catalog)")
    build.set_defaults(fn=cmd_catalog_build)

    search = catalog_sub.add_parser(
        "search", help="the sorted asset ids matching every filter (AND)")
    search.add_argument("--kind", default=None,
                        help="manifest kind (model | texture_set | mesh | "
                             "material)")
    search.add_argument("--tag", action="append", default=None, metavar="TAG",
                        help="require this namespaced tag (repeatable, AND)")
    search.add_argument("--pack", default=None,
                        help="require this pack namespace")
    search.add_argument("--uses", action="append", default=None, metavar="ID",
                        help="require this used asset id (repeatable, AND)")
    search.add_argument("--json", action="store_true",
                        help="single JSON document on stdout (errors too)")
    search.set_defaults(fn=cmd_catalog_search)

    file_parser = asset_sub.add_parser(
        "file", help="absolute payload path of a file asset; a generator id "
                     "prints its registered template")
    file_parser.add_argument("id", help="asset id (pack:<ns>/<path>)")
    file_parser.set_defaults(fn=cmd_file)

    selection = asset_sub.add_parser("selection", help="the decision layer: "
                                                      "spec -> lock checks")
    selection_sub = selection.add_subparsers(dest="selection_cmd",
                                             required=True)
    check = selection_sub.add_parser(
        "check", help="resolve every *.yaml spec against the catalog and "
                      "verify its .lock (digest + resolved ids; exit 1 on "
                      "drift)")
    check.add_argument("--dir", default=None,
                       help="selection face directory (default: every "
                            "mounted pack's selection/ directory)")
    check.set_defaults(fn=cmd_selection_check)


__all__ = [
    "AssetCliError",
    "add_asset_subparser",
    "cmd_catalog_build",
    "cmd_catalog_search",
    "cmd_file",
    "cmd_selection_check",
]
