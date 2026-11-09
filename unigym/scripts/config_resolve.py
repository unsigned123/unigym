"""config_resolve.py -- explicit glob + short-name config resolution.

ONE helper shared by ``unigym run|submit|debug|probe``:

- absolute path: used VERBATIM (must exist);
- RELATIVE SPELLINGS ARE NEVER CWD-RESOLVED (owner ruling): they
  always anchor at ``config/`` in the current checkout (= the package
  root, worktree-safe);
- glob metachars (``*?[``) -> ``sorted(CONFIG_DIR.glob(rel))``
  anchored at ``config/`` (a leading ``config/`` in the spec is
  consumed, so repo-root spellings like ``config/g1pick/pk0*.yaml``
  work); empty expansion is an error + candidates;
- exact name relative to ``config/`` (``x.yaml`` or ``sub/x.yaml``,
  a leading ``config/`` consumed) -> used if that exact file exists;
  everything else -- bare stems, slash prefixes, wrong names -- is a
  hard ``config not found`` (per owner: no short names, no
  prefix matching, explicit only).

Discipline: glob requires its explicit trigger; name resolution is
EXACT ONLY -- nothing is ever silently reinterpreted as a pattern,
a prefix, or a name.

Stdlib only (pathlib); importable without isaaclab/mujoco/torch.
"""
from __future__ import annotations

from pathlib import Path

from unigym.core.asset_root import repo_root
from unigym.packs import MountedPack, find, pack_roots, repo_pack


def _repo_runs_face() -> Path:
    """The repo mount's ``runs`` face (the checkout's ``config/``)."""
    face = repo_pack().faces.get("runs")
    return face if face is not None else Path(repo_root()) / "config"


CONFIG_DIR = _repo_runs_face()
_GLOB_CHARS = ("*", "?", "[")
_MAX_CANDIDATES = 8                      # listed in error messages


class ConfigResolveError(ValueError):
    """A config spec resolved to nothing/many; message lists candidates."""


def resolve_config(spec: str) -> list[Path]:
    """Expand one CLI config spec to concrete path(s).  See module doc."""
    s = str(spec)
    if any(c in s for c in _GLOB_CHARS):
        return _glob_expand(s)
    p = Path(s).expanduser()
    if p.is_absolute():                  # absolute: the only verbatim form
        if not p.is_file():
            raise ConfigResolveError(f"config not found: {p}")
        return [p.resolve()]
    return _exact_expand(s)              # relative: always config/-anchored


def _runs_names() -> list[str]:
    """Candidate spellings from the mounted ``runs`` faces.

    Each path is relative to its owning face root.  A module constant
    replaced by a hermetic test tree keeps the literal ``rglob`` view of
    that tree (the ``runs`` face enumeration cannot see it).
    """
    packs = pack_roots()
    mounted = {runs.resolve() for pack in packs
               if (runs := pack.faces.get("runs")) is not None}
    if CONFIG_DIR.resolve() in mounted:
        return sorted(str(path.relative_to(runs))
                      for pack, path in find("runs", suffix=".yaml")
                      if (runs := pack.faces.get("runs")) is not None)
    return sorted(str(p.relative_to(CONFIG_DIR))
                  for p in CONFIG_DIR.rglob("*.yaml"))


def _candidates_line() -> str:
    names = _runs_names()
    head = ", ".join(names[:_MAX_CANDIDATES])
    more = "" if len(names) <= _MAX_CANDIDATES \
        else f" (+{len(names) - _MAX_CANDIDATES} more; GET /api/configs)"
    return head + more


def _glob_expand(spec: str) -> list[Path]:
    if Path(spec).is_absolute() or spec.startswith("~"):
        raise ConfigResolveError(
            f"glob config spec must be config-dir-relative: {spec!r}")
    rel = spec[2:] if spec.startswith("./") else spec
    prefix = CONFIG_DIR.name + "/"
    if rel.startswith(prefix):           # repo-root spelling -> config-anchored
        rel = rel[len(prefix):]
    hits = sorted(CONFIG_DIR.glob(rel))
    if not hits:
        raise ConfigResolveError(
            f"config glob {spec!r} matched nothing under {CONFIG_DIR} "
            f"(candidates: {_candidates_line()})")
    return [p.resolve() for p in hits]


def _exact_expand(spec: str) -> list[Path]:
    """Relative spec anchored at CONFIG_DIR: ``x.yaml`` or ``sub/x.yaml``.

    A leading ``config/`` is consumed (repo-root spelling).  Relative
    spellings are NEVER resolved against the CWD (Option B): the anchor
    is the package root, so behavior is CWD- and worktree-independent.
    An exact relative spelling that also exists on an external pack's
    ``runs`` face resolves to the external copy (the pack shadows the
    repository); two external hits stay an ambiguity error.  Bare stems /
    prefix forms / wrong names are hard errors listing candidates:
    exactness is the point.
    """
    rel = spec[2:] if spec.startswith("./") else spec
    prefix = CONFIG_DIR.name + "/"
    if rel.startswith(prefix):           # repo-root spelling -> config-anchored
        rel = rel[len(prefix):]
    external = _external_exact(rel)
    if len(external) > 1:
        listing = ", ".join(f"{pack.name}:{path}" for pack, path in external)
        raise ConfigResolveError(
            f"ambiguous config {spec!r}: found in {listing}; "
            f"use 'pack:<name>/{rel}' or an absolute path")
    if external:
        return [external[0][1].resolve()]
    p = CONFIG_DIR / rel
    if p.is_file():
        return [p.resolve()]
    raise ConfigResolveError(
        f"config not found: {spec!r} (exact name under {CONFIG_DIR} like "
        f"'props/drawer_scripted_mj.yaml', an explicit path, or a glob; "
        f"candidates: {_candidates_line()})")


def _external_exact(rel: str) -> list[tuple[MountedPack, Path]]:
    """Every external pack whose ``runs`` face holds ``rel`` (mount order)."""
    if not rel or Path(rel).is_absolute():
        return []
    hits: list[tuple[MountedPack, Path]] = []
    for pack in pack_roots():
        if pack.root is None:
            continue
        face = pack.faces.get("runs")
        if face is None:
            continue
        candidate = face / rel
        if not candidate.is_file():
            continue
        real = candidate.resolve()
        if face.resolve() not in real.parents:
            continue
        hits.append((pack, real))
    return hits
