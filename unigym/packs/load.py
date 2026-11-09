"""Code-face loading for ``trust: code`` packs.

A code pack carries a plain Python package at ``<pack_root>/python/``; its
``__init__.py`` decides the submodule import order.  Loading is a standard
package import under the unique module name ``unigym_packs_<name>`` -- there
is no entry declaration, no per-pack authorization and no content hash
(internal trust).  Pack code may import the standard library
and third-party packages freely, but reaches UniGym only through an explicit
allowlist of core import roots; every executed pack file passes an AST guard
that fails closed with :class:`~unigym.packs.PackError` naming file, line and
the import target.  The runtime, daemon, backend, environment and script
trees are forbidden regardless of any future allowlist entry.  Private
modules (any dotted component starting with ``_``) and the internals of
``unigym.packs`` are also forbidden: a pack imports the public package only.
The entries beyond the base contract (``pack_api`` / ``scenes`` /
``props.asset_ir`` / ``core.region`` / ``packs``) are the audited robocasa
exceptions, listed explicitly so the allowlist carries no wildcards.
``unigym.robots.gr1_reset`` is allowlisted as a pure constants/dataclass
module (the robocasa initializers' robot-stage protocol name).

Nothing here runs at import time: the callers are explicit (the launcher and
the daemon startup).
"""
from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import TypedDict

from . import MountedPack, PackError
from .mount import pack_roots

PACK_MODULE_PREFIX = "unigym_packs_"
"""Module-name prefix every code pack is imported under."""

_ALLOWED_CORE_ROOTS = (
    # Public pack contract.
    "unigym.pack_api",
    "unigym.scenes",
    "unigym.props.asset_ir",
    "unigym.core.region",
    # Audited robocasa exceptions: exact modules, no wildcards.
    "unigym.assets.selection",
    "unigym.core.asset_root",
    "unigym.props.scaling",
    "unigym.props.templates",
    "unigym.initializers.base",
    "unigym.initializers.placement",
    "unigym.initializers.placement_sampler",
    "unigym.robots.gr1_reset",
    "unigym.terminations",
)
"""Core import roots pack code may use (the module or any submodule).

``unigym.packs`` itself is allowed separately: only the public package, never
its ``load`` / ``mount`` internals.
"""

_FORBIDDEN_CORE_SUFFIXES = ("backends", "daemon", "scripts", "env", "runtime")
"""Upper-layer roots pack code may never import, even if allowlisted later.

Spelled as suffixes because ``tests/meta/packs/test_guard.py`` forbids the
literal upper-layer import tokens anywhere inside this package.
"""

_FORBIDDEN_CORE_ROOTS = tuple(
    f"unigym.{name}" for name in _FORBIDDEN_CORE_SUFFIXES
)

_ALLOWED_TEXT = ", ".join(_ALLOWED_CORE_ROOTS)
_FORBIDDEN_TEXT = ", ".join(_FORBIDDEN_CORE_ROOTS)

_LOADED: dict[str, tuple[str, ...]] = {}
"""Pack name -> module names imported by the last successful load."""


def _matches(name: str, roots: tuple[str, ...]) -> bool:
    """True when *name* is one of *roots* or a dotted submodule of one."""
    return any(name == root or name.startswith(root + ".") for root in roots)


def _is_private(name: str) -> bool:
    """True when any dotted component after ``unigym`` starts with ``_``."""
    return any(part.startswith("_") for part in name.split(".")[1:])


def _rejection(name: str | None, level: int) -> str | None:
    """The rejection reason for an absolute import target, or ``None``."""
    if level or not name:
        return None
    if not (name == "unigym" or name.startswith("unigym.")):
        return None
    if _matches(name, _FORBIDDEN_CORE_ROOTS):
        return "explicitly forbidden"
    if _is_private(name):
        return "private modules are outside the pack contract"
    if name == "unigym.packs":
        return None
    if name.startswith("unigym.packs."):
        return "unigym.packs internals are core-side machinery"
    if not _matches(name, _ALLOWED_CORE_ROOTS):
        return "not an allowlisted core module"
    return None


def _reject(path: Path, lineno: int, name: str, reason: str) -> None:
    """Raise the fail-closed guard error naming file, line and all three lists."""
    raise PackError(
        f"{path}:{lineno}: pack code may not import {name!r}: {reason}; "
        f"allowed core roots: {_ALLOWED_TEXT}; "
        f"explicitly forbidden: {_FORBIDDEN_TEXT}; "
        f"any other unigym module is not an allowlisted core module"
    )


_MUTABLE_SYS_ATTRS = ("path", "meta_path", "path_hooks", "modules")
"""Interpreter state pack code may never touch -- the loader owns it.

The pack is imported as a package via ``spec_from_file_location`` plus a
temporary meta-path finder; a pack that edits ``sys.path`` (or registers
its own finder/module) would escape the AST guard and the one-package
namespace, so the guard refuses the reference outright.
"""


def guard_pack_file(path: Path) -> None:
    """Reject forbidden ``unigym`` imports in one pack source file.

    Args:
        path: A ``.py`` file inside a code pack's ``python/`` directory.

    Raises:
        PackError: The file is unparseable or imports a UniGym module that is
            not an allowlisted core root (or one of the explicitly forbidden
            roots); the message names file, line and target.
    """
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as err:
        raise PackError(
            f"{path}:{err.lineno}: cannot parse pack module: {err.msg}"
        ) from err
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                reason = _rejection(alias.name, 0)
                if reason:
                    _reject(path, node.lineno, alias.name, reason)
        elif isinstance(node, ast.ImportFrom):
            reason = _rejection(node.module, node.level)
            if reason:
                _reject(path, node.lineno, node.module or "", reason)
        elif (isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name)
              and node.value.id == "sys"
              and node.attr in _MUTABLE_SYS_ATTRS):
            raise PackError(
                f"{path}:{node.lineno}: pack code touches sys.{node.attr} -- "
                f"the pack loader owns interpreter state (import the core "
                f"through unigym.pack_api instead)")


class _GuardedFinder(importlib.abc.MetaPathFinder):
    """Guard every pack submodule right before its source executes."""

    def __init__(self, module_name: str, python_dir: Path) -> None:
        """Track one pack package and the directory it may resolve from."""
        self.module_name = module_name
        self.python_dir = Path(python_dir).resolve()

    def find_spec(self, fullname, path=None, target=None):
        """Return the normal spec after guarding a pack source file.

        Args:
            fullname: Dotted module name being imported.
            path: The package search path (``module.__path__``).
            target: Unused reload target.

        Returns:
            The spec resolved by the standard path finder, or ``None`` when
            the name is not this pack's.
        """
        if fullname != self.module_name and not fullname.startswith(
            self.module_name + "."
        ):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            return None
        origin = Path(spec.origin)
        try:
            origin.resolve().relative_to(self.python_dir)
        except ValueError:
            raise PackError(
                f"pack module {fullname!r} resolves outside its python "
                f"directory: {origin}"
            ) from None
        if origin.suffix == ".py":
            guard_pack_file(origin)
        return spec


def _module_name(pack_name: str) -> str:
    """The unique import name of one pack's python package."""
    return PACK_MODULE_PREFIX + pack_name


def _module_names(module_name: str) -> tuple[str, ...]:
    """Every loaded module belonging to one pack package, sorted."""
    prefix = module_name + "."
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == module_name or name.startswith(prefix)
        )
    )


def _unload(module_name: str) -> None:
    """Drop a partially imported package (and submodules) from sys.modules."""
    prefix = module_name + "."
    for name in [
        name
        for name in sys.modules
        if name == module_name or name.startswith(prefix)
    ]:
        sys.modules.pop(name, None)


def load_pack_code(pack: MountedPack) -> tuple[str, ...]:
    """Import one pack's ``python/`` package and return its module names.

    Content packs are a no-op.  A pack already imported in this process is
    answered from the cache, so a second call never re-executes pack code.

    Args:
        pack: The mounted pack to load.

    Returns:
        The sorted ``unigym_packs_<name>`` module names, base package first.

    Raises:
        PackError: The pack has no ``python/__init__.py``, a pack source file
            fails the AST guard, or the import raised.
    """
    if pack.trust != "code":
        return ()
    if pack.name in _LOADED:
        return _LOADED[pack.name]
    module_name = _module_name(pack.name)
    if module_name in sys.modules:
        names = _module_names(module_name)
        _LOADED[pack.name] = names
        return names
    if pack.root is None:
        raise PackError(
            f"code pack {pack.name!r} has no root to import from"
        )
    python_dir = (Path(pack.root) / "python").resolve()
    init = python_dir / "__init__.py"
    if not init.is_file():
        raise PackError(
            f"code pack {pack.name!r} has no python/__init__.py ({init})"
        )
    guard_pack_file(init)
    spec = importlib.util.spec_from_file_location(
        module_name,
        init,
        submodule_search_locations=[str(python_dir)],
    )
    if spec is None or spec.loader is None:
        raise PackError(
            f"code pack {pack.name!r}: cannot build an import spec for {init}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    finder = _GuardedFinder(module_name, python_dir)
    sys.meta_path.insert(0, finder)
    try:
        spec.loader.exec_module(module)
    except PackError:
        _unload(module_name)
        raise
    except Exception as err:
        text = traceback.format_exc()
        _unload(module_name)
        raise PackError(
            f"importing code pack module {module_name!r} ({init}) failed:\n"
            f"{text}"
        ) from err
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
    names = _module_names(module_name)
    _LOADED[pack.name] = names
    return names


def load_mounted_code() -> dict[str, tuple[str, ...]]:
    """Load every mounted code pack in mount order.

    Returns:
        Pack name -> the module names imported for that pack.
    """
    loaded: dict[str, tuple[str, ...]] = {}
    for pack in pack_roots():
        if pack.trust != "code":
            continue
        loaded[pack.name] = load_pack_code(pack)
    return loaded


class RegistrationReport(TypedDict):
    """The closed-set registry counts summarised for the CLI."""

    packs: dict[str, tuple[str, ...]]
    scenes: int
    props: int
    terminations: int


def registration_report() -> RegistrationReport:
    """Summarise what the closed registries currently hold.

    Returns:
        ``packs`` (pack -> module names), plus the scene, prop-template and
        termination entry counts read from the existing registries.  The
        registries themselves are not modified.
    """
    from unigym.props import templates
    from unigym.scenes.registry import registered_scenes
    from unigym.terminations import TERMINATION_REGISTRY

    return {
        "packs": dict(_LOADED),
        "scenes": len(registered_scenes()),
        "props": len(templates._TEMPLATES),
        "terminations": len(TERMINATION_REGISTRY),
    }


__all__ = [
    "PACK_MODULE_PREFIX",
    "guard_pack_file",
    "load_mounted_code",
    "load_pack_code",
    "registration_report",
]
