"""Stub policy: fakes stay under _support, are manifested and do not leak mocks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_FAKES_ROOT = _TESTS_ROOT / "_support" / "fakes"
_SUPPORT_ROOT = _TESTS_ROOT / "_support"
_CONFORMANCE_ROOT = _TESTS_ROOT / "conformance"
_FAKES_README = _FAKES_ROOT / "README.md"
_PACKAGE_FILES = frozenset({"__init__.py"})
_MANIFEST_ROW = re.compile(r"^\|\s*`([^`]+\.py)`\s*\|", re.MULTILINE)


def _python_files(root: Path):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_under(path: Path, root: Path) -> bool:
    return root == path or root in path.parents


def _manifest_modules() -> set[str]:
    text = _FAKES_README.read_text(encoding="utf-8")
    if "## Manifest" not in text:
        return set()
    return set(_MANIFEST_ROW.findall(text))


def _fake_offenses(path: Path) -> list[str]:
    offenses = []
    if path.stem.lower().startswith(("fake", "stub")):
        offenses.append(f"{path}: fake-named module")
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ClassDef) and (
            "fake" in node.name.lower() or "stub" in node.name.lower()
        ):
            offenses.append(f"{path}: class {node.name}")
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_function and node.name.lower().startswith(("fake_", "stub_")):
            offenses.append(f"{path}: function {node.name}")
    return offenses


def _mock_offenses(path: Path) -> list[str]:
    offenses = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "unittest.mock" or alias.name.startswith("unittest.mock."):
                    offenses.append(f"{path}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "unittest.mock" or module.startswith("unittest.mock."):
                offenses.append(f"{path}: from {module}")
            if module == "unittest" and any(alias.name == "mock" for alias in node.names):
                offenses.append(f"{path}: from unittest import mock")
        elif isinstance(node, ast.Attribute):
            if (
                node.attr == "mock"
                and isinstance(node.value, ast.Name)
                and node.value.id == "unittest"
            ):
                offenses.append(f"{path}: unittest.mock attribute")
        elif isinstance(node, ast.Name) and node.id == "MagicMock":
            offenses.append(f"{path}: MagicMock reference")
    return offenses


def _production_import_offenses(path: Path) -> list[str]:
    offenses = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "unigym" and any(part.startswith("_") for part in parts[1:]):
                    offenses.append(f"{path}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if parts[0] == "unigym" and any(part.startswith("_") for part in parts[1:]):
                offenses.append(f"{path}: from {node.module}")
            if parts[0] == "unigym":
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("_"):
                        offenses.append(f"{path}: from {node.module} import {alias.name}")
    return offenses


def test_fake_modules_live_only_under_support_fakes():
    """No fake/stub module, class or function lives outside _support/fakes."""
    offenders = []
    for path in _python_files(_TESTS_ROOT):
        if _is_under(path, _FAKES_ROOT):
            continue
        offenders.extend(_fake_offenses(path))
    assert not offenders, f"fakes outside tests/_support/fakes: {offenders}"


def test_every_fake_module_has_a_manifest_entry():
    """The fakes README manifest names every fake module, and no stale module."""
    actual = {path.name for path in _python_files(_FAKES_ROOT) if path.name not in _PACKAGE_FILES}
    manifest = _manifest_modules()
    assert actual, "no fake modules found under tests/_support/fakes"
    assert actual <= manifest, f"fake modules missing a manifest entry: {sorted(actual - manifest)}"
    assert manifest <= actual, f"manifest entries without a module: {sorted(manifest - actual)}"


def test_no_unittest_mock_or_magicmock_outside_support():
    """Tests outside _support use real objects or typed fakes, never mocks."""
    offenders = []
    for path in _python_files(_TESTS_ROOT):
        if _is_under(path, _SUPPORT_ROOT):
            continue
        offenders.extend(_mock_offenses(path))
    assert not offenders, f"mocks outside tests/_support: {offenders}"


def test_conformance_imports_no_private_production_names():
    """Conformance tests import only public production surfaces."""
    offenders = []
    for path in _python_files(_CONFORMANCE_ROOT):
        offenders.extend(_production_import_offenses(path))
    assert not offenders, f"private production imports in conformance: {offenders}"


def test_conformance_imports_the_fake_only_from_support_backend():
    """The conformance fake is imported from tests._support.fakes.backend only."""
    offenders = []
    for path in _python_files(_CONFORMANCE_ROOT):
        for node in ast.walk(_parse(path)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if module.startswith("tests._support.fakes") and (
                    module != "tests._support.fakes.backend"
                ):
                    offenders.append(f"{path}: {module}")
                if module == "tests._support" and isinstance(node, ast.ImportFrom):
                    if any(alias.name == "fakes" for alias in node.names):
                        offenders.append(f"{path}: from tests._support import fakes")
    assert not offenders, f"unexpected fake imports in conformance: {offenders}"
