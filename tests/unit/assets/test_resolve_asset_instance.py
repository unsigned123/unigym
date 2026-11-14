"""File-backed asset instance resolution: format, path and closed failures.

The plan boundary is stubbed (no pack runtime starts); every resolved
reference names a real file under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.assets.plan import AssetPlan, resolve_asset_instance


class _PathContext:
    """AssetContext boundary that resolves exactly the references it is given."""

    def __init__(self, resolutions: dict[str, str]) -> None:
        self._resolutions = resolutions
        self.requests: list[str] = []

    def resolve(self, value: str) -> str:
        """Record one boundary request and return its mapped path."""
        self.requests.append(value)
        try:
            return self._resolutions[value]
        except KeyError:
            raise AssertionError(f"unexpected asset reference {value!r}") from None


def _plan(resolutions: dict[str, str]) -> AssetPlan:
    """Build an asset plan over an explicit reference-to-path boundary."""
    return AssetPlan(_PathContext(resolutions))


def _files(tmp_path: Path) -> dict[str, str]:
    """Write one existing file per asset format and return the paths."""
    files = {fmt: str(tmp_path / f"cup.{fmt}") for fmt in ("usd", "urdf", "mjcf")}
    for path in files.values():
        Path(path).write_text("asset\n", encoding="utf-8")
    return files


def _identity(files: dict[str, str]) -> dict[str, str]:
    """A boundary that returns each existing path unchanged."""
    return {path: path for path in files.values()}


@pytest.mark.parametrize(
    ("formats", "expected"),
    [(("usd", "mjcf"), "usd"), (("mjcf", "usd"), "mjcf")],
    ids=["usd-first", "mjcf-first"],
)
def test_resolve_asset_instance_follows_the_backend_preference_order(
    tmp_path: Path, formats: tuple[str, ...], expected: str
) -> None:
    """The first declared format in backend order wins, not the AssetRef order."""
    files = _files(tmp_path)
    asset = AssetRef(mjcf=files["mjcf"], usd=files["usd"])

    fmt, path = resolve_asset_instance(asset, _plan(_identity(files)), formats)

    assert fmt == expected
    assert path == files[expected]


def test_resolve_asset_instance_rejects_unsupported_only_declarations(
    tmp_path: Path,
) -> None:
    """An unsupported-only declaration fails naming the backend formats."""
    files = _files(tmp_path)
    asset = AssetRef(urdf=files["urdf"])

    with pytest.raises(AssetRefError, match="no compatible format") as info:
        resolve_asset_instance(asset, _plan(_identity(files)), ("usd", "mjcf"))

    assert "['usd', 'mjcf']" in str(info.value)


def test_resolve_asset_instance_rejects_an_empty_preference_list(tmp_path: Path) -> None:
    """An empty backend preference list is a teaching AssetRefError."""
    files = _files(tmp_path)

    with pytest.raises(AssetRefError, match="must not be empty"):
        resolve_asset_instance(AssetRef(usd=files["usd"]), _plan(_identity(files)), ())


def test_resolve_asset_instance_rejects_an_asset_reference_without_a_format() -> None:
    """An AssetRef that declares no format fails naming the backend formats."""
    with pytest.raises(AssetRefError, match="no compatible format") as info:
        resolve_asset_instance(AssetRef(), _plan({}), ("usd", "mjcf"))

    assert "['usd', 'mjcf']" in str(info.value)


def test_resolve_asset_instance_rejects_a_whitespace_only_selected_path() -> None:
    """A whitespace-only selected path fails naming the selected format."""
    with pytest.raises(
        AssetRefError,
        match="selected format 'usd' with an empty or whitespace-only path",
    ):
        resolve_asset_instance(AssetRef(usd="   "), _plan({}), ("usd",))


def test_resolve_asset_instance_rejects_a_missing_file(tmp_path: Path) -> None:
    """An absolute reference that names no file fails naming the ref and format."""
    missing = str(tmp_path / "absent.mjcf")
    asset = AssetRef(mjcf=missing)

    with pytest.raises(FileNotFoundError) as info:
        resolve_asset_instance(asset, _plan({missing: missing}), ("mjcf",))

    message = str(info.value)
    assert missing in message
    assert "mjcf" in message


def test_resolve_asset_instance_does_not_fall_back_to_another_format(
    tmp_path: Path,
) -> None:
    """A missing preferred format fails instead of selecting a later format."""
    files = _files(tmp_path)
    missing = str(tmp_path / "absent.usd")
    resolutions = _identity(files)
    resolutions[missing] = missing
    asset = AssetRef(usd=missing, mjcf=files["mjcf"])

    with pytest.raises(FileNotFoundError) as info:
        resolve_asset_instance(asset, _plan(resolutions), ("usd", "mjcf"))

    assert missing in str(info.value)


def test_resolve_asset_instance_returns_the_boundary_path_unchanged(tmp_path: Path) -> None:
    """An absolute path the plan boundary resolved is returned unchanged."""
    files = _files(tmp_path)
    target = files["mjcf"]

    fmt, path = resolve_asset_instance(AssetRef(mjcf=target), _plan({target: target}), ("mjcf",))

    assert (fmt, path) == ("mjcf", target)
    assert Path(path).is_absolute()


def test_resolve_asset_instance_rejects_a_relative_boundary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CWD-relative boundary resolution fails instead of succeeding via the CWD."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cup.mjcf").write_text("asset\n", encoding="utf-8")
    asset = AssetRef(mjcf="cup.mjcf")

    with pytest.raises(AssetRefError, match="relative") as info:
        resolve_asset_instance(asset, _plan({"cup.mjcf": "cup.mjcf"}), ("mjcf",))

    message = str(info.value)
    assert "cup.mjcf" in message
    assert "mjcf" in message


def test_resolve_asset_instance_delegates_a_pack_reference_to_the_boundary(
    tmp_path: Path,
) -> None:
    """A ``pack:`` reference reaches the boundary verbatim and resolves once."""
    target = tmp_path / "cup.mjcf"
    target.write_text("asset\n", encoding="utf-8")
    ref = "pack:acme/models/cup.mjcf"
    context = _PathContext({ref: str(target), str(target): str(target)})

    fmt, path = resolve_asset_instance(AssetRef(mjcf=ref), AssetPlan(context), ("mjcf",))

    assert (fmt, path) == ("mjcf", str(target))
    assert context.requests[0] == ref


def test_resolve_asset_instance_expands_a_known_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``${UNIGYM_DEPS_ROOT}`` reference expands before the boundary resolves it."""
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path))
    target = tmp_path / "cup.xml"
    target.write_text("asset\n", encoding="utf-8")

    fmt, path = resolve_asset_instance(
        AssetRef(mjcf="${UNIGYM_DEPS_ROOT}/cup.xml"),
        _plan({str(target): str(target)}),
        ("mjcf",),
    )

    assert (fmt, path) == ("mjcf", str(target))
