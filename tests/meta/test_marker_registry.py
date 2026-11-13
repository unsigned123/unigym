"""Marker registry and auto-tagging invariants for the test harness."""

from __future__ import annotations

import pytest

from tests._support.markers import (
    BACKEND_MARKER_PREFIX,
    BACKEND_MARKERS,
    FAST_LAYERS,
    LAYER_MARKERS,
    RESOURCE_MARKERS,
    UNIT_DOMAIN_MARKERS,
    UnknownTestPathError,
    backend_marker,
    is_fast_lane,
    layer_marker,
    markers_for_path,
    unit_domain_marker,
)

_BUILTIN_MARKERS = frozenset(
    {"parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings"}
)


def _marker_names(item):
    return {marker.name for marker in item.iter_markers()}


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("smoke", "test_import.py"), "layer_smoke"),
        (("unit", "core", "test_foo.py"), "layer_unit"),
        (("contract", "test_control_contract.py"), "layer_contract"),
        (("conformance", "test_foo.py"), "layer_conformance"),
        (("backends", "mujoco", "test_env.py"), "layer_backend"),
        (("integration", "test_cli.py"), "layer_integration"),
        (("meta", "test_marker_registry.py"), "layer_meta"),
    ],
)
def test_layer_mapping(parts, expected):
    assert layer_marker(parts) == expected


@pytest.mark.parametrize("domain_dir", sorted(UNIT_DOMAIN_MARKERS))
def test_unit_domain_mapping(domain_dir):
    parts = ("unit", domain_dir, "test_module.py")
    assert unit_domain_marker(parts) == UNIT_DOMAIN_MARKERS[domain_dir]
    assert markers_for_path(parts) == frozenset({"layer_unit", UNIT_DOMAIN_MARKERS[domain_dir]})


@pytest.mark.parametrize("backend_dir", sorted(BACKEND_MARKERS))
def test_backend_mapping(backend_dir):
    parts = ("backends", backend_dir, "test_env.py")
    assert backend_marker(parts) == BACKEND_MARKERS[backend_dir]
    assert markers_for_path(parts) == frozenset({"layer_backend", BACKEND_MARKERS[backend_dir]})


def test_non_unit_and_non_backend_paths_have_no_domain_or_backend_marker():
    assert unit_domain_marker(("contract", "test_foo.py")) is None
    assert backend_marker(("contract", "test_foo.py")) is None
    assert markers_for_path(("contract", "test_foo.py")) == frozenset({"layer_contract"})


def test_integration_path_adds_slow():
    assert markers_for_path(("integration", "test_cli.py")) == frozenset(
        {"layer_integration", "slow"}
    )


@pytest.mark.parametrize(
    "parts",
    [
        ("regression", "test_foo.py"),
        ("_support", "test_support.py"),
        ("unknown", "test_foo.py"),
        (),
    ],
)
def test_unknown_layer_raises_with_path_and_known_keys(parts):
    with pytest.raises(UnknownTestPathError) as excinfo:
        markers_for_path(parts)
    message = str(excinfo.value)
    assert "unknown test layer" in message
    path_text = "/".join(parts) or "<empty>"
    assert path_text in message
    for known in LAYER_MARKERS:
        assert known in message


@pytest.mark.parametrize(
    "parts",
    [
        ("unit", "unknown_domain", "test_foo.py"),
        ("unit", "test_foo.py"),
    ],
)
def test_unknown_unit_domain_raises_with_path_and_known_keys(parts):
    with pytest.raises(UnknownTestPathError) as excinfo:
        markers_for_path(parts)
    message = str(excinfo.value)
    assert "unknown unit domain" in message
    assert "/".join(parts) in message
    for known in UNIT_DOMAIN_MARKERS:
        assert known in message


@pytest.mark.parametrize(
    "parts",
    [
        ("backends", "gazebo", "test_env.py"),
        ("backends", "test_env.py"),
    ],
)
def test_unknown_backend_raises_with_path_and_known_keys(parts):
    with pytest.raises(UnknownTestPathError) as excinfo:
        markers_for_path(parts)
    message = str(excinfo.value)
    assert "unknown backend" in message
    assert "/".join(parts) in message
    for known in BACKEND_MARKERS:
        assert known in message


def test_resource_markers_cover_phase0_capability_and_deviation():
    assert RESOURCE_MARKERS == frozenset(
        {
            "slow",
            "gpu",
            "sim",
            "subprocess",
            "network",
            "serial",
            "requires_capabilities",
            "known_deviation",
            "requires_test_packs",
        }
    )


@pytest.mark.parametrize("layer", sorted(FAST_LAYERS))
def test_fast_layers_are_fast_without_resource_markers(layer):
    assert is_fast_lane(layer, {layer}) is True


@pytest.mark.parametrize("layer", sorted(set(LAYER_MARKERS.values()) - FAST_LAYERS))
def test_non_fast_layers_are_never_fast(layer):
    assert is_fast_lane(layer, {layer}) is False


@pytest.mark.parametrize("resource", sorted(RESOURCE_MARKERS))
def test_any_resource_marker_denies_fast(resource):
    assert is_fast_lane("layer_unit", {"layer_unit", resource}) is False


def test_every_item_has_exactly_one_layer_marker(request):
    """Every collected item carries exactly one directory-derived layer marker."""
    offenders = []
    for item in request.session.items:
        layers = sorted(name for name in _marker_names(item) if name.startswith("layer_"))
        if len(layers) != 1:
            offenders.append((item.nodeid, layers))
    assert not offenders, f"items without exactly one layer marker: {offenders}"


def test_every_item_has_exactly_one_backend_marker(request):
    """Every collected item carries exactly one backend marker, never a contradiction."""
    offenders = []
    for item in request.session.items:
        backends = sorted(
            name for name in _marker_names(item) if name.startswith(BACKEND_MARKER_PREFIX)
        )
        if len(backends) != 1:
            offenders.append((item.nodeid, backends))
    assert not offenders, f"items without exactly one backend marker: {offenders}"


def test_integration_items_are_slow(request):
    """Integration items are always tagged slow by the harness."""
    offenders = [
        item.nodeid
        for item in request.session.items
        if "layer_integration" in _marker_names(item) and "slow" not in _marker_names(item)
    ]
    assert not offenders, f"integration items missing slow: {offenders}"


def test_fast_items_are_not_integration_backend_or_resource(request):
    """A fast item is never integration/backend and carries no resource marker."""
    offenders = []
    for item in request.session.items:
        names = _marker_names(item)
        if "fast" not in names:
            continue
        if "layer_integration" in names or "layer_backend" in names:
            offenders.append(item.nodeid)
        elif names & RESOURCE_MARKERS:
            offenders.append(item.nodeid)
    assert not offenders, f"invalid fast items: {offenders}"


def test_used_markers_are_registered(request):
    """Every marker applied to a collected item is declared in pyproject.toml."""
    registered = {entry.split(":", 1)[0].strip() for entry in request.config.getini("markers")}
    used = set()
    for item in request.session.items:
        used.update(_marker_names(item))
    missing = sorted(used - registered - _BUILTIN_MARKERS)
    assert not missing, f"unregistered markers: {missing}"


def test_d0_lane_has_no_test_pack_dependencies(request):
    """The provision-free D0 lane never collects a pack-dependent test."""
    offenders = [
        item.nodeid
        for item in request.session.items
        if "requires_test_packs" in _marker_names(item)
    ]
    assert not offenders, f"D0 items requiring test packs: {offenders}"
