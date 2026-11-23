"""Structural rewind-provider discovery over sink capabilities."""

from __future__ import annotations

import pytest

from unigym.runner.sinks.fanout import FanOutSink
from unigym.runtime.rewind_providers import build_rewind_providers


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name

    def rewind_state(self) -> dict:
        return {}

    def apply_rewind_state(self, state: dict) -> None:
        del state


class _IncompleteProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def rewind_state(self) -> dict:
        return {}


class _SupplierSink:
    """Sink whose ``rewind_provider`` is a callable supplier."""

    def __init__(self, provider: object) -> None:
        self._provider = provider
        self.supplier_calls = 0

    def rewind_provider(self) -> object:
        self.supplier_calls += 1
        return self._provider


class _AttributeSink:
    """Sink whose ``rewind_provider`` is a provider attribute."""

    def __init__(self, provider: object) -> None:
        self.rewind_provider = provider


class _DirectProviderWithChildren(_Provider):
    """A provider that also exposes ``sinks`` children."""

    def __init__(self, name: str, *children: object) -> None:
        super().__init__(name)
        self.sinks = list(children)


def test_direct_provider_is_discovered_as_itself():
    """A node satisfying the provider protocol is its own provider."""
    provider = _Provider("acme.direct")

    assert build_rewind_providers(provider) == [provider]


def test_direct_provider_wins_over_supplier_and_children():
    """Direct capability takes precedence over a supplier or child walk."""
    child = _Provider("acme.child")
    direct = _DirectProviderWithChildren("acme.direct", child)
    direct.rewind_provider = lambda: _Provider("acme.supplier")

    assert build_rewind_providers(direct) == [direct]


def test_rewind_provider_attribute_supplies_the_provider():
    """A non-callable ``rewind_provider`` attribute mounts its provider."""
    provider = _Provider("acme.attribute")
    sink = _AttributeSink(provider)

    assert build_rewind_providers(sink) == [provider]


def test_rewind_provider_method_supplies_the_provider():
    """A callable ``rewind_provider`` method supplies the provider once."""
    provider = _Provider("acme.method")
    sink = _SupplierSink(provider)

    providers = build_rewind_providers(sink)

    assert providers == [provider]
    assert sink.supplier_calls == 1


def test_nested_fanout_preserves_construction_order():
    """Depth-first recursion keeps construction order across nesting."""
    first = _Provider("acme.first")
    second = _AttributeSink(_Provider("acme.second"))
    third = _Provider("acme.third")
    sink = FanOutSink(FanOutSink(first), FanOutSink(second, FanOutSink(third)))

    providers = build_rewind_providers(sink)

    assert [provider.name for provider in providers] == [
        "acme.first",
        "acme.second",
        "acme.third",
    ]


def test_shared_child_is_discovered_once_by_identity():
    """A diamond re-visit does not mount the same node twice."""
    shared = _Provider("acme.shared")
    sink = FanOutSink(shared, FanOutSink(shared))

    providers = build_rewind_providers(sink)

    assert len(providers) == 1
    assert providers[0] is shared


def test_nodes_without_provider_capability_are_skipped():
    """Plain sinks and empty fan-outs contribute no provider."""
    assert build_rewind_providers(object()) == []
    assert build_rewind_providers(FanOutSink(object(), FanOutSink(object()))) == []


def test_duplicate_provider_names_fail_at_build():
    """Provider names are payload keys; duplicates fail with a teaching error."""
    sink = FanOutSink(_Provider("acme.pick"), _Provider("acme.pick"))

    with pytest.raises(ValueError, match="duplicate rewind provider names") as info:
        build_rewind_providers(sink)

    assert "must be unique" in str(info.value)


@pytest.mark.parametrize(
    ("supplied", "fragment"),
    [
        (_IncompleteProvider("acme.pick"), "apply_rewind_state"),
        (object(), "name"),
    ],
    ids=["missing-apply", "missing-name"],
)
def test_malformed_supplier_fails_with_a_teaching_error(supplied, fragment):
    """A declared supplier that yields a non-provider is rejected, not skipped."""
    sink = _SupplierSink(supplied)

    with pytest.raises(TypeError, match="rewind_provider must supply") as info:
        build_rewind_providers(sink)

    assert fragment in str(info.value)
