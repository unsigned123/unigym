"""Unit invariants for the shared provenance records and side tables."""

from __future__ import annotations

from unigym.provenance import (
    OriginTable,
    ProviderInfo,
    ProviderOrigin,
    RegistrySnapshot,
    attribution,
)


def test_provider_origin_defaults_and_constructors() -> None:
    """An absent origin is explicit; ``inline`` keeps its own kind."""
    assert ProviderOrigin.unknown() == ProviderOrigin(kind="unknown")
    assert ProviderOrigin.unknown().pack is None
    assert ProviderOrigin.inline() == ProviderOrigin(kind="inline")


def test_provider_origin_records_pack_manifest_and_seam_fields() -> None:
    """The record carries pack, locator, activation and entry-point-seam facts."""
    origin = ProviderOrigin(
        kind="manifest",
        pack="acme",
        pack_version="1.2.0",
        root="/packs/acme",
        discovered_by="entry-point",
        face="providers",
        item_id="acme.arm",
        resource="robots/acme_arm.yaml",
        entrypoint=None,
        activation_entrypoint="acme:register",
        location_imported=True,
    )

    assert origin.kind == "manifest"
    assert origin.resource == "robots/acme_arm.yaml"
    assert origin.location_imported is True
    assert origin == ProviderOrigin(
        kind="manifest",
        pack="acme",
        pack_version="1.2.0",
        root="/packs/acme",
        discovered_by="entry-point",
        face="providers",
        item_id="acme.arm",
        resource="robots/acme_arm.yaml",
        entrypoint=None,
        activation_entrypoint="acme:register",
        location_imported=True,
    )
    assert hash(origin) == hash(
        ProviderOrigin(
            kind="manifest",
            pack="acme",
            pack_version="1.2.0",
            root="/packs/acme",
            discovered_by="entry-point",
            face="providers",
            item_id="acme.arm",
            resource="robots/acme_arm.yaml",
            entrypoint=None,
            activation_entrypoint="acme:register",
            location_imported=True,
        )
    )


def test_origin_table_set_get_remove_and_unknown_fallback() -> None:
    """The side table stores, reports and forgets one origin per name."""
    table = OriginTable()
    origin = ProviderOrigin(kind="core")

    assert table.get("manual") is None
    assert table.origin("manual") == ProviderOrigin.unknown()

    table.set("manual", origin)
    assert table.get("manual") == origin
    assert table.origin("manual") == origin
    assert "manual" in table
    assert len(table) == 1

    table.remove("manual")
    assert table.get("manual") is None
    assert table.origin("manual") == ProviderOrigin.unknown()
    table.remove("manual")


def test_origin_table_rejects_a_non_origin() -> None:
    """The side table accepts only :class:`ProviderOrigin` values."""
    table = OriginTable()

    try:
        table.set("acme.arm", object())  # type: ignore[arg-type]
    except TypeError as info:
        assert "ProviderOrigin" in str(info)
    else:  # pragma: no cover - the assertion is the behavior
        raise AssertionError("OriginTable.set accepted a non-origin")


def test_origin_table_snapshot_is_independent_and_restore_is_exact() -> None:
    """A snapshot copy survives later writes; restore reinstates it exactly."""
    table = OriginTable({"a": ProviderOrigin(kind="core")})
    snapshot = table.snapshot()
    table.set("b", ProviderOrigin(kind="unknown"))

    assert snapshot == {"a": ProviderOrigin(kind="core")}

    table.restore(snapshot)

    assert table.names() == ("a",)
    assert table.origin("a") == ProviderOrigin(kind="core")
    assert table.origin("b") == ProviderOrigin.unknown()


def test_registry_snapshot_captures_values_and_origins_together() -> None:
    """Capture/restore is total over the value mapping and the side table."""
    registry: dict[str, object] = {"a": 1}
    table = OriginTable({"a": ProviderOrigin(kind="core")})
    snapshot = RegistrySnapshot.capture(registry, table)

    registry["b"] = 2
    registry["a"] = 3
    table.set("b", ProviderOrigin(kind="unknown"))

    snapshot.restore(registry, table)

    assert registry == {"a": 1}
    assert table.snapshot() == {"a": ProviderOrigin(kind="core")}


def test_attribution_labels_only_names_new_in_the_window() -> None:
    """The window diff is factual: new names get the template, old ones do not."""
    origin = ProviderOrigin(kind="pack-code", pack="acme", item_id=None)
    before = {"manual": object()}
    after = {"manual": object(), "acme.arm": object()}

    labels = attribution(before, after, origin)

    assert labels == {"acme.arm": origin}


def test_provider_info_is_a_live_record() -> None:
    """The shared live-provider record defaults to active and is comparable."""
    origin = ProviderOrigin(kind="manifest", pack="acme")
    info = ProviderInfo(name="acme.arm", kind="robot", origin=origin)

    assert info.active is True
    assert info == ProviderInfo(name="acme.arm", kind="robot", origin=origin, active=True)
