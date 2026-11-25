"""Contact owner/aggregation and one-read force/active-body facts."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.contacts.base import (
    ContactGroupSpec,
    contact_force_facts,
)
from unigym.contacts.manager import ContactManager


def _spec(**kw):
    base = dict(
        name="feet",
        sensor_refs=[{"owner": "robot", "articulation": "g1", "body": "foot"}],
        frame="world",
    )
    base.update(kw)
    return ContactGroupSpec(**base)


class _RecordingBackend:
    def __init__(self, raw):
        self.raw = raw
        self.reads = 0

    def setup_scene(self):
        return {}

    def bind_env(self, env):
        return None

    def read_raw(self, spec):
        self.reads += 1
        return self.raw

    def group_axes(self, spec):
        return (self.raw.shape[1], 0)

    def close(self):
        return None


def test_owner_is_derived_and_aggregation_is_validated():
    spec = _spec(
        sensor_refs=[{"owner": "object", "name": "crate"}],
        aggregation="sum_magnitudes",
    )
    assert spec.owner == "object"
    assert spec.metadata()["aggregation"] == "sum_magnitudes"
    with pytest.raises(ValueError):
        _spec(aggregation="max")


def test_net_vs_sum_magnitudes():
    raw = np.array([[[3.0, 4.0, 0.0], [-3.0, -4.0, 0.0]]], dtype=np.float32)

    net_force, net_active = contact_force_facts(raw, _spec(aggregation="net"))
    assert net_force.tolist() == pytest.approx([0.0])
    assert net_active.tolist() == [2]

    sum_force, sum_active = contact_force_facts(raw, _spec(aggregation="sum_magnitudes"))
    assert sum_force.tolist() == pytest.approx([10.0])
    assert sum_active.tolist() == [2]


def test_active_bodies_respects_contact_threshold():
    raw = np.array([[[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
    spec = _spec(format="contact", threshold=1.0, aggregate="none")
    force, active = contact_force_facts(raw, spec)
    assert active.tolist() == [1]


def test_manager_one_raw_read_serves_force_and_active_bodies():
    raw = np.array([[[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    backend = _RecordingBackend(raw)
    manager = ContactManager(
        [
            _spec(
                sensor_refs=[{"owner": "fixture", "name": "microwave"}],
                aggregation="sum_magnitudes",
            )
        ],
        backend,
    )
    out = manager.read()
    assert backend.reads == 1
    assert set(out) == {"contact/feet", "contact/feet/force", "contact/feet/active_bodies"}
    assert out["contact/feet/force"].tolist() == pytest.approx([6.0])
    assert out["contact/feet/active_bodies"].tolist() == [2]
    assert manager.metadata()["feet"]["owner"] == "fixture"


def test_pairwise_force_facts_reduce_filters():
    raw = np.array([[[[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]]], dtype=np.float32)
    spec = _spec(
        filter_refs=[{"owner": "robot", "body": "ball"}],
        aggregation="sum_magnitudes",
    )
    force, active = contact_force_facts(raw, spec)
    assert force.tolist() == pytest.approx([5.0])
    assert active.tolist() == [1]


def test_sum_magnitudes_applies_configured_filter_aggregate():
    raw = np.array(
        [[[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]],
        dtype=np.float32,
    )  # one sensor body, two filter contacts with magnitudes 3 and 4

    def force(aggregate):
        spec = _spec(
            filter_refs=[{"owner": "robot", "body": "ball"}],
            aggregation="sum_magnitudes",
            aggregate=aggregate,
        )
        return contact_force_facts(raw, spec)[0].tolist()

    assert force("sum") == pytest.approx([7.0])
    assert force("max") == pytest.approx([4.0])
    assert force("mean") == pytest.approx([3.5])
    assert force("none") == pytest.approx([7.0])


def test_metadata_carries_explicit_object_and_support_refs():
    spec = _spec(
        name="crate_table",
        sensor_refs=[{"owner": "object", "name": "crate"}],
        filter_refs=[{"owner": "support", "name": "table"}],
        format="contact",
        threshold=0.5,
        aggregate="any",
    )
    metadata = spec.metadata()
    assert metadata["owner"] == "object"
    assert metadata["mode"] == "pairwise"
    assert metadata["sensor_refs"] == [{"owner": "object", "name": "crate"}]
    assert metadata["filter_refs"] == [{"owner": "support", "name": "table"}]
