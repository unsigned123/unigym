"""The analysis gate reads both results.v1 and benchmark results."""

from __future__ import annotations

import json

import pytest

from unigym.analysis.gate import GateError, load_results


def _write(tmp_path, schema):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "episodes": [{"env": 0, "episode_id": 3, "verdict": "success", "gate": "g1/stand"}],
            }
        )
    )
    return path


@pytest.mark.parametrize("schema", ["results.v1", "results"])
def test_gate_accepts_current_result_schemas(tmp_path, schema):
    _write(tmp_path, schema)
    doc = load_results(tmp_path)
    assert doc["schema"] == schema
    assert doc["episodes"][0]["verdict"] == "success"
    assert doc["episodes"][0]["gate"] == "g1/stand"


def test_gate_rejects_unknown_result_schema(tmp_path):
    _write(tmp_path, "results.v99")
    with pytest.raises(GateError, match="results.v99"):
        load_results(tmp_path)
