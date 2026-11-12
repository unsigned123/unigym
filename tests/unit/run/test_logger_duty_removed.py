"""The summary no longer aggregates foot duty metrics."""

from __future__ import annotations

from unigym.logger import DataCollectLogger


def test_summary_has_no_duty_factor_section(tmp_path):
    """A run summary carries no foot duty aggregation."""
    logger = DataCollectLogger({"run_name": "x"}, str(tmp_path), 1, total_steps=1)
    logger.log_step(0, {"vx": 0.0, "base_z": 1.0, "alive": 1.0})

    summary = logger.write_summary(total_records=1, shards=0)

    assert "duty_factor" not in summary["stats"]
    assert not any(key.startswith("duty_") for key in summary["stats"])
