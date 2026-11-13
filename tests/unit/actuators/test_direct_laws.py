"""Direct-drive laws: clip to ctrlrange and write only their own slots."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unigym.actuators.direct import DirectTorqueActuator, DirectVelocityActuator


@pytest.mark.parametrize("law", [DirectTorqueActuator, DirectVelocityActuator])
def test_direct_law_clips_targets_to_ctrlrange_and_preserves_other_slots(law):
    """A direct write is clip(target, ctrlrange) over the bound slots only."""
    actuator = law()
    actuator._bound = True
    actuator._slots = np.array([1, 3], dtype=np.int64)
    actuator._targets = np.array([50.0, -50.0], dtype=np.float64)
    model = SimpleNamespace(
        actuator_ctrlrange=np.array(
            [[0.0, 0.0], [-2.0, 2.0], [0.0, 0.0], [-5.0, 5.0], [0.0, 0.0]],
            dtype=np.float64,
        )
    )
    data = SimpleNamespace(ctrl=np.zeros(5, dtype=np.float64))

    actuator.step(model, data)

    assert data.ctrl.tolist() == pytest.approx([0.0, 2.0, 0.0, -5.0, 0.0])
