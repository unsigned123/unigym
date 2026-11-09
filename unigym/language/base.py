"""Language generator interface (torch-only, unit-testable)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class LanguageGenerator(ABC):
    """Generate natural-language descriptions from motion states.

    Implementations receive a batched ``states`` dict (per-env tensors of
    shape ``(N,)`` or lists of length ``N``) and the raw command tensor,
    and return a list of ``N`` description strings.
    """

    @abstractmethod
    def describe(self, states: dict, command: torch.Tensor) -> List[str]:
        """Generate one description string per env.

        Args:
            states: per-env motion quantities.  Keys (all optional;
                implementations should use ``.get`` with sensible defaults):
                vx, vy, wz: base linear/angular velocity (body frame, m/s, rad/s)
                base_z: base height (m)
                gait_name: gait category ("trot"/"pace"/"bound"/"pronk")
                gait_freq: gait frequency (Hz)
                footswing: foot swing height command (m)
                body_height_cmd: body height offset command (m)
                pitch_cmd: pitch command (rad)
                stance_width: stance width (m)
                stance_length: stance length (m)
                robot_name: robot name for subject noun
            command: (N, dim) command tensor (adapter-specific layout).

        Returns:
            list of N description strings.
        """
