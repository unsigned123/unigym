"""MuJoCo Warp native backend package (v094 P3).

``MjwarpEnv`` is an INDEPENDENT env orchestration (a copy of the mujoco
backend adapted to ``mujoco_warp``); it shares only the engine-agnostic
``unigym.backends.collect_core`` bookkeeping and the MuJoCo library modules
that operate on the real ``MjModel`` / mirror ``MjData`` pair.

The read face is a CPU ``MjData`` mirror: every warp step is followed by
``mjw.get_data_into(mirror, mjm, wd, 0)``.  See ``env.py`` for the precise
divergences from the CPU mujoco backend.
"""
from __future__ import annotations

from .env import BACKEND_CAPABILITIES, MjwarpEnv

__all__ = ["BACKEND_CAPABILITIES", "MjwarpEnv"]
