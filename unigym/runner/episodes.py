"""Episode assembly helpers (moved verbatim from collect.py, B0).

Shared by the benchmark and rewind consumers: slice a per-step rec dict per
env and stack an env's step slices into ``(T, D)`` episode arrays.
"""

from __future__ import annotations

import numpy as np
import torch


def slice_step_for_env(rec: dict, env_idx: int) -> dict:
    """Extract one env's data slice from a per-step rec dict."""
    sliced = {}
    for k, v in rec.items():
        if isinstance(v, torch.Tensor):
            sliced[k] = v[env_idx].cpu().clone()
        elif isinstance(v, list):
            sliced[k] = v[env_idx] if env_idx < len(v) else ""
        elif isinstance(v, np.ndarray):
            sliced[k] = v[env_idx].copy()
        else:
            sliced[k] = v
    return sliced


def assemble_episode(steps: list[dict]) -> dict:
    """Stack per-env step dicts into episode_data with (T, D) numpy arrays."""
    if not steps:
        return {}
    keys: set[str] = set()
    for s in steps:
        keys.update(s.keys())
    result = {}
    for k in keys:
        vals = [s[k] for s in steps if k in s]
        if not vals:
            continue
        first = vals[0]
        if isinstance(first, str):
            result[k] = np.array(vals, dtype=object)
        elif isinstance(first, torch.Tensor):
            result[k] = torch.stack(vals).cpu().numpy()
        else:
            result[k] = np.array(vals)
    return result
