"""Episode assembly helpers (moved verbatim from collect.py, B0).

Shared by CollectSink and future benchmark sinks: slice a per-step rec dict
per env, stack an env's step slices into ``(T, D)`` episode arrays, and run
the episode-level filter chain before handing kept episodes to the writer.
"""

from __future__ import annotations

import numpy as np
import torch

from unigym.filters import run_filter_chain


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


def _flush_episode(ep_steps: list[dict], filter_chain, writer, logger):
    """Assemble, filter, and optionally write one episode."""
    ep_data = assemble_episode(ep_steps)
    if not ep_data:
        return
    keep, rejected_by = run_filter_chain(filter_chain, ep_data)
    tie = ep_data.get("time_in_episode")
    dur = float(tie[-1]) if tie is not None and len(tie) > 0 else 0.0
    logger.log_episode({"duration_s": dur}, kept=keep, filter_reason=rejected_by)
    if keep:
        writer.append_episode(ep_data)
