"""Depth-frame preprocessing helpers for the ``depth_image`` obs source.

Standalone module so IsaaclabEnv keeps a
self-contained, adapter-independent depth pipeline (crop -> clip -> normalize
to [-0.5, 0.5]).  A future robot profile with a trained depth encoder only
needs to request the ``depth_image`` obs key; the numeric conventions below
are the deployment provenance (factual history, kept as-is).
"""
from __future__ import annotations

import torch

DEPTH_CLIP_MIN = 0.3         # m
DEPTH_CLIP_MAX = 4.0         # m

# Training/rendered full-res sensor frame that the policy input (58x87) is
# cropped from: the TSC training camera renders 106x60 @ 87 deg horizontal
# FOV (``depth.original``) and ``process_depth_image`` crops border rows/cols
# ``[1:-1, 10:-9]`` — reproducing that crop keeps the *per-pixel ray geometry*
# identical to what the depth encoder was trained on (a full-rate 87x58
# render has a ~15% wider effective FOV and mis-biases the CNN's layout).
DEPTH_FULL_H = 60
DEPTH_FULL_W = 106
DEPTH_CROP_ROWS = (1, -1)    # keep rows  1..58 of the 60-h frame
DEPTH_CROP_COLS = (10, -9)   # keep cols 10..96 of the 106-w frame


def crop_depth_image(d: torch.Tensor) -> torch.Tensor:
    """Reproduce the training ``crop_depth_image`` on a full-res depth frame.

    The TSC training camera renders ``(60, 106)`` @ 87 deg and keeps rows
    ``[1:-1]`` / cols ``[10:-9]`` -> the ``(58, 87)`` encoder input (so does
    the deployed realsense decimation+crop).  Applying the same crop here
    reproduces the per-pixel ray geometry the depth encoder was trained on.
    Accepts ``(..., 60, 106)`` or ``(..., 1, 60, 106)``; the trailing (H, W)
    dims are cropped, so it returns ``(..., 58, 87)``.
    """
    return d[..., DEPTH_CROP_ROWS[0]:DEPTH_CROP_ROWS[1],
             DEPTH_CROP_COLS[0]:DEPTH_CROP_COLS[1]]


def preprocess_depth(raw: torch.Tensor) -> torch.Tensor:
    """Deployment depth normalization: crop, clip(0.3, 4.0), (d-0.3)/3.7-0.5.

    Accepts raw full-res distance frames ``(..., 60, 106, 1)`` / ``(..., 1,
    60, 106)`` / ``(..., 60, 106)`` (meters, inf beyond the camera far clip);
    crops to the trained ``(58, 87)`` layout, returns float32 values in
    [-0.5, 0.5].
    """
    d = torch.nan_to_num(raw.float(), nan=DEPTH_CLIP_MAX, posinf=DEPTH_CLIP_MAX)
    if d.dim() >= 2 and d.shape[-1] == 1:
        d = d.squeeze(-1)
    d = crop_depth_image(d)
    d = d.clamp(DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
    d = (d - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN) - 0.5
    return d
