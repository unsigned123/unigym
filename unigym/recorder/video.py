"""MP4 video encoding via imageio (numpy-only, unit-testable).

Writes a list of ``np.ndarray`` uint8 frames ``(H, W, 3)`` to an MP4 file
using the bundled ffmpeg backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_mp4(frames: list[np.ndarray], path: str, fps: int = 50) -> str:
    """Write *frames* to an MP4 file at *path*.

    Parameters
    ----------
    frames
        List of ``(H, W, 3)`` uint8 arrays.
    path
        Output ``.mp4`` path.
    fps
        Frames per second.
    """
    if not frames:
        raise ValueError("no frames to write")
    frames_u8 = [np.asarray(f, dtype=np.uint8) for f in frames]
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    from unigym._optional import require
    iio = require("imageio.v2", "record", feature="MP4 recording")
    writer = iio.get_writer(path, fps=fps, codec="libx264",
                            quality=8, macro_block_size=1)
    for f in frames_u8:
        writer.append_data(f)
    writer.close()
    return path
