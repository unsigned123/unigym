"""Visualization recorder: MP4 video from an EXPLICIT in-scene camera.

The recorder owns **no sensors** (owner ruling: the self-built
``camera:`` path is removed — there is no provision/sugar fallback).  It
binds a rig declared under the yaml ``cameras:`` section (CameraManager) via
the REQUIRED ``camera_ref`` key and reads that rig's cached RGB frames: the
same camera serves obs and video (zero extra sensors/renders; known
trade-off: shared resolution + interval — declare a second rig when they
must differ, docs/contracts/camera_contract.md §8).

Usage in ``collect.py``::

    camera_manager = build_camera_manager(...)          # FIRST (scene cfg)
    recorder = build_recorder(cfg.get("recorder"), num_envs, camera_manager)
    ...
    frame = recorder.capture_frame()          # per record_interval steps
    recorder.write_video(frames, path, fps)   # at episode end
"""
from __future__ import annotations

import numpy as np

from .video import write_mp4

__all__ = ["Recorder", "build_recorder"]


class Recorder:
    """Video recorder bound to one CameraManager rig (``camera_ref``).

    When disabled (no ``recorder:`` yaml section or ``enabled: false``) all
    methods are no-ops, so ``collect.py`` can call them unconditionally.

    ``record_env`` selects which env's slice of the referenced camera's
    frames is recorded (the manager serves per-env frames; with a ``chase``
    rig, set ``record_env`` = the rig's ``target_env`` to film the chased
    robot).  ``record_interval`` subsamples the capture; episode slicing and
    mp4 encoding stay here.
    """

    def __init__(self, config: dict | None, num_envs: int,
                 camera_manager=None):
        cfg = dict(config or {})
        self._num_envs = int(num_envs)
        self._enabled = bool(cfg.get("enabled", False))
        self._record_env = int(cfg.get("record_env", 0))
        self._record_interval = max(1, int(cfg.get("record_interval", 1)))
        self._camera_ref = cfg.get("camera_ref")
        self._camera_manager = camera_manager

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def camera_ref(self) -> str | None:
        return self._camera_ref

    @property
    def record_env(self) -> int:
        return self._record_env

    @property
    def record_interval(self) -> int:
        return self._record_interval

    def capture_frame(self) -> np.ndarray | None:
        """Read the referenced camera's cached RGB frame for ``record_env``.

        Returns ``(H, W, 3)`` uint8, or ``None`` when disabled / the rig has
        not rendered yet.  Follow semantics belong to the rig's mounting
        (declare ``rig: {type: chase}`` for a video-follow camera).
        """
        if not self._enabled:
            return None
        n = getattr(self._camera_manager, "num_envs", None)
        if n is not None and not 0 <= self._record_env < int(n):
            raise ValueError(
                f"recorder.record_env {self._record_env} out of range for "
                f"{int(n)} env(s) (camera '{self._camera_ref}')")
        return self._camera_manager.video_rgb(self._camera_ref,
                                              self._record_env)

    def write_video(self, frames: list[np.ndarray], path: str,
                    fps: int = 50) -> str | None:
        """Write *frames* to an MP4 file. No-op if disabled.

        WYSIWYG meta marker (v054): when the referenced camera carries
        a NON-default live render state (the ``render`` live key), a
        ``<path>.render.json`` sidecar records the exact RenderSpec the
        video was shot under (the video itself shows it --
        ``capture_frame`` re-renders through the rig's live state)."""
        if not self._enabled or not frames:
            return None
        out = write_mp4(frames, path, fps)
        self._write_render_sidecar(path)
        return out

    def _write_render_sidecar(self, video_path: str) -> None:
        getter = getattr(self._camera_manager, "render_state", None)
        if getter is None or not self._camera_ref:
            return
        try:
            state = getter(self._camera_ref)
        except Exception:                      # noqa: BLE001 (meta only)
            return
        if not state:
            return
        import json
        side = f"{video_path}.render.json"
        with open(side, "w") as f:
            json.dump({"schema": 1, "camera": self._camera_ref,
                       "render_source": state["render_source"],
                       "render": state["render"],
                       "note": "WYSIWYG: this video was shot under the "
                               "live render state (null = the plain "
                               "publish path)"}, f, indent=1)
        print(f"[recorder] live render '{state['render_source']}' active "
              f"on '{self._camera_ref}' -> {side}", flush=True)


def _available_names(camera_manager) -> list[str]:
    if camera_manager is None:
        return []
    return list(getattr(camera_manager, "names", []))


def build_recorder(cfg: dict | None, num_envs: int,
                   camera_manager=None) -> Recorder:
    """Factory: build a :class:`Recorder` from the yaml ``recorder:`` section.

    The recorder can ONLY bind a rig explicitly declared under ``cameras:``
    — there is NO self-built camera path to fall back to.  Build the
    :class:`~unigym.cameras.CameraManager` FIRST (its sensors must
    join the scene cfg before env construction).

    Fail-fast ``ValueError`` (both messages list the available cameras):
    * enabled without ``camera_ref`` — the key is REQUIRED;
    * ``camera_ref`` naming an unknown / non-instantiated rig;
    * a referenced camera that does not render an RGB stream.
    """
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return Recorder(cfg, num_envs, camera_manager)
    available = sorted(_available_names(camera_manager))
    ref = cfg.get("camera_ref")
    if not ref:
        raise ValueError(
            "recorder.camera_ref is REQUIRED: the recorder only records "
            "cameras explicitly declared under yaml 'cameras:' (no "
            "self-built camera path exists). Available cameras: "
            f"{available or 'NONE — declare one under cameras:'}")
    if camera_manager is None or ref not in available:
        raise ValueError(
            f"recorder.camera_ref '{ref}' does not name an instantiated "
            f"camera rig; available cameras: "
            f"{available or 'NONE — declare it under cameras:'}")
    if f"cam/{ref}/rgb" not in camera_manager.camera_keys():
        raise ValueError(
            f"recorder.camera_ref '{ref}': camera does not render an RGB "
            f"stream (add 'rgb' to its types: list); available cameras: "
            f"{available}")
    return Recorder(cfg, num_envs, camera_manager)
