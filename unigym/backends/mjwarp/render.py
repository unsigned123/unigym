"""Native MuJoCo-Warp batch renderer (v095 C3).

One render context covers the N worlds of the backend's batched warp ``Data``
(``nworld=N``); cameras are addressed by their compiled MuJoCo camera name and
read back per world.  This replaces the CPU-mirror ``mujoco.Renderer`` path
only for N>1 -- the N=1 path stays the historical mirror renderer (no golden
churn, see ``unigym/backends/mjwarp/cameras.py``).

Contract mapping (docs/contracts/camera_contract.md):

* RGB    -> ``get_rgb`` -> ``(N, H, W, 3)`` uint8 (0..1 float unpacked, then
  rounded back to the packed byte value).
* depth  -> NOT ``get_depth`` (it returns ``clamp(val/depth_scale, 0, 1)`` --
  normalized, not metric).  ``rc.depth_data`` already holds the METRIC planar
  z written by the render kernel; we slice it by ``rc.depth_adr`` per camera,
  reshape to ``(N, H, W)`` and apply the contract sentinel: no-hit (0.0) and
  anything beyond the far clip become ``inf``.  Returned as ``(N, 1, H, W)``
  float32.

Resolution constraint: ``create_render_context`` takes ONE ``cam_res`` for
every active camera in that context.  A renderer therefore owns one
resolution; changing it (or the active camera set) rebuilds the context.

Pose updates: the renderer reads ``d.cam_xpos``/``d.cam_xmat``.  Writing
``cam_pos``/``cam_quat`` on the model does NOT move those until
``mjw.camlight`` runs (``mjw.kinematics`` only updates bodies/geoms/sites in
mujoco-warp 3.17; ``mjw.step`` does refresh them but is far more expensive).
"""
from __future__ import annotations

import numpy as np

__all__ = ["WarpRenderer"]


class WarpRenderer:
    """Per-rig batch renderer over one warp ``Model``/``Data`` (N worlds)."""

    def __init__(self, mjm, wm, wd, nworld: int, camera_names,
                 width: int, height: int, *, with_depth: bool = False):
        import mujoco_warp as mjw

        self._mjw = mjw
        self._mjm = mjm
        self._wm = wm
        self._wd = wd
        self._nworld = int(nworld)
        self._width = int(width)
        self._height = int(height)
        self._with_depth = bool(with_depth)
        self._names = self._normalize(camera_names)
        self._rc = None
        self._adr = {}
        self._rebuild()

    # -- configuration ---------------------------------------------------- #
    @staticmethod
    def _normalize(camera_names) -> tuple:
        if isinstance(camera_names, (str, int)):
            camera_names = [camera_names]
        names = tuple(camera_names)
        if not names:
            raise ValueError("WarpRenderer needs at least one camera")
        return names

    @property
    def camera_names(self) -> tuple:
        """Active camera names in this context (order = read index)."""
        return self._names

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` of the single context resolution."""
        return (self._width, self._height)

    def reconfigure(self, camera_names=None, width=None, height=None) -> None:
        """Rebuild the context when the active set/resolution changes."""
        names = (self._names if camera_names is None
                 else self._normalize(camera_names))
        w = self._width if width is None else int(width)
        h = self._height if height is None else int(height)
        if (names, w, h) == (self._names, self._width, self._height):
            return
        self._names, self._width, self._height = names, w, h
        self._rebuild()

    def _rebuild(self) -> None:
        self._rc = self._mjw.create_render_context(
            self._mjm,
            nworld=self._nworld,
            cam_res=(self._width, self._height),
            render_rgb=True,
            render_depth=self._with_depth,
            cam_active=list(self._names),
        )
        # get_rgb/get_depth index the ACTIVE camera list, not the MuJoCo id.
        self._adr = {
            name: {
                "rgb": int(np.asarray(self._rc.rgb_adr.numpy()).ravel()[i]),
                "depth": (int(np.asarray(self._rc.depth_adr.numpy())
                              .ravel()[i]) if self._with_depth else None),
            }
            for i, name in enumerate(self._names)
        }

    # -- rendering -------------------------------------------------------- #
    def render(self) -> None:
        """Render the active cameras for every world (one kernel pass)."""
        self._mjw.render(self._wm, self._wd, self._rc)

    def rgb(self, camera_index: int = 0) -> np.ndarray:
        """``(N, H, W, 3)`` uint8 RGB for one active camera (owned copy)."""
        import warp as wp

        out = wp.zeros((self._nworld, self._height, self._width),
                       dtype=wp.vec3, device=self._rc.rgb_data.device)
        self._mjw.get_rgb(self._rc, int(camera_index), out)
        arr = np.asarray(out.numpy(), dtype=np.float64)
        arr = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return arr

    def depth(self, camera_index: int = 0, far: float = float("inf")) -> np.ndarray:
        """``(N, 1, H, W)`` float32 METRIC planar z (``inf`` = nothing)."""
        if not self._with_depth:
            raise RuntimeError("WarpRenderer was built without depth")
        raw = np.asarray(self._rc.depth_data.numpy(), dtype=np.float32)
        adr = self._adr[self._names[int(camera_index)]]["depth"]
        n_pix = self._height * self._width
        z = raw[:, adr:adr + n_pix].reshape(self._nworld, self._height,
                                            self._width).copy()
        z[z <= 0.0] = np.inf          # no ray hit -> "nothing observed"
        z[z > float(far)] = np.inf    # contract far-clip sentinel
        return z[:, None, :, :]
