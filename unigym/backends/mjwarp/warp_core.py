"""Batched warp ``Data`` plumbing for the mjwarp backend (v094 P4).

Kept out of ``env.py`` to hold that module's maintainability index: the mixin
only touches documented attributes (``_model``, ``_capacity``, ``_datas``,
``_wdata``, ``_wm``, ``num_envs``, the actuator seam) and never imports the
env module (no cycle).

One warp ``Model`` + ONE ``Data`` with ``nworld=N``; the N CPU ``MjData``
mirrors are the read face.  Every per-world write is a ``wd.<field>[e]``
slice copy; 1-D ``time`` is written as a rebuilt vector because warp forbids
item indexing on 1-D arrays.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
import mujoco_warp as mjw

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco optional runtime)
    import mujoco

    from unigym.backends.mjwarp.sizing import Capacity


class WarpBatchMixin:
    """Warp <-> mirror plumbing + batched stepping for ``MjwarpEnv``."""

    # Host ``MjwarpEnv`` attributes the mixin touches (declared so the mixin
    # is analyzable on its own, without importing the env module).
    _model: mujoco.MjModel
    _datas: list[mujoco.MjData]
    _capacity: Capacity
    num_envs: int
    # warp ``Model`` / batched device ``Data`` / opaque warp arrays.
    _wm: Any
    _wdata: Any

    # ------------------------------------------------------------------ #
    # warp <-> mirror plumbing
    # ------------------------------------------------------------------ #
    def _new_warp_data(self, mirror, nworld: int = 1):
        """Build a fresh ``nworld``-batched device Data seeded from *mirror*.

        ``put_data`` broadcasts world 0 to every world.  ``nconmax`` is a
        PER-WORLD budget; warp allocates ``naconmax = nconmax * nworld`` by
        default, which is asserted here so a batch can never silently
        overflow the contact arena at N>1.
        """
        wdata = mjw.put_data(self._model, mirror, nworld=int(nworld),
                             nconmax=self._capacity.nconmax,
                             njmax=self._capacity.njmax)
        naconmax = getattr(wdata, "naconmax", None)
        want = int(self._capacity.nconmax) * int(nworld)
        if naconmax is not None and int(naconmax) < want:
            raise RuntimeError(
                f"mjwarp backend: allocated naconmax={int(naconmax)} does "
                f"not cover nworld={nworld} x nconmax="
                f"{int(self._capacity.nconmax)} (need >= {want})")
        return wdata

    @staticmethod
    def _source_array(dst, src) -> np.ndarray:
        """NumPy staging array for a warp copy into ``dst``."""
        if dst.dtype == wp.bool:
            return np.ascontiguousarray(src, dtype=np.bool_)
        return np.ascontiguousarray(src, dtype=np.float32)

    def _copy_field(self, target, e: int, name: str, mirror) -> None:
        """Copy one mirror state field into world ``e`` of *target*.

        Warp forbids item indexing on 1-D arrays, so ``time`` (shape
        ``(nworld,)``) is written as a whole vector rebuilt from the mirrors;
        every other state field is a ``(nworld, ...)`` array whose first axis
        slices into a writable per-world view (the required ``wd.<field>[e]``
        path).
        """
        dst_all = getattr(target, name, None)
        src = getattr(mirror, name, None)
        if dst_all is None or src is None or dst_all.size == 0:
            return
        if int(target.nworld) == 1:
            arr = self._source_array(dst_all, src)
            wp.copy(dst_all, wp.array(arr, dtype=dst_all.dtype,
                                      device=dst_all.device))
            return
        dst = dst_all[e]
        arr = self._source_array(dst, src)
        wp.copy(dst, wp.array(arr, dtype=dst.dtype, device=dst.device))

    def _copy_time(self, target, mirror) -> None:
        """Write the 1-D device time vector (item indexing is unsupported)."""
        if int(target.nworld) == 1:
            values = [float(mirror.time)]
        else:
            values = [float(d.time) for d in self._datas]
        arr = np.asarray(values, dtype=np.float32)
        wp.copy(target.time, wp.array(arr, dtype=target.time.dtype,
                                      device=target.time.device))

    _STATE_FIELDS = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied",
                     "eq_active", "mocap_pos", "mocap_quat",
                     "qacc_warmstart")

    def _restore_warp_state(self, warp_state: dict) -> None:
        """Copy warp-only continuation arrays (qacc/qacc_smooth) back."""
        for name, value in warp_state.items():
            target = getattr(self._wdata, name, None)
            if target is None:
                continue
            arr = np.ascontiguousarray(value, dtype=np.float32)
            wp.copy(target, wp.array(arr, dtype=wp.float32,
                                     device=target.device))

    def _copy_warmstart(self, mirror, e: int = 0, wdata=None) -> None:
        """Re-assert the mirror's solver warm-start on its device world."""
        target = self._wdata if wdata is None else wdata
        warm = getattr(mirror, "qacc_warmstart", None)
        dst_all = getattr(target, "qacc_warmstart", None)
        if warm is None or dst_all is None or dst_all.size == 0:
            return
        dst = dst_all[e] if int(target.nworld) > 1 else dst_all
        arr = np.ascontiguousarray(warm, dtype=np.float32)
        wp.copy(dst, wp.array(arr, dtype=dst.dtype, device=dst.device))

    def _push_mirror_to_device(self, mirror, e: int = 0, wdata=None) -> None:
        """Re-seed device world ``e`` from the CPU mirror (reset/restore).

        Only world ``e``'s slice is written -- other worlds keep their live
        state (the P4 subset-reset contract).  The solver warm-start is
        re-asserted explicitly so a restored snapshot replays the same
        trajectory.
        """
        target = self._wdata if wdata is None else wdata
        for name in self._STATE_FIELDS:
            if name == "qacc_warmstart":
                continue
            self._copy_field(target, e, name, mirror)
        self._copy_time(target, mirror)
        self._copy_warmstart(mirror, e, target)

    # ---- director writes: mirror state + model-global fields -> device --- #
    #: model arrays a director op can mutate after ``put_model`` (one
    #: MjModel is shared by every world; E6 pushes these to the batched
    #: device model).  ``eq_data`` is padded 11 -> 12 by mujoco_warp on
    #: mujoco < 3.16, so the source is padded to the device component
    #: count rather than copied verbatim.
    _MODEL_FIELDS = ("body_pos", "body_quat", "eq_data", "eq_obj2id",
                     "eq_type")

    @staticmethod
    def _model_source_array(dst, src) -> np.ndarray:
        """NumPy staging array for a model-field copy into ``dst``."""
        arr = np.asarray(src)
        if dst.dtype == wp.bool:
            return np.ascontiguousarray(arr, dtype=np.bool_)
        if dst.dtype in (wp.int32, wp.int64):
            return np.ascontiguousarray(arr, dtype=np.int32)
        # vector field (vec3f / quatf / vec12f): pad/truncate the component
        # axis to the device's component count (eq_data 11 -> 12 on 3.13).
        comps = np.asarray(dst.numpy()).shape[-1]
        if arr.ndim >= 2 and arr.shape[-1] != comps:
            pad = comps - arr.shape[-1]
            if pad > 0:
                arr = np.pad(arr, ((0, 0), (0, pad)))
            else:
                arr = arr[..., :comps]
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _copy_model_field(self, e: int, name: str, wm=None) -> None:
        """Copy one MjModel array into the (batched) device model."""
        target = self._wm if wm is None else wm
        dst_all = getattr(target, name, None)
        src = getattr(self._model, name, None)
        if dst_all is None or src is None or dst_all.size == 0:
            return
        # model arrays are batched (``*`` axis) except the per-eq id/type
        # vectors; batch fields take the per-world slice at every N.
        dst = dst_all if dst_all.ndim == 1 else dst_all[e]
        arr = self._model_source_array(dst, src)
        wp.copy(dst, wp.array(arr, dtype=dst.dtype, device=dst.device))

    def _push_model_fields(self, e: int = 0, wm=None) -> None:
        """Push director-mutable model fields (body/eq tables) to the device."""
        for name in self._MODEL_FIELDS:
            self._copy_model_field(e, name, wm)

    def _push_director_writes(self, e: int = 0) -> None:
        """Push the CPU mirror state AND model fields of world ``e``.

        Director ops mutate the mirror (``qpos``/``qvel``/``mocap_*``/
        ``eq_active``) and, for articulation re-anchors / weld slots, the
        single ``MjModel`` (``body_pos``/``body_quat``/``eq_data``).  Warp
        steps consume the DEVICE copy, so the next step only sees these
        writes after this explicit push.
        """
        self._push_mirror_to_device(self._datas[e], e)
        self._push_model_fields(e)

    def _warp_put_ctrl(self, d, e: int = 0, wdata=None) -> None:
        """Copy the mirror's ctrl vector into device world ``e``."""
        target = self._wdata if wdata is None else wdata
        self._copy_field(target, e, "ctrl", d)

    def _warp_sync_mirror(self, d, e: int = 0, wdata=None) -> None:
        """Refresh every mirror field from the device (the read face)."""
        target = self._wdata if wdata is None else wdata
        mjw.get_data_into(d, self._model, target, world_id=e)

    def _sync_all_mirrors(self) -> None:
        """Refresh every per-env mirror from its device world (read face)."""
        for e, d in enumerate(self._datas):
            self._warp_sync_mirror(d, e)

    def _push_all_ctrl(self) -> None:
        """Write every mirror's ctrl into the batched device Data."""
        for e, d in enumerate(self._datas):
            self._warp_put_ctrl(d, e)

    def _warp_step(self, d, e: int = 0, wdata=None) -> None:
        """One warp physics step + mirror sync (NEVER a CPU mj_step)."""
        target = self._wdata if wdata is None else wdata
        self._warp_put_ctrl(d, e, target)
        mjw.step(self._wm, target)
        self._warp_sync_mirror(d, e, target)

    def _warp_step1(self, d, e: int = 0, wdata=None) -> None:
        """Phase-split step 1 (M/qfrc_bias) + mirror sync."""
        target = self._wdata if wdata is None else wdata
        mjw.step1(self._wm, target)
        self._warp_sync_mirror(d, e, target)

    def _warp_step2(self, d, e: int = 0, wdata=None) -> None:
        """Phase-split step 2 (ctrl) + mirror sync."""
        target = self._wdata if wdata is None else wdata
        self._warp_put_ctrl(d, e, target)
        mjw.step2(self._wm, target)
        self._warp_sync_mirror(d, e, target)

    # ---- batched stepping: ONE kernel pass advances every world ---------- #
    def _warp_step_all(self) -> None:
        """One batched ``mjw.step`` for all worlds + sync every mirror."""
        mjw.step(self._wm, self._wdata)
        self._sync_all_mirrors()

    def _warp_step1_all(self) -> None:
        """One batched phase-split step 1 + sync every mirror."""
        mjw.step1(self._wm, self._wdata)
        self._sync_all_mirrors()

    def _warp_step2_all(self) -> None:
        """One batched phase-split step 2 + sync every mirror."""
        self._push_all_ctrl()
        mjw.step2(self._wm, self._wdata)
        self._sync_all_mirrors()
