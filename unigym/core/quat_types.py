"""Typed quaternion component orders (design v098 §2.5).

Two nominal types make the component order part of the type itself:
``QuatWXYZ`` is the repo-internal order (mujoco / IsaacLab 2.x) and
``QuatXYZW`` is the external order (scipy / IsaacLab 3.x / Warp / PhysX).
Mixing orders in an operation is a runtime ``TypeError``; conversion is
explicit (``.as_wxyz()`` / ``.as_xyzw()``).

The wrapper is a frozen ``slots`` dataclass holding a *reference* to the
backing numpy array / torch tensor -- construction never copies.  The hot
path (``env.step`` / PD / address tables) keeps using raw arrays/tensors;
the wrapper lives on introspection surfaces, public APIs and pack/file
boundaries.

numpy is first-class.  torch is supported for the trivial ops
(composition / conjugation / normalisation / rotation / equality);
``__eq__`` on torch follows ``torch.equal`` semantics, so the two payloads
must share dtype and device -- a cross-device comparison raises from torch
rather than silently returning False.

Adoption is staged (N2 + a later sweep): this module is additive and no
call site has been migrated yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar, cast

import numpy as np
import torch

from . import quat as _quat

__all__ = ["QuatWXYZ", "QuatXYZW"]

_Q = TypeVar("_Q", bound="_QuatMixin")

_IDENTITY = {
    "wxyz": (1.0, 0.0, 0.0, 0.0),
    "xyzw": (0.0, 0.0, 0.0, 1.0),
}


def _mix_error(left: _QuatMixin, right: _QuatMixin, op: str) -> TypeError:
    """Build the fail-loud message for an order mismatch."""
    return TypeError(
        f"cannot {op} {type(left).__name__} (order={left.order!r}) with "
        f"{type(right).__name__} (order={right.order!r}): component orders "
        f"differ; convert explicitly with .as_wxyz() / .as_xyzw()")


def _require_same_order(left: _QuatMixin, right: _QuatMixin,
                        op: str) -> None:
    """Raise :class:`TypeError` unless both operands share a component order."""
    if left.order != right.order:
        raise _mix_error(left, right, op)


def _compose(left: _QuatMixin, right: _QuatMixin) -> _QuatMixin:
    """Hamilton product, order-preserving (canonical wxyz math)."""
    _require_same_order(left, right, "compose")
    if left.order == "wxyz":
        return QuatWXYZ(_quat.mul(left.data, right.data))
    return QuatXYZW(_quat.to_xyzw(
        _quat.mul(_quat.to_wxyz(left.data), _quat.to_wxyz(right.data))))


class _QuatMixin:
    """Shared order-aware implementation (no fields, no ``__dict__``)."""

    __slots__ = ()

    order: ClassVar[str]
    _vec: ClassVar[slice]

    data: np.ndarray | torch.Tensor

    def __init__(self, data: np.ndarray | torch.Tensor) -> None:
        """Initialiser (the frozen dataclass subclasses generate their own)."""
        self.data = data

    def __post_init__(self) -> None:
        """Validate the payload kind and its trailing axis (no copy)."""
        data = self.data
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"{type(self).__name__}.data must be a numpy array or torch "
                f"tensor, got {type(data).__name__} "
                f"(use .from_array() for raw sequences)")
        shape = data.shape
        if len(shape) == 0 or shape[-1] != 4:
            raise ValueError(
                f"{type(self).__name__} expects a trailing axis of 4 "
                f"((4,) or (N, 4)), got shape {tuple(shape)}")

    @classmethod
    def from_array(cls: type[_Q], a: np.ndarray | torch.Tensor) -> _Q:
        """Wrap *a* without copying (raw sequences are converted first)."""
        if not isinstance(a, (np.ndarray, torch.Tensor)):
            a = np.asarray(a)
        return cls(a)

    @classmethod
    def identity(cls: type[_Q], *, batch: int | None = None,
                 like: object | None = None) -> _Q:
        """Unit quaternion in this order, optionally batched / dtype-like."""
        vec = _IDENTITY[cls.order]
        payload = like.data if isinstance(like, _QuatMixin) else like
        if isinstance(payload, torch.Tensor):
            out = torch.tensor(vec, dtype=payload.dtype, device=payload.device)
            if batch is not None:
                out = out.repeat(int(batch), 1)
            return cls(out)
        if isinstance(payload, np.ndarray):
            out = np.array(vec, dtype=payload.dtype)
        else:
            out = np.array(vec, dtype=np.float64)
        if batch is not None:
            out = np.tile(out, (int(batch), 1))
        return cls(out)

    def as_wxyz(self) -> QuatWXYZ:
        """Explicit conversion to the wxyz order (same order -> self)."""
        if self.order == "wxyz":
            return cast("QuatWXYZ", self)
        return QuatWXYZ(_quat.to_wxyz(self.data, _quat.XYZW))

    def as_xyzw(self) -> QuatXYZW:
        """Explicit conversion to the xyzw order (same order -> self)."""
        if self.order == "xyzw":
            return cast("QuatXYZW", self)
        return QuatXYZW(_quat.to_xyzw(self.data))

    def _wrap(self: _Q, data: np.ndarray | torch.Tensor) -> _Q:
        """Re-wrap a payload in the same concrete order type."""
        return cast(_Q, type(self)(data))

    def normalized(self: _Q) -> _Q:
        """Unit-normalised payload (numpy normalisation preserves float32)."""
        return self._wrap(_quat.normalize(self.data))

    def is_unit(self, tol: float = 1e-6) -> bool:
        """True when every quaternion has unit norm within *tol*."""
        data = self.data
        if isinstance(data, torch.Tensor):
            norm = torch.linalg.norm(data, dim=-1)
            return bool(torch.allclose(norm, torch.ones_like(norm),
                                       atol=tol, rtol=0.0))
        norm = np.linalg.norm(np.asarray(data), axis=-1)
        return bool(np.allclose(norm, 1.0, atol=tol, rtol=0.0))

    def conjugate(self: _Q) -> _Q:
        """Conjugate (vector part negated in this order's layout)."""
        data = self.data
        if isinstance(data, torch.Tensor):
            out = data.clone()
            out[..., self._vec] = -out[..., self._vec]
            return self._wrap(out)
        arr = np.array(data, copy=True)
        arr[..., self._vec] = -arr[..., self._vec]
        return self._wrap(arr)

    def inverse(self: _Q) -> _Q:
        """Multiplicative inverse (conjugate / squared norm)."""
        data = self.data
        conj = self.conjugate().data
        if isinstance(data, torch.Tensor):
            n2 = (data * data).sum(dim=-1, keepdim=True).clamp_min(1e-24)
        else:
            n2 = np.sum(np.asarray(data) * np.asarray(data),
                        axis=-1, keepdims=True)
            n2 = np.clip(n2, 1e-24, None)
        return self._wrap(conj / n2)

    def rotate(self, v: np.ndarray | torch.Tensor):
        """Rotate vector(s) *v* (last axis 3) by this quaternion."""
        data = self.data
        if isinstance(data, torch.Tensor):
            if self.order == "wxyz":
                w, u = data[..., 0:1], data[..., 1:4]
            else:
                u, w = data[..., 0:3], data[..., 3:4]
            vec = v if isinstance(v, torch.Tensor) else torch.as_tensor(
                v, dtype=data.dtype, device=data.device)
            t = 2.0 * torch.cross(u, vec, dim=-1)
            return vec + w * t + torch.cross(u, t, dim=-1)
        if self.order == "wxyz":
            w_np, u_np = data[..., 0:1], data[..., 1:4]
        else:
            u_np, w_np = data[..., 0:3], data[..., 3:4]
        vec = np.asarray(v)
        t = 2.0 * np.cross(u_np, vec)
        return vec + w_np * t + np.cross(u_np, t)

    def to_rotmat(self):
        """Rotation matrix (``(..., 3, 3)``, world <- local)."""
        data = self.as_wxyz().data
        if isinstance(data, torch.Tensor):
            return _quat.to_rotmat(data)
        q = np.asarray(data, dtype=np.float64)
        w, x, y, z = np.moveaxis(q, -1, 0)
        return np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                      2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                      2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                      1 - 2 * (x * x + y * y)], axis=-1),
        ], axis=-2)

    def __getitem__(self, key):
        """Index the payload; a still-4-wide slice stays typed."""
        data = self.data[key]
        if (isinstance(data, (np.ndarray, torch.Tensor))
                and data.ndim >= 1 and data.shape[-1] == 4):
            return self._wrap(data)
        return data

    def __len__(self) -> int:
        """Length of the leading axis (component count for a single quat)."""
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Payload shape (``(4,)`` or ``(N, 4)``)."""
        return tuple(self.data.shape)

    @property
    def dtype(self):
        """Payload dtype (numpy or torch)."""
        return self.data.dtype

    def __eq__(self, other: object) -> bool:
        """Elementwise/bool equality; a different order is a type error."""
        if not isinstance(other, _QuatMixin):
            return NotImplemented
        if self.order != other.order:
            raise _mix_error(self, other, "compare")
        left, right = self.data, other.data
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not (isinstance(left, torch.Tensor)
                    and isinstance(right, torch.Tensor)):
                return False
            return bool(torch.equal(left, right))
        return bool(np.array_equal(left, right))

    def __repr__(self) -> str:
        """Order first, payload compacted."""
        payload = str(self.data)
        if len(payload) > 160:
            payload = payload[:157] + "..."
        return f"{type(self).__name__}(order={self.order!r}, data={payload})"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class QuatWXYZ(_QuatMixin):
    """Quaternion with ``(w, x, y, z)`` component order."""

    data: np.ndarray | torch.Tensor
    order: ClassVar[str] = "wxyz"
    _vec: ClassVar[slice] = slice(1, 4)

    def __matmul__(self, other: QuatWXYZ) -> QuatWXYZ:
        """Compose with another wxyz quaternion (``self (x) other``)."""
        if not isinstance(other, _QuatMixin):
            return NotImplemented  # type: ignore[return-value]
        return cast("QuatWXYZ", _compose(self, other))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class QuatXYZW(_QuatMixin):
    """Quaternion with ``(x, y, z, w)`` component order."""

    data: np.ndarray | torch.Tensor
    order: ClassVar[str] = "xyzw"
    _vec: ClassVar[slice] = slice(0, 3)

    def __matmul__(self, other: QuatXYZW) -> QuatXYZW:
        """Compose with another xyzw quaternion (``self (x) other``)."""
        if not isinstance(other, _QuatMixin):
            return NotImplemented  # type: ignore[return-value]
        return cast("QuatXYZW", _compose(self, other))
