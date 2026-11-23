"""Composite termination group (OR / AND combination) + inversion wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Termination

if TYPE_CHECKING:
    import torch

    from unigym.evaluation import EvalContext


def _require_vector_context(ctx: EvalContext) -> None:
    """Termination groups judge the whole live vector, never one episode."""
    if ctx.env.env_id is not None:
        raise ValueError(
            "TerminationGroup requires a vector evaluation context (env_id=None), "
            f"got env_id={ctx.env.env_id!r}"
        )


def _normalize_device(device: str | torch.device) -> torch.device:
    """Normalize an unindexed ``cuda`` device to the current CUDA index."""
    import torch

    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return normalized


def _same_device(left: str | torch.device, right: str | torch.device) -> bool:
    """Whether two device specs address the same type/index."""
    return _normalize_device(left) == _normalize_device(right)


def _validate_mask(term_name: str, mask: object, ctx: EvalContext) -> torch.Tensor:
    """Validate one term's mask: bool ``(num_envs,)`` on the context device."""
    import torch

    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"termination {term_name!r} returned {type(mask).__name__}, "
            f"expected a torch.Tensor bool mask"
        )
    if mask.dtype != torch.bool:
        raise TypeError(
            f"termination {term_name!r} returned dtype {mask.dtype}, expected torch.bool"
        )
    if tuple(mask.shape) != (ctx.num_envs,):
        raise ValueError(
            f"termination {term_name!r} returned shape {tuple(mask.shape)}, "
            f"expected ({ctx.num_envs},)"
        )
    if not _same_device(mask.device, ctx.env.device):
        raise ValueError(
            f"termination {term_name!r} returned device {mask.device}, "
            f"expected {ctx.env.device}"
        )
    return mask


class InvertedTermination(Termination):
    """Logical NOT of a wrapped termination.

    Turns a *fault* detector into a *health* detector, e.g. wrapping
    ``FallTermination`` yields "body upright and high enough" — the
    exit/health condition of recovery-style switching. Built from
    yaml via the per-entry ``invert: true`` flag (see
    :func:`terminations.build_terminations`).
    """

    def __init__(self, term: Termination, name: str | None = None):
        super().__init__(None, name or f"not_{term.name}")
        self.term = term

    def check(self, ctx: EvalContext) -> torch.Tensor:
        return ~self.term.check(ctx)


class TerminationGroup:
    """Combine several terminations into one mask.

    Args:
        terminations: list of :class:`Termination` instances.
        mode: ``"or"`` (default; any term fires -> fire) or ``"and"``
            (all terms must fire).

    An empty group never fires (all-False mask), matching the
    "no terminations configured -> nothing triggers" contract.
    """

    def __init__(self, terminations: list[Termination] | None = None, mode: str = "or"):
        if mode not in ("or", "and"):
            raise ValueError(f"mode must be 'or' or 'and', got {mode!r}")
        self.terminations = list(terminations or [])
        self.mode = mode

    def __len__(self) -> int:
        return len(self.terminations)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.terminations]

    def check_detail(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        """Per-termination masks ``{name: (N,) bool}`` (diagnostics)."""
        _require_vector_context(ctx)
        detail: dict[str, torch.Tensor] = {}
        for term in self.terminations:
            detail[term.name] = _validate_mask(term.name, term.check(ctx), ctx)
        return detail

    def combine(self, detail: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Reduce per-termination masks (from :meth:`check_detail`)."""
        if not detail:
            return None
        masks = list(detail.values())
        out = masks[0]
        for mask in masks[1:]:
            out = (out | mask) if self.mode == "or" else (out & mask)
        return out

    def check(self, ctx: EvalContext) -> torch.Tensor:
        """Combined ``(N,)`` bool mask (True = the group fires)."""
        import torch

        detail = self.check_detail(ctx)
        combined = self.combine(detail)
        if combined is None:
            return torch.zeros(ctx.num_envs, dtype=torch.bool, device=ctx.env.device)
        return combined
