"""Episode filter interface (numpy-only, unit-testable).

An :class:`EpisodeFilter` judges a single episode and returns ``True`` to keep
it or ``False`` to discard it.  Filters are chained in order; the first
``False`` discards the entire episode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EpisodeFilter(ABC):
    """Judge a single episode; return True=keep / False=discard.

    The ``episode_data`` dict is assembled by collect.py at episode end and
    contains per-step arrays (already truncated to the episode's actual
    length — no padding).  Guaranteed keys:

    - ``"joint_pos"``:        (T, 12)  policy joint order
    - ``"base_pos"``:         (T, 3)   world frame
    - ``"base_lin_vel"``:     (T, 3)   body frame
    - ``"base_ang_vel"``:     (T, 3)   body frame
    - ``"command"``:           (T, 15)
    - ``"time_in_episode"``:   (T,)
    - ``"episode_id"``:        scalar
    - ``"language"``:          list[str] or (T,) object array
    - other fields (action_raw, joint_vel, …)

    All arrays use the same leading time dimension ``T``.
    """

    @abstractmethod
    def __call__(self, episode_data: dict) -> bool:
        """Return True to keep the episode, False to discard."""

    @property
    def name(self) -> str:
        """Human-readable filter name (defaults to class name)."""
        return type(self).__name__
