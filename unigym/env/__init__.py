"""IsaaclabEnv subpackage.

Note: the ``isaaclab2/env.py`` module depends on isaaclab (imported only after
AppLauncher); the core logic of ``episode`` / ``dr_events`` is torch-only /
pure python, unit-testable locally:

    from unigym.env.episode import parse_episode_length
    from unigym.backends.isaaclab2.dr_events import build_event_terms
    from unigym.backends.isaaclab2.env import IsaaclabEnv  # server side
"""

from __future__ import annotations
