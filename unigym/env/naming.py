"""Robot-agnostic naming helpers (pure string logic, unit-testable).

Same style as :mod:`unigym.env.episode` /
:mod:`unigym.backends.isaaclab2.init_pose`: pure string / torch-free
functions, consumed by backend composition, unit-testable locally without an
engine environment.
"""

from __future__ import annotations

ENV_REGEX_NS = "{ENV_REGEX_NS}"
"""The per-env clone namespace prefix (``{ENV_REGEX_NS}``)."""
