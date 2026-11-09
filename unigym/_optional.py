"""Optional-dependency boundary helper.

The base distribution carries the simulation contract (numpy, torch,
mujoco, PyYAML).  Everything else is an extra: the import site calls
:func:`require` instead of a bare ``import`` so a missing package fails
with one teaching line that names the exact install command, never a
bare traceback from deep inside a worker.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtraError(ImportError):
    """A capability boundary needs an extra that is not installed.

    Attributes:
        module: the import name that was missing.
        extra: the distribution extra that provides it.
        feature: the user-facing capability that needs it.
        install_hint: the exact pip command for the extra.
    """

    def __init__(self, module: str, extra: str, feature: str) -> None:
        """Build the teaching message and keep the facts addressable."""
        self.module = module
        self.extra = extra
        self.feature = feature
        self.install_hint = f"pip install 'unigym[{extra}]'"
        super().__init__(
            f"{feature} requires the optional dependency '{module}'; "
            f"install it with {self.install_hint} "
            "(or pip install 'unigym[all]')"
        )


def require(module: str, extra: str, *, feature: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingExtraError`.

    Parameters:
        module: dotted import name of the optional dependency.
        extra: the extra that declares it in ``pyproject.toml``.
        feature: short capability name used in the teaching message.

    Returns:
        The imported module.

    Raises:
        MissingExtraError: when the top-level package of ``module`` is
            absent.  A missing transitive dependency of an installed
            package is re-raised unchanged (it is not an extra problem).
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        root = module.split(".", 1)[0]
        if missing == root or missing.startswith(root + "."):
            raise MissingExtraError(module, extra, feature) from exc
        raise
