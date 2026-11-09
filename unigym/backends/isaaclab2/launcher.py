"""IsaacLab kit launcher (``BackendInfo.kit_launcher`` face).

The kit must start before any isaaclab-dependent builder runs (robot
profiles import ``isaaclab_assets``).  ``unigym.scripts.collect_cli`` asks
the registry for this face instead of comparing the backend name, so the
``isaaclab.app`` import stays inside this package.
"""
from __future__ import annotations


def kit_launcher(parser, warn=None):
    """Launch the kit and return ``(simulation_app, args_cli)``.

    Args:
        parser: the collect CLI ``ArgumentParser`` (mutated with the
            AppLauncher argument group).
        warn: optional ``callable(args_cli)`` for the retired-flag notices.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    if warn is not None:
        warn(args_cli)
    app_launcher = AppLauncher(args_cli)
    return app_launcher.app, args_cli


__all__ = ["kit_launcher"]
