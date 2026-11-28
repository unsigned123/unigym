"""IsaacLab 3 kit launcher (shared ``BackendBindings.kit_launcher`` face).

Both IsaacLab 3 identities start Kit through the same ``AppLauncher`` path;
the ``isaaclab.app`` import stays inside :func:`kit_launcher`, so importing
this module (or the stdlib-only registry) never loads the engine.
"""

from __future__ import annotations

__all__ = ["kit_launcher"]


def kit_launcher(parser, warn=None):
    """Launch Kit and return ``(simulation_app, args_cli)``.

    rc1 no longer exposes ``--headless`` / ``--enable_cameras`` as launcher
    CLI flags, so the headless/camera decisions are set on the parsed args
    before the app starts.

    Args:
        parser: the collect CLI ``ArgumentParser`` (mutated with the
            ``AppLauncher`` argument group).
        warn: optional ``callable(args_cli)`` for the retired-flag notices.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.enable_cameras = True
    if warn is not None:
        warn(args_cli)
    app_launcher = AppLauncher(args_cli)
    return app_launcher.app, args_cli
