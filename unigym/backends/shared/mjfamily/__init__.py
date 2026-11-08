"""Shared MuJoCo-family contract package (v100).

The family covers ``backends/mujoco`` and ``backends/mjwarp``: both operate
on the same ``MjModel`` / ``MjData`` types.  G6 moved the remaining shared
implementation modules here (``scene`` / ``props`` / ``director`` /
``cameras`` / ``model``); see ``README.md`` for the import rules the guards
enforce.
"""
