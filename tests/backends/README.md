# tests/backends

Backend-local tests. Each canonical backend subdirectory (`mujoco`, `mjwarp`,
`isaaclab2`, `isaaclab3_physx`, `isaaclab3_newton`) runs only in that backend's
environment. The two `isaaclab3_*` directories are the P1 scaffolding for the
two IsaacLab 3 backend identities: the canonical names are exactly
`isaaclab3_physx` and `isaaclab3_newton`, and `isaaclab3` is not a config
alias, lane or directory. Neither backend implements a capability yet, so both
lanes currently own environment and stack probes only.

This directory is outside the default `testpaths`; run a backend lane
explicitly. A lane may be a stack probe only; capability fixtures arrive with
the backend implementation.
