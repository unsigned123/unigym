# tests/backends

Backend-local tests. Each subdirectory (`mujoco`, `mjwarp`, `isaaclab2`,
`isaaclab3_physx`, `isaaclab3_newton`) runs only in that backend's environment.
The two `isaaclab3_*` directories are the P0 scaffolding for the two IsaacLab 3
backend identities. The plan is for `isaaclab3` to become a config alias to
`isaaclab3_physx` in P1; until then the lanes probe the current `isaaclab3`
stub.

This directory is outside the default `testpaths`; run a backend lane
explicitly. A lane may be a stack probe only; capability fixtures arrive with
the backend implementation.
