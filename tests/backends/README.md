# tests/backends

Backend-local tests. Each subdirectory (`mujoco`, `mjwarp`, `isaaclab2`,
`isaaclab3`) runs only in that backend's environment.

This directory is outside the default `testpaths`; run a backend lane
explicitly. Phase 0 creates the directory only and adds no backend fixtures.
