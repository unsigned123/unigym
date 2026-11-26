# UniGym

[![PyPI](https://img.shields.io/pypi/v/unigym?label=PyPI&color=blue)](https://pypi.org/project/unigym/)
[![Python](https://img.shields.io/pypi/pyversions/unigym)](https://pypi.org/project/unigym/)
[![CI](https://github.com/unsigned123/unigym/actions/workflows/ci.yml/badge.svg)](https://github.com/unsigned123/unigym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/unsigned123/unigym/blob/main/LICENSE)
[![Status: public beta](https://img.shields.io/badge/status-public%20beta-orange.svg)](#project-status)

**Multi-backend robot simulation with one explicit contract.**

Define robots, scenes, actions and runs once. Execute them on MuJoCo,
MuJoCo Warp or Isaac Lab, with explicit configuration and no hidden backend
aliases.

[Quickstart](#quickstart) · [Packages](https://github.com/unsigned123/unigym-packages) · [Backends](#backends) · [Documentation](#documentation) · [Examples](https://github.com/unsigned123/unigym/tree/main/examples)

## Quickstart

Install UniGym:

```bash
python -m pip install unigym
```

Run the packaged manual-driver example:

```bash
python -m unigym.examples.manual_driver
```

Expected output:

```text
t=0 target=[0.10000000149011612]
t=1 target=[0.20000000298023224]
t=2 target=[0.30000001192092896]
t=3 target=[0.4000000059604645]
t=4 target=[0.5]
final hinge qpos: 0.11291945913207943
```

The example uses a tiny MIT-clean MJCF, the `ManualAdapter` and
`ExternalRunner`. No policy, robot pack or external asset is required.

UniGym requires a compatible PyTorch build (`>=2.4,<3`); `pip` will use one that
is already installed. For a CPU-only machine or a smaller download, install the
CPU wheel first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install unigym
```

For CUDA or Isaac Lab environments, install the PyTorch build that matches that
environment before installing UniGym.

Next steps:

- [Action system](https://github.com/unsigned123/unigym/blob/main/docs/actions.md)
- [Inspect a pack](https://github.com/unsigned123/unigym/blob/main/docs/pack_api.md)
- [Simulation facade](https://github.com/unsigned123/unigym/blob/main/docs/sim_facade.md)
- [More examples](https://github.com/unsigned123/unigym/tree/main/examples)

## Why UniGym

- **One contract, multiple simulators.**
  Define robot, scene, action and run behavior once; execute on MuJoCo,
  MuJoCo Warp or Isaac Lab without rewriting the task.

- **Explicit and inspectable.**
  Packs, resources and run configuration are explicit. `unigym pack show`
  prints what a pack declares; `--runtime` shows what activation actually
  registered.

- **Bring your own policy.**
  The core ships a manual adapter and runner boundaries; add your policy
  without forking the framework.

UniGym is built for robotics engineers, reinforcement-learning researchers and
evaluation/benchmark authors who need more than one simulation stack.

## Core and packages

This repository is the simulation **core**: the framework, backends, explicit
run/action/observation contracts, manual adapter and evaluation hooks. It does
not ship robot assets, task packs, policy adapters or a benchmark loop.

For RL and benchmarking, use the companion packages in
[`unigym-packages`](https://github.com/unsigned123/unigym-packages):

| Package | Purpose |
|---|---|
| `unigym-rl` | Gymnasium-shaped task environments and RL task hooks |
| `unigym-bench` | Benchmark flows, reports and evaluation harness |
| `unigym-robots` | Pinned Unitree Go1/Go2/G1 MJCF and USD assets |

The core stays framework-only; packages depend on a compatible core version.

## Backends

| Backend | Install | Status |
|---|---|---|
| `mujoco` | base install | Supported, CPU |
| `mjwarp` | `pip install "unigym[warp]"` | Optional / nightly |
| `isaaclab2` | external Isaac Lab environment | Supported in container |
| `isaaclab3_physx` | external Isaac Lab environment | Nightly |
| `isaaclab3_newton` | external Isaac Lab environment | Experimental |

The base package is asset-agnostic: use your own MJCF/USD, or load assets from an
explicit pack. Isaac Lab is not installed by `pip`; use a pinned Isaac Lab
environment.

## Choose your setup

| Goal | Start here |
|---|---|
| Local CPU simulation | `pip install unigym` |
| MuJoCo Warp | `pip install "unigym[warp]"` |
| Isaac Lab 2 | external Isaac Lab environment |
| Isaac Lab 3 PhysX | external Isaac Lab environment |
| Isaac Lab 3 Newton | external Isaac Lab environment |
| Daemon / Web UI | `pip install "unigym[webui]"` |
| Recording | `pip install "unigym[record]"` |
| Media streaming | `pip install "unigym[media]"` |

## Project status

UniGym is in **public beta (0.x)**. APIs may change between minor releases.
MuJoCo is the most complete backend; Isaac Lab requires an external environment;
IsaacLab 3 Newton is still experimental. Version `1.0.0` will mark the stable
contract.

What works today:

- MuJoCo, MuJoCo Warp and Isaac Lab backend paths
- Explicit pack discovery, inspection and activation
- Manual adapter, external runner and config-driven runs
- Observation, action, runner and evaluation contracts
- Daemon, web UI and recording extras

Current limits:

- No bundled robot assets, task packs or policy adapters
- Isaac Lab requires a separately managed environment
- IsaacLab 3 Newton capability parity is still in progress

## Documentation

- [Pack API](https://github.com/unsigned123/unigym/blob/main/docs/pack_api.md)
- [Action system](https://github.com/unsigned123/unigym/blob/main/docs/actions.md)
- [Simulation facade](https://github.com/unsigned123/unigym/blob/main/docs/sim_facade.md)
- [Observations](https://github.com/unsigned123/unigym/blob/main/docs/observations.md)
- [Examples](https://github.com/unsigned123/unigym/tree/main/examples)
- [Changelog](https://github.com/unsigned123/unigym/blob/main/CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. For larger design or contract changes,
open an issue first so the architecture and migration path can be discussed.

If UniGym is useful to you, star the repository and share the examples you
build.

## License

UniGym is MIT-licensed. See [LICENSE](https://github.com/unsigned123/unigym/blob/main/LICENSE)
and [THIRD_PARTY_NOTICES.md](https://github.com/unsigned123/unigym/blob/main/THIRD_PARTY_NOTICES.md).
