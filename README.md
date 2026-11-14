# UniGym

UniGym is a multi-backend robot simulation and evaluation framework. This
repository is the **public core**: the importable `unigym` package, the
MuJoCo / MuJoCo-Warp / Isaac Lab backend code, the declarative run/control
contract, and one built-in adapter (`manual`) for external drivers.

It intentionally does **not** bundle robot assets, task packs, scene preset packs,
reference data, policy adapters, or the previous CI and documentation tree.
Those are separate artifacts; the public docs are being rewritten.

## Scope of this release

- `unigym/**` framework code (simulation core, run API, backends, props,
  cameras, contacts, runners, trace, web UI package data).
- One adapter: `unigym.adapters.manual.ManualAdapter` for user-driven loops.
- Three backends: `mujoco`, `mjwarp` (optional extra), `isaaclab2`.
  `isaaclab3` is an early-access skeleton and is incomplete.
- Scene data uses the `scene_presets` pack face (provider code is declared on
  the `providers` face with `kind: scene`); run configs reference it with the
  `scene_preset:` key. The removed `scenes` face and `scene:` key have no
  aliases; see [`docs/pack_api.md`](docs/pack_api.md#removed-scene-vocabulary).
- No bundled robot model: pass your own MJCF through `world_base` or direct
  backend construction.

## Pack objects

Packs are acquired explicitly; there is no attribute sugar.

```python
import unigym.packs

packs = unigym.packs.require("acme_robots")   # filtered scope, or installed()
pack = packs.pack("acme_robots")              # the only pack handle
robot = pack.robot("acme.arm")                # RobotRef, no activation
path = packs.resource_path(robot)             # declared resource path
```

`packs.pack(name).robot(id)` / `.asset(id)` / `.scene_preset(id)` /
`.scene_provider(id)` mint refs from manifests without importing pack Python or
mutating a registry. `packs.robots()` / `.scene_presets()` / `.providers()` are
the cross-pack views; `packs.origins()` reports where each pack was discovered.
See [`docs/pack_api.md`](docs/pack_api.md#pack-objects) for the full query
surface.

## Backend support levels

| Backend | State | Notes |
|---|---|---|
| `mujoco` | Supported | CPU host lane; the default and most complete backend. |
| `mjwarp` | Optional | Install the `warp` extra; treated as an optional/nightly lane. |
| `isaaclab2` | Supported in container | Requires an Isaac Lab environment; not a host-only install. |
| `isaaclab3` | Early access | Skeleton only; interfaces may still change. |

## Install

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"
# optional warp lane in the same environment
uv pip install --python .venv/bin/python -e ".[dev,warp]"
```

Requires Python >= 3.11. The public core targets MuJoCo 4.0.0 and
MuJoCo-Warp 4.0.0; NumPy is supported across the 1.26–2.x range, while the
development lock records the exact tested version.

## Manual / external-driver example

`ManualAdapter` has no policy forward: the caller owns the clock and supplies
named actions.

```python
from unigym.adapters.manual import ManualAdapter
from unigym.runtime.actions import NamedAction
from unigym.runner import ExternalRunner

adapter = ManualAdapter()
runner = ExternalRunner.from_instance(instance)   # adapter, env, sink, ...
runner.step({
    "body": NamedAction(values, ["joint_a", "joint_b"], kind="position"),
})
```

A complete runnable version with a tiny self-authored MJCF lives in
[`examples/manual_driver.py`](examples/manual_driver.py):

```bash
PYTHONPATH=. python examples/manual_driver.py
```

Implement your own policy adapter by subclassing
`unigym.adapters.base.Adapter` and registering it in your own code. The
public core does not ship vendor policy adapters.

## Documentation

The full documentation is being rewritten. See [`docs/README.md`](docs/README.md).
English and Chinese versions will live under `docs/en/` and `docs/zh/`.

## Tests

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m ruff check unigym examples tests
```

## Licenses

UniGym is MIT-licensed; see [`LICENSE`](LICENSE). Third-party notices for
retained adapted code are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md);
license texts live in [`LICENSES/`](LICENSES/).
