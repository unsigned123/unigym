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
- Five canonical backends: `mujoco`, `mjwarp` (optional extra), `isaaclab2`,
  `isaaclab3_physx`, `isaaclab3_newton`. The PhysX identity has P2 capability
  coverage and a nightly lane; the Newton identity is still a stack-probe lane
  until its P3 capability work.
- Scene data uses the `scene_presets` pack face (provider code is declared on
  the `providers` face with `kind: scene`); run configs reference it with the
  `scene_preset:` key. The removed `scenes` face and `scene:` key have no
  aliases; see [`docs/pack_api.md`](docs/pack_api.md#removed-scene-vocabulary).
- No bundled robot model: pass your own MJCF through `world_base` or direct
  backend construction.

## Pack objects

Packs are acquired explicitly; there is no attribute sugar.

```python
from unigym.pack_api import Face
from unigym.packs import PackLoader, require

packs = require("acme_robots")                # filtered scope, or installed()
pack = packs.pack("acme_robots")              # the stored Pack declaration
loader = PackLoader.from_pack_set(packs)      # object loading + resources
robot = loader.robot_profile("acme_robots", "acme.arm")  # RobotProfile
item = pack.item(Face.ASSETS, "acme.cup")     # the stored PackItem
path = loader.resource_for(item)              # declared resource path
```

`loader.robot_profile(pack, id)` loads the complete `RobotProfile` (resource
document or entrypoint factory) without mutating the robot registry.
`packs.asset(id)` / `.scene_preset(id)` / `.scene_provider(id)` mint inert
`AssetResourceRef` / `ScenePresetRef` / `SceneProviderRef` projections from
stored `PackItem` values; each ref has a `from_item(item)` classmethod and no
manifest is imported or resource path read while minting. `packs.items()` /
`.providers()` / `.scene_presets()` are the cross-pack views; `packs.origins()`
reports where each pack was discovered. The old `unigym.packs.catalog` and
`unigym.packs.runtime` modules and the `PackCatalog`/`PackRuntime` names are
gone: `PackSet` owns cross-pack lookup and uniqueness, and `PackLoader` owns
discovery, resources and activation. See
[`docs/pack_api.md`](docs/pack_api.md#pack-objects) for the full query surface.

## Backend support levels

| Backend | State | Notes |
|---|---|---|
| `mujoco` | Supported | CPU host lane; the most complete backend. |
| `mjwarp` | Optional | Install the `warp` extra; treated as an optional/nightly lane. |
| `isaaclab2` | Supported in container | Requires an Isaac Lab 2 environment; not a host-only install. |
| `isaaclab3_physx` | Lane-tested, nightly | Contacts, director, cameras/recorder, DR events and terrain are lane-tested. |
| `isaaclab3_newton` | Stack-probe lane | Canonical identity; capability parity waits for P3. |

The backend surface is canonical-only. `backend` is **required**, and its value
must be exactly one of the five names above. `backend: isaaclab` and
`backend: isaaclab3` are **not aliases** and are rejected with a teaching
error.

Migration:

- **Configs**: configs that omitted `backend` or used `isaaclab` /
  `isaaclab3` must choose `isaaclab2`, `isaaclab3_physx` or
  `isaaclab3_newton` explicitly; there is no version-based family
  auto-selection or silent fallback.
- **Python API**: `BACKEND_ALIASES` is removed. `build_name(None)`,
  `Run(backend=None)` and `build_env_backend(None)` no longer have an implicit
  default; pass a canonical name explicitly. `isaaclab` and `isaaclab3` are
  rejected like any unknown spelling, not resolved as aliases.
- **Executor `backends:` bindings**: binding keys are canonical too. Rename a
  legacy `backends: {isaaclab: ...}` key to the concrete backend(s) it serves
  (`isaaclab2`, `isaaclab3_physx` and/or `isaaclab3_newton`). The executor
  **name** `"isaaclab"` is unchanged (it is the docker image face), so
  `backends: {isaaclab2: isaaclab}` remains valid.

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
a sequence of self-describing `NamedAction` entries. Every action column is
addressed by `(articulation, joint_name)`; sequence order is not semantic and
each entry's `joint_names` order fixes its own column order. Runtime values
carry no action-group field.

```python
from unigym.adapters.manual import ManualAdapter
from unigym.runtime.actions import NamedAction
from unigym.runner import ExternalRunner

adapter = ManualAdapter()
runner = ExternalRunner.from_instance(instance)   # adapter, env, sink, ...
runner.step((
    NamedAction(
        articulation="robot",                     # owning articulation
        values=values,                            # (num_envs, len(joint_names))
        joint_names=("joint_a", "joint_b"),       # column order
        kind="position",
    ),
))
```

A complete runnable version with a tiny self-authored MJCF lives in
[`examples/manual_driver.py`](examples/manual_driver.py):

```bash
PYTHONPATH=. python examples/manual_driver.py
```

Implement your own policy adapter by subclassing
`unigym.adapters.base.Adapter` and registering it in your own code. The
public core does not ship vendor policy adapters.

## Run termination mode

By default (`termination_mode: core`) a run builds its termination group from
the top-level `terminations:` section and reports the per-env
`StepResult.terminated` mask. A task layer that owns termination itself (for
example an RL `TaskEnv`) declares `termination_mode: external`:

```yaml
termination_mode: external
terminations: []
runner:
  auto_reset: false
  reset_on_termination: false
```

External mode builds an empty core termination group that never fires, so core
`StepResult.terminated` is all-False. `terminations:` must be absent or empty,
and both reset policy keys must be explicitly false: the caller owns
terminated-row resets. Core still exposes `reset(env_ids)`; there is no core
`terminate()` API.

## Documentation

The full documentation is being rewritten. See [`docs/README.md`](docs/README.md).
English and Chinese versions will live under `docs/en/` and `docs/zh/`.

Action system reference: [`docs/actions.md`](docs/actions.md).

## Tests

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m ruff check unigym examples tests
```

## Licenses

UniGym is MIT-licensed; see [`LICENSE`](LICENSE). Third-party notices for
retained adapted code are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md);
license texts live in [`LICENSES/`](LICENSES/).
