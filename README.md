# UniGym

UniGym is a multi-backend robot simulation and evaluation framework. This
repository is the **public core**: the importable `unigym` package, the
MuJoCo / MuJoCo-Warp / Isaac Lab backend code, the declarative run/control
contract, and one built-in adapter (`manual`) for external drivers.

It intentionally does **not** bundle robot assets, task packs, scene packs,
reference data, policy adapters, or the previous CI and documentation tree.
Those are separate artifacts; the public docs are being rewritten.

## Scope of this release

- `unigym/**` framework code (simulation core, run API, backends, props,
  cameras, contacts, runners, trace, web UI package data).
- One adapter: `unigym.adapters.manual.ManualAdapter` for user-driven loops.
- Three backends: `mujoco`, `mjwarp` (optional extra), `isaaclab2`.
  `isaaclab3` is an early-access skeleton and is incomplete.
- No bundled robot model: pass your own MJCF through `world_base` or direct
  backend construction.

## Backend support levels

| Backend | State | Notes |
|---|---|---|
| `mujoco` | Supported | CPU host lane; the default and most complete backend. |
| `mjwarp` | Optional | Install the `warp` extra; treated as an optional/nightly lane. |
| `isaaclab2` | Supported in container | Requires an Isaac Lab environment; not a host-only install. |
| `isaaclab3` | Early access | Skeleton only; interfaces may still change. |

## Install

```bash
python -m pip install -e .
# developer tooling (pytest + ruff + import-linter)
python -m pip install -e ".[dev]"
```

Requires Python >= 3.10. The core runtime pins NumPy 1.26.4 and MuJoCo 3.11.0.

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
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python -m ruff check unigym examples tests
```

## Licenses

UniGym is MIT-licensed; see [`LICENSE`](LICENSE). Third-party notices for
retained adapted code are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md);
license texts live in [`LICENSES/`](LICENSES/).
