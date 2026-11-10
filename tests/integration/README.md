# tests/integration

Composition, CLI, daemon, subprocess and real-run paths. Integration tests are
automatically tagged `slow` and are outside the fast lane.

## Current scope: CLI subprocess smoke

`test_cli_smoke.py` runs the launcher (`python -m unigym.scripts.launch`) as a
real subprocess from the repository root, in a clean environment with the
dispatch overrides removed. It proves the process boundary only:

- top-level `--help` renders the command surface and exits 0;
- an unknown subcommand is an argparse usage error (exit 2);
- `run --help` renders backend choices without an asset/deps root;
- `daemon --help` renders its subcommands without the daemon extra.

No simulator, network, GPU or daemon is started. Assertions use stable CLI
tokens, not full help wording or machine paths.

## Deferred

Daemon `status` and the real start/stop lifecycle are a later slice: they need
the optional `zmq` extra and a separate process-state design.

Run explicitly:

```bash
pytest tests/integration -q
make test-integration
```
