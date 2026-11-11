# tests/integration

Composition, CLI, daemon, subprocess and real-run paths. Integration tests are
automatically tagged `slow` and are outside the fast lane.

## Current scope

### CLI subprocess smoke

`test_cli_smoke.py` runs the launcher (`python -m unigym.scripts.launch`) as a
real subprocess from the repository root, in a clean environment with the
dispatch overrides removed. It proves the process boundary only:

- top-level `--help` renders the command surface and exits 0;
- an unknown subcommand is an argparse usage error (exit 2);
- `run --help` renders backend choices without an asset/deps root;
- `daemon --help` renders its subcommands without the daemon extra.

No simulator, network, GPU or daemon is started. Assertions use stable CLI
tokens, not full help wording or machine paths.

### Daemon missing-extra boundary (base lane)

`test_daemon_cli.py` runs `daemon status` in the base environment with a
temporary `UNIGYM_DEPS_ROOT` and `UNIGYM_DAEMON_STATE`. When the optional
`zmq` extra is absent it asserts a clean teaching error (exit 2, the extra
name, no traceback) and that no state file or daemon process is created. The
test skips when `zmq` is already installed; run it before the daemon lane has
provisioned the extra to exercise the missing-extra branch.

### Daemon lifecycle (daemon lane)

`test_daemon_lifecycle.py` requires the `unigym[daemon]` extra through
`tests/_support/envs.require_optional_module`: a local run skips when `zmq` is
missing, and a strict lane fails. The test starts a detached daemon on two
ephemeral loopback ports with a temporary state file and log, verifies
`status --json` reports `alive`/`health.ok`, then stops it and verifies the
state file is gone and the status reports down. A second test pins the
idempotent "nothing to stop" behavior. Daemons bind loopback only; the lane
uses no external network, real runs, or webui/media traffic.

Every started daemon is stopped in a `finally` block. If a cleanup stop leaves
the process alive, the test fails with the pid and daemon log path instead of
silently leaking it. Ports, state and logs live under pytest `tmp_path`; no
fixed ports or machine paths are used. The lane sets
`UNIGYM_ORPHAN_SWEEP=off` so a test daemon cannot probe or sweep unrelated
machine-global SHM/daemon state.

## Run explicitly

```bash
pytest tests/integration -q
python tools/ci/run_lane.py --lane daemon
make test-integration
```
