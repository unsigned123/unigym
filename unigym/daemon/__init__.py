"""unigymd -- the resident daemon (media gateway).

Three-layer architecture thin-CLI -> unigymd -> N x self-contained run
instances; the runs are PURE SHM workers (zero ports, zero
tokens) and the daemon terminates all browser traffic (single 8600
source).

    unigym/daemon/service.py  DaemonService: control API server + webui
                              portal server + instance-table rebuild +
                              spawn/job scheduling (report-only watch);
                              the auth tri-state lives here
    unigym/daemon/api.py      control API (127.0.0.1:8670, curl-friendly)
    unigym/daemon/portal.py   webui portal v2 (8600: run selector/spawn/
                              terminate/live view; single browser source)
    unigym/daemon/media.py    media gateway: shared SHM frame readers +
                              MJPEG + console/info passthrough (liftable)
    unigym/daemon/webrtc.py   aiortc face of the gateway (liftable; the
                              only aiortc import, loaded lazily)
    unigym/daemon/manage.py   `unigym daemon start|stop|status` (the CLI's
                              ONLY local process operations)
    unigym/daemon/state.py    daemon identity file (/tmp/unigym-daemon.json)
    unigym/daemon/retention.py  v068 hourly sweeper for the runtime
                              artifact tree (mtime > 7d; live-run paths
                              protected)

Invariants (§1): run instances are self-contained and setsid-detached --
the daemon dying is invisible to them; the instance table is a
rebuildable view (SHM scan + registries + PID liveness); attachment is
stateless; crash supervision only REPORTS (no restart, default off).
Same-host promise : SHM never crosses machines -- one daemon per
host.

The daemon is an extra (``unigym[daemon]``), which bundles pyzmq for the
policy transports used by daemon-supervised runs.  The boundary asserts
the declared extra here so a base install gets the teaching error instead
of a traceback from a spawned worker.
"""

from unigym._optional import require

require("zmq", "daemon", feature="the unigymd daemon")
