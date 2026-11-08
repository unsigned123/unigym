/* api.js -- the ONLY webui module that knows the HTTP API.
 *
 * Method names mirror the contract one-to-one
 * (docs/contracts/daemon_api_contract.md): api.runs() <-> GET /api/runs,
 * api.metrics(id) <-> GET /api/runs/<id>/metrics, ...  Views never
 * build URLs themselves; media and artifact URLs come from the helpers
 * below.
 *
 * Everything is relative to the portal origin (8600 -- the single
 * source): the browser never touches any other port.  POST faces carry
 * the daemon token from the page URL when one is present (?token=...).
 */

const qs = () => new URLSearchParams(location.search);
const token = () => qs().get("token") || "";
const tokq = () => token() ? `?token=${encodeURIComponent(token())}` : "";

async function request(method, path, body) {
  let url = path + tokq();
  const opts = {method, headers: {}};
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  let doc = null;
  try { doc = await r.json(); } catch { doc = null; }
  if (!r.ok && (doc === null || typeof doc !== "object")) {
    doc = {ok: false, error: `HTTP ${r.status}`};
  }
  return {status: r.status, doc};
}

async function GET(path) { return request("GET", path); }
async function POST(path, body) { return request("POST", path, body ?? {}); }

/* One run media face (mounted by the daemon media gateway, 8600). */
export const media = {
  status: (id) => GET(`/run/${id}/status`),
  info: (id) => GET(`/run/${id}/info`),
  mjpegUrl: (id, cam) => `/run/${encodeURIComponent(id)}/cam/` +
      `${encodeURIComponent(cam)}.mjpg` + tokq(),
  offer: (id, sdp) => POST(`/run/${id}/offer`, {sdp, type: "offer"}),
};

/* Artifact direct-link face (portal /artifact whitelist file server).
 * Same token discipline as request(): the token rides along when the
 * page URL carries one.  params: {run|job, name}. */
export const artifactUrl = (params) => {
  const q = new URLSearchParams(params);
  if (token()) q.set("token", token());
  return `/artifact?${q.toString()}`;
};

export const api = {
  /* portal face */
  health: () => GET("/health"),

  /* ledger face (contract §1.21): run.json scan + live merge */
  runsHistory: (limit) =>
      GET(`/api/runs?history=1&limit=${limit ?? 12}`),

  /* control face (contract §1) -- same paths, 8600 mirror */
  runs: () => GET("/api/runs"),
  run: (id) => GET(`/api/runs/${encodeURIComponent(id)}`),
  runStatus: (id) => GET(`/api/runs/${encodeURIComponent(id)}/status`),
  metrics: (id) => GET(`/api/runs/${encodeURIComponent(id)}/metrics`),
  config: (id) => GET(`/api/runs/${encodeURIComponent(id)}/config`),
  report: (id) => GET(`/api/runs/${encodeURIComponent(id)}/report`),
  scene: (id, full) =>
      GET(`/api/runs/${encodeURIComponent(id)}/scene`
          + (full ? "?full=1" : "")),
  logTail: (id, tail) =>
      GET(`/api/runs/${encodeURIComponent(id)}/log?tail=${tail ?? 4000}`),
  events: (id, tail, kind) =>
      GET(`/api/runs/${encodeURIComponent(id)}/events?tail=${tail ?? 50}`
          + (kind ? `&kind=${encodeURIComponent(kind)}` : "")),
  configs: () => GET("/api/configs"),
  adapterParams: () => GET("/api/adapters/params"),
  status: (tail) => GET(`/api/status?tail=${tail ?? 8}`),

  start: (body) => POST("/api/start", body),
  deleteRun: (output_dir) => POST("/api/runs/delete", {output_dir}),
  op: (id, op, body) =>
      POST(`/api/op/${encodeURIComponent(id)}/${op}`, body ?? {}),

  /* named ops used by the views (documented in the contract §1.14) */
  terminate: (id) => api.op(id, "shutdown"),
  pause: (id) => api.op(id, "pause"),
  resume: (id) => api.op(id, "resume"),
  step: (id, n) => api.op(id, "step", {n}),
  eval: (id, expr) => api.op(id, "inspect", {expr}),
  cameraOp: (id, body) => api.op(id, "camera", body),   // freecam
  /* director console (contract §5): the node-action face of the
   * scene card -- SAME /api/op/<run>/direct/<op> path as the CLI. */
  directorStatus: (id) => api.op(id, "direct/status"),
  direct: (id, op, body) => api.op(id, `direct/${op}`, body ?? {}),
  streamBind: (id, camera) =>
      POST(`/run/${encodeURIComponent(id)}/control`,
           {type: "stream_bind", camera}),
};

export const errMsg = (r, fallback) =>
    (r && r.doc && r.doc.error) ? r.doc.error : (fallback || "request failed");

export const query = (name) => qs().get(name);

/* SSE subscription face (contract §1.22): one EventSource on
 * /api/events.  The browser owns reconnection AND Last-Event-ID (it
 * resends the last seen id automatically); this wrapper only tracks
 * stream health (onState) and funnels typed events into onEvent as
 * {type, run_id, ts, ...payload}.  Polling fallback stays fully in
 * charge while the stream is down -- the polls are the correctness
 * floor, this is a latency win. */
export function events(onEvent, {types, runId, onState} = {}) {
  const q = new URLSearchParams();
  if (token()) q.set("token", token());   // EventSource cannot send headers
  if (types && types.length) q.set("types", types.join(","));
  if (runId) q.set("run_id", runId);
  const es = new EventSource(`/api/events?${q.toString()}`);
  let up = null;
  const mark = (v) => { if (v !== up) { up = v; if (onState) onState(v); } };
  for (const t of ["run.spawned", "run.phase", "run.exit", "bench.draft",
                   "watcher.fired", "dropped", "reset"]) {
    es.addEventListener(t, (e) => {
      mark(true);
      let doc = {};
      try { doc = JSON.parse(e.data); } catch { doc = {}; }
      onEvent({type: t, ...doc});
    });
  }
  es.onopen = () => mark(true);
  es.onerror = () => mark(false);   // browser keeps retrying on its own
  return {close: () => es.close()};
}
