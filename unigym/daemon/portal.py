"""daemon.portal -- the unigymd webui portal (pure static face).

8600 = the SINGLE browser source (gate d).  This module generates NO HTML:
pages live in the ``unigym/webui/`` package directory (zero Python, zero
build) and are served verbatim; everything dynamic is the HTTP API -- the
``/api/*`` control face mirrored by delegation (api.py module-level
routing, byte-identical to 8670) and the ``/run/<id>/`` media gateway
(media.py / webrtc.py).  Auth = the daemon's single-point tri-state:
GET open, POST faces check the token per request (``?token=``).
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from pathlib import Path

from . import api as control_api
from . import events
from . import queries
from .api import _err

#: The static face ships as package data.  Resolving through
#: importlib.resources keeps the source checkout, an editable install and
#: a wheel install on the same lookup path.
WEBUI_DIR = Path(str(files("unigym").joinpath("webui")))
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


class PortalHandler(BaseHTTPRequestHandler):
    """Static webui + /api mirror + media gateway + artifact whitelist."""

    def log_message(self, format: str, *args: object) -> None:
        """Quiet per-request access logging (portal writes no access log)."""
        pass     # noqa: E704

    def _service(self) -> Any:
        """The daemon service attached to the enclosing HTTPServer."""
        return getattr(self.server, "service")

    # -- plumbing
    def _send(self, code: int, raw: bytes, ctype: str,
              cache: str = "no-cache") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _json_send(self, code: int, obj: dict) -> None:
        self._send(code, (json.dumps(obj, indent=1, default=str) + "\n")
                   .encode(), "application/json")

    def _redirect(self, to: str) -> None:
        self.send_response(303)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _body_raw(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > 65536:
            # explicit 400, never a silent truncate-to-{} (misleading 200)
            raise ValueError("request body too large (65536 bytes max)")
        return self.rfile.read(n) if n > 0 else b""

    def _form(self) -> dict:
        raw = self._body_raw().decode("utf-8", "replace")
        return {k: v[0] for k, v in
                urllib.parse.parse_qs(raw, keep_blank_values=True).items()}

    def _read_json_body(self) -> dict:
        raw = self._body_raw()
        doc = json.loads(raw) if raw else {}
        if not isinstance(doc, dict):
            raise ValueError("body must be a JSON object")
        return doc

    # -- verbs
    def do_GET(self):                          # noqa: N802 (stdlib name)
        """Route one GET request.

        /run media gateway, /api mirror, webui statics, /health,
        /artifact whitelist (see module docstring).
        """
        parts = urllib.parse.urlparse(self.path)
        path = parts.path
        if path.startswith("/run/"):
            return self._media(parts, post=False)
        if path.startswith("/api/"):
            return self._api(parts, post=False)
        if path in ("/", "/index.html"):
            return self._static(WEBUI_DIR / "index.html")
        if path.startswith("/view/"):
            return self._static(WEBUI_DIR / "run.html")
        if path.startswith("/assets/"):
            f = WEBUI_DIR / "assets" / Path(path[len("/assets/"):]).name
            return self._static(f, cache="max-age=300")
        if path == "/health":
            return self._send(200, b"ok", "text/plain")
        if path == "/artifact":
            return self._artifact(urllib.parse.parse_qs(parts.query))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):                         # noqa: N802 (stdlib name)
        """Route one POST request.

        /run media commands, /api mirror, /spawn + /terminate form
        faces (token checked per the auth tri-state).
        """
        parts = urllib.parse.urlparse(self.path)
        path = parts.path
        if path.startswith("/run/"):
            return self._media(parts, post=True)
        if path.startswith("/api/"):
            return self._api(parts, post=True)
        if path == "/spawn":
            return self._spawn(self._form())
        if path == "/terminate":
            return self._terminate(self._form())
        self._send(404, b"not found", "text/plain")

    def _media(self, parts, post: bool) -> None:
        from .media import route_run_get, route_run_post
        try:
            if post:
                route_run_post(self, parts.path, parts.query,
                               self._read_json_body(), self._service())
            else:
                route_run_get(self, parts.path, parts.query,
                              self._service())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as e:            # oversized/malformed body
            self._json_send(*_err(str(e)))

    def _api(self, parts, post: bool) -> None:
        svc = self._service()
        try:
            if not post and parts.path.rstrip("/") == "/api/events":
                # (contract §1.22): the browser's subscription
                # face -- auth rides the tokened posture (EventSource
                # cannot send headers; ?token= is the standing channel)
                if not svc.media.token_ok(parts.query, self.headers):
                    return self._json_send(401, {
                        "ok": False, "error": "invalid or missing token",
                        "hint": "append ?token=<token> (daemon log "
                                "[webui] line)"})
                return events.serve_sse(
                    self, svc, urllib.parse.parse_qs(parts.query))
            if post and not svc.media.token_ok(parts.query, self.headers):
                return self._json_send(401, {
                    "ok": False, "error": "invalid or missing token",
                    "hint": "append ?token=<token> (daemon log [webui] line)"})
            if post:
                code, doc = control_api.api_post(svc, parts.path,
                                                 self._read_json_body())
            else:
                code, doc = control_api.api_get(
                    svc, parts.path, urllib.parse.parse_qs(parts.query))
            self._json_send(code, doc)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as e:
            self._json_send(*_err(str(e)))

    def _static(self, path: Path, cache: str = "no-cache") -> None:
        ctype = MIME.get(path.suffix)
        real: Path | None = None
        raw: bytes | None = None
        try:
            real = path.resolve()
            raw = real.read_bytes()
        except OSError:
            raw = None
        inside = (raw is not None and real is not None
                  and WEBUI_DIR.resolve() in real.parents)
        if ctype is None or raw is None or not inside:
            return self._send(404, b"not found", "text/plain")
        self._send(200, raw, ctype, cache=cache)

    def _authorized_form(self, form: dict) -> bool:
        import hmac
        svc = self._service()
        return svc.control_token is None or bool(hmac.compare_digest(
            form.get("token") or "", svc.control_token))

    def _redirect_home(self) -> None:
        tok = self._service().control_token
        self._redirect("/" + (f"?token={urllib.parse.quote(tok)}"
                              if tok else ""))

    def _form_fail(self, code: int, msg: str, doc: dict) -> None:
        self._send(code, f"{msg}: {doc.get('error', '?')}\n".encode(),
                   "text/plain")

    def _spawn(self, form: dict) -> None:
        if not self._authorized_form(form):
            return self._send(403, b"bad token\n", "text/plain")
        config = (form.get("config") or "").strip()
        if not config:
            return self._send(400, b"config required\n", "text/plain")
        ov = {"backend": form.get("backend") or None,
              "gpu": form.get("gpu") or None}
        code, doc = control_api.api_post(self._service(), "/api/start",
                                         {"config": config,
                                          "overrides": ov})
        if code != 200:
            return self._form_fail(400, "spawn failed", doc)
        self._redirect_home()

    def _terminate(self, form: dict) -> None:
        if not self._authorized_form(form):
            return self._send(403, b"bad token\n", "text/plain")
        run = form.get("run") or ""
        code, doc = control_api.api_post(self._service(),
                                         f"/api/op/{run}/shutdown", {})
        if code != 200:
            return self._form_fail(400, "terminate failed", doc)
        self._redirect_home()

    # -- artifacts (whitelist; resolution lives in daemon.queries) --------
    def _artifact(self, qs: dict) -> None:
        code, msg, ctype, payload = queries.resolve_artifact(
            self._service(), (qs.get("run") or [""])[0],
            (qs.get("job") or [""])[0], (qs.get("name") or [""])[0])
        if payload is None:
            return self._send(code, msg.encode(), "text/plain")
        self._send(code, payload, ctype)
