"""Loopback-only HTTP inference. No MCP connection, tool execution, or URL fetching."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from .core import inspect, unique_fields, validate_model

MAX_BODY = 16384


def make_server(model, port=8765):
    validate_model(model)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except ConnectionError:
                self.close_connection = True

        def log_message(self, format, *args):
            pass  # Avoid logging user-supplied metadata.

        def respond(self, status, body, mime="application/json"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected:
                self.respond(403, json.dumps({"error": "Host must match the loopback server address."}))
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{expected}":
                self.respond(403, json.dumps({"error": "Cross-origin requests are not accepted."}))
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, files("mcp_shield").joinpath("dashboard.html").read_text(encoding="utf-8"), "text/html")
            elif path == "/health":
                self.respond(200, '{"status":"ok","mode":"offline-synthetic-simulator"}')
            else:
                self.respond(404, '{"error":"Route not found."}')

        def do_POST(self):
            self.connection.settimeout(5)
            if not self.authorized():
                return
            if self.path != "/api/inspect":
                self.respond(404, '{"error":"Route not found."}')
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.respond(415, '{"error":"Use Content-Type: application/json."}')
                return
            try:
                raw_length = self.headers.get("Content-Length", "")
                if (len(self.headers.get_all("Content-Length", [])) != 1
                        or not raw_length.isascii() or not raw_length.isdigit()
                        or self.headers.get("Transfer-Encoding") is not None):
                    raise ValueError("A valid Content-Length is required; chunked bodies are unsupported.")
                length = int(raw_length)
                if not 0 < length <= MAX_BODY:
                    self.respond(413, '{"error":"Body must be 1 to 16384 bytes."}')
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request body.")
                result = inspect(json.loads(body.decode("utf-8"), object_pairs_hook=unique_fields), model)
            except (ValueError, UnicodeError, RecursionError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}))
                return
            except TimeoutError:
                self.respond(408, '{"error":"Request body timed out. Send a complete JSON request."}')
                return
            self.respond(200, json.dumps(result))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
