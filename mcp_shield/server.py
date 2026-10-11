"""Loopback-only HTTP inference. No MCP connection, tool execution, or URL fetching."""

import json
import copy
import threading
import math
import time
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from .core import inspect, unique_fields, validate_model
from .offline import (compare_observations, count_boundaries, fingerprint, inspect_batch,
                      model_info, projection, receipt, sensitivity, validation_report)

MAX_BODY = 16384


def finish_rejection(connection):
    """Brief bounded drain prevents unread request bytes resetting an error reply."""
    try:
        connection.shutdown(socket.SHUT_WR)
        deadline = time.monotonic() + .1
        remaining_bytes = MAX_BODY * 2
        while remaining_bytes > 0:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                break
            connection.settimeout(remaining_time)
            chunk = connection.recv(min(4096, remaining_bytes))
            if not chunk:
                break
            remaining_bytes -= len(chunk)
    except OSError:
        pass


def make_server(model, port=8765, max_workers=8, body_timeout=5):
    if type(max_workers) is not int or not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be an integer from 1 to 32.")
    if type(body_timeout) not in (int, float) or not math.isfinite(body_timeout) or not 0 < body_timeout <= 30:
        raise ValueError("body_timeout must be finite and greater than 0, at most 30 seconds.")
    validate_model(model)
    model = copy.deepcopy(model)
    model_identity = fingerprint(model)
    operations = {
        "/api/inspect": lambda value: inspect(value, model),
        "/api/batch": lambda value: inspect_batch(value, model),
        "/api/compare": lambda value: compare_observations(value, model),
        "/api/project": projection,
        "/api/receipt": lambda value: receipt(value, model),
        "/api/sensitivity": lambda value: sensitivity(value, model),
        "/api/boundaries": lambda value: count_boundaries(value, model),
        "/api/validate": validation_report,
    }
    slots = threading.BoundedSemaphore(max_workers)

    class BoundedServer(ThreadingHTTPServer):
        def process_request(self, request, client_address):
            if not slots.acquire(blocking=False):
                body = b'{"error":"Local request capacity is busy. Retry after an active request finishes."}'
                response = (f"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json; charset=utf-8\r\n"
                            f"Content-Length: {len(body)}\r\nConnection: close\r\nRetry-After: 1\r\n"
                            "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n").encode() + body
                try:
                    request.settimeout(1)
                    request.sendall(response)
                    finish_rejection(request)
                except OSError:
                    pass
                finally:
                    self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                slots.release()

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

        def send_error(self, code, message=None, explain=None):
            if code == 501:
                self.respond(405, '{"error":"Unsupported method. Use GET, HEAD or POST on a documented route."}')
            else:
                self.respond(code, json.dumps({"error": self.responses.get(code, ("Invalid HTTP request",))[0]}))

        def respond(self, status, body, mime="application/json"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if status == 405:
                self.send_header("Allow", "GET, HEAD, POST")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
            if status >= 400:
                self.close_connection = True
                finish_rejection(self.connection)

        def authorized(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if len(self.headers.get_all("Host", [])) != 1 or self.headers.get("Host") != expected:
                self.respond(403, json.dumps({"error": "Host must match the loopback server address."}))
                return False
            origin = self.headers.get("Origin")
            if len(self.headers.get_all("Origin", [])) > 1 or (origin is not None and origin != f"http://{expected}"):
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
                self.respond(200, json.dumps({"status": "ok", "mode": "offline-synthetic-simulator",
                                              "model_sha256": model_identity, "training_rows": model["training_rows"]}))
            elif path == "/api/model":
                self.respond(200, json.dumps(model_info(model)))
            else:
                self.respond(404, '{"error":"Route not found."}')

        def do_POST(self):
            self.connection.settimeout(5)
            if not self.authorized():
                return
            if self.path not in operations:
                self.respond(404, '{"error":"Route not found."}')
                return
            if (len(self.headers.get_all("Content-Type", [])) != 1
                    or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
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
                deadline = time.monotonic() + body_timeout
                chunks = []
                received = 0
                while received < length:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Request body deadline exceeded.")
                    self.connection.settimeout(remaining)
                    chunk = self.rfile.read1(min(4096, length - received))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)
                self.connection.settimeout(5)
                body = b"".join(chunks)
                if len(body) != length:
                    raise ValueError("Incomplete request body.")
                result = operations[self.path](json.loads(body.decode("utf-8"), object_pairs_hook=unique_fields))
            except (ValueError, UnicodeError, RecursionError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}))
                return
            except TimeoutError:
                self.connection.settimeout(5)
                self.respond(408, '{"error":"Request body timed out. Send a complete JSON request."}')
                return
            self.respond(200, json.dumps(result))

        do_HEAD = do_GET

    return BoundedServer(("127.0.0.1", port), Handler)
