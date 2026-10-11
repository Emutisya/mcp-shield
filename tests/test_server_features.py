import copy
import http.client
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_shield.core import train
from mcp_shield.data import dataset, split
from mcp_shield.offline import fingerprint
from mcp_shield.server import make_server


class ServerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.model = train(split(dataset())[0])
        self.sample = json.loads((Path(__file__).resolve().parents[1] / "examples" / "catalog.json").read_text())
        self.server = make_server(self.model, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, path, value=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        body = None if value is None else json.dumps(value)
        defaults = {"Content-Type": "application/json"} if value is not None else {}
        connection.request(method, path, body, {**defaults, **(headers or {})})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, json.loads(data)

    def test_server_snapshots_model_and_exposes_content_identity(self):
        digest = fingerprint(self.model)
        self.model["prior"] = .999999
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["model_sha256"], digest)
        self.assertEqual(self.request("POST", "/api/inspect", self.sample)[1]["decision"], "allow")
        self.assertEqual(self.request("GET", "/api/model")[1]["model_sha256"], digest)

    def test_duplicate_boundary_headers_are_rejected_not_first_value_wins(self):
        host = f"127.0.0.1:{self.server.server_port}"
        for header, value, expected in (("Host", host, 403), ("Origin", f"http://{host}", 403),
                                         ("Content-Type", "application/json", 415)):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            connection.putrequest("POST", "/api/inspect", skip_host=True)
            connection.putheader("Host", host)
            if header != "Content-Type":
                connection.putheader("Content-Type", "application/json")
            connection.putheader(header, value)
            if header != "Host":
                connection.putheader(header, value)
            connection.putheader("Content-Length", "2")
            connection.endheaders(b"{}")
            response = connection.getresponse()
            self.assertEqual(response.status, expected)
            self.assertIn("error", json.loads(response.read()))
            connection.close()
        self.assertEqual(self.request("GET", "/health")[0], 200)

    def test_unsupported_methods_have_json_recovery_and_head_has_no_body(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("PATCH", "/api/inspect")
        response = connection.getresponse()
        self.assertEqual(response.status, 405)
        self.assertEqual(response.getheader("Allow"), "GET, HEAD, POST")
        self.assertIn("Unsupported method", json.loads(response.read())["error"])
        connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("HEAD", "/health")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertGreater(int(response.getheader("Content-Length")), 0)
        self.assertEqual(response.read(), b"")
        connection.close()
