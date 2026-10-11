import copy
import http.client
import json
import subprocess
import sys
import threading
import unittest
from pathlib import Path

from mcp_shield.core import FLAGS, features, inspect, train, validate, validate_model
from mcp_shield.data import dataset, split
from mcp_shield.evaluation import evaluate
from mcp_shield.server import make_server

ROOT = Path(__file__).resolve().parents[1]


class ShieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = dataset()
        cls.training, cls.heldout = split(cls.rows)
        cls.model = train(cls.training)
        cls.sample = json.loads((ROOT / "examples" / "catalog.json").read_text())

    def test_hard_gates_override_benign_model(self):
        benign_model = {**self.model, "prior": 0.000001}
        for flag in FLAGS:
            with self.subTest(flag=flag):
                value = copy.deepcopy(self.sample)
                value["signals"][flag] = True
                result = inspect(value, benign_model)
                self.assertEqual(result["decision"], "deny")
                self.assertIn(flag, result["gates"])

    def test_review_and_allow(self):
        self.assertEqual(inspect(self.sample, self.model)["decision"], "allow")
        for key, value in (("metadata_changes", 1), ("unexpected_calls", 1), ("verified_publisher", False)):
            item = copy.deepcopy(self.sample)
            item["signals"][key] = value
            self.assertEqual(inspect(item, self.model)["decision"], "review")
        high = {**self.model, "prior": 0.999999}
        self.assertEqual(inspect(self.sample, high)["decision"], "review")

    def test_strict_invalid_inputs(self):
        mutations = [
            lambda x: x.update(extra=True),
            lambda x: x["signals"].update(credential_request="false"),
            lambda x: x["signals"].update(response_bytes=True),
            lambda x: x["signals"].update(metadata_changes=-1),
            lambda x: x["signals"].update(unexpected_calls=101),
            lambda x: x["signals"].update(response_bytes=1000001),
            lambda x: x["signals"].pop("hidden_instruction"),
            lambda x: x["tool"].update(description=" " * 10),
            lambda x: x["tool"].update(description="a" * 2001),
            lambda x: x["server"].update(name="\x00"),
        ]
        for mutation in mutations:
            item = copy.deepcopy(self.sample)
            mutation(item)
            with self.assertRaises(ValueError):
                validate(item)
        for value in (None, [], "", {"server": None}):
            with self.assertRaises(ValueError):
                validate(value)

    def test_model_validation_and_explanation(self):
        with self.assertRaises(ValueError):
            validate_model({**self.model, "prior": float("nan")})
        result = inspect(self.sample, self.model)
        self.assertEqual(len(result["contributions"]), len(features(self.sample)))
        self.assertTrue(0 <= result["uncalibrated_ml_probability"] <= 1)

    def test_training_labels_and_artifact_metadata_are_strict(self):
        for rows in (None, [], [None], [{"input": self.sample, "label": True}],
                     [{"input": self.sample, "label": -1}], [{"input": self.sample, "label": 2}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                train(rows)
        for field, value in (("version", True), ("training_rows", True),
                             ("training_rows", 0), ("training_rows", None)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_model({**self.model, field: value})

    def test_split_is_reproducible_and_template_disjoint(self):
        self.assertEqual(self.rows, dataset())
        self.assertFalse({r["family"] for r in self.training} & {r["family"] for r in self.heldout})
        def template(row):
            return row["input"]["tool"]["description"].split(". Synthetic case")[0]
        self.assertFalse({template(r) for r in self.training} & {template(r) for r in self.heldout})
        self.assertEqual(len({json.dumps(r["input"], sort_keys=True) for r in self.rows}), len(self.rows))
        item = copy.deepcopy(self.sample)
        item["server"]["name"] = "arbitrary unrelated name"
        item["tool"]["name"] = "new_name"
        self.assertEqual(features(item), features(self.sample))

    def test_report_exposes_failures(self):
        report = evaluate()
        self.assertGreater(report["ml"]["fn"], 0)
        self.assertGreater(report["ml"]["fp"], 0)
        self.assertEqual(sum(report["policy_decisions"].values()), report["test_rows"])
        self.assertEqual(report, json.loads((ROOT / "reports" / "baseline.json").read_text()))

    def test_cli_rejects_invalid_ports_before_loading_model(self):
        for port in ("-1", "65536", "999999999999999999999"):
            with self.subTest(port=port):
                result = subprocess.run(
                    [sys.executable, "-m", "mcp_shield", "serve", "--port", port,
                     "--model", "models/missing-port-test.json"],
                    cwd=ROOT, text=True, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--port must be between 0 and 65535", result.stderr)
                self.assertIn("use 0 to choose an available port", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_local_http_inference_and_errors(self):
        server = make_server(self.model, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def request(method, path, body=None, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request(method, path, body, headers or {})
                response = connection.getresponse()
                return response.status, response.read(), response.getheader("Content-Security-Policy")
            finally:
                connection.close()
        try:
            status, body, csp = request("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn(b"clawpilotTheme", body)
            self.assertIn("frame-ancestors 'none'", csp)
            self.assertEqual(request("GET", "/health")[0], 200)
            headers = {"Content-Type": "application/json"}
            status, body, _ = request("POST", "/api/inspect", json.dumps(self.sample), headers)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), inspect(self.sample, self.model))
            self.assertEqual(request("POST", "/api/inspect", "{}", headers)[0], 400)
            self.assertEqual(request("POST", "/api/inspect", "{broken", headers)[0], 400)
            self.assertEqual(request("POST", "/api/inspect", "a" * 16385, headers)[0], 413)
            self.assertEqual(request("POST", "/api/inspect", "{}", {**headers, "Origin": "https://example.invalid"})[0], 403)
            self.assertEqual(request("GET", "/", headers={"Host": "example.invalid"})[0], 403)
            self.assertEqual(request("POST", "/api/inspect", "{}")[0], 415)
            self.assertEqual(request("GET", "/missing")[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
