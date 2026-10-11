import copy
from contextlib import redirect_stderr
import io
import hashlib
import http.client
import json
import math
import subprocess
import sys
import threading
import socket
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from mcp_shield.core import FLAGS, features, inspect, train, validate, validate_model
from mcp_shield.data import dataset, split
from mcp_shield.evaluation import evaluate, metrics
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

    def test_extreme_valid_probabilities_keep_explanations_finite(self):
        model = copy.deepcopy(self.model)
        model["prior"] = 1e-320
        model["probabilities"][0] = [.99] * len(model["features"])
        model["probabilities"][1] = [1e-320] * len(model["features"])
        item = copy.deepcopy(self.sample)
        item["signals"]["verified_publisher"] = False
        item["signals"]["metadata_changes"] = 1
        result = inspect(item, model)
        self.assertTrue(all(math.isfinite(row["log_odds_contribution"]) for row in result["contributions"]))
        self.assertTrue(0 <= result["uncalibrated_ml_probability"] <= 1)

    def test_rounding_cannot_move_probability_across_evaluation_threshold(self):
        model = copy.deepcopy(self.model)
        model["probabilities"] = [[.5] * len(model["features"]) for _ in range(2)]
        for prior, expected in ((.4999999, 0), (.5000001, 1)):
            model["prior"] = prior
            probability = inspect(self.sample, model)["uncalibrated_ml_probability"]
            self.assertAlmostEqual(probability, prior, places=10)
            self.assertEqual(int(probability >= .5), expected)
        model["prior"] = .5
        model["probabilities"][1][0] = .500001
        model["probabilities"][1][1] = .500002
        result = inspect(self.sample, model)
        self.assertEqual(result["contributions"][0]["feature"], model["features"][1])
        self.assertEqual(result["contributions"][1]["feature"], model["features"][0])
        self.assertEqual(result["contributions"][0]["log_odds_contribution"],
                         result["contributions"][1]["log_odds_contribution"])

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

    def test_invalid_dataset_parameters_are_rejected(self):
        for seed, count in ((True, 100), ("17", 100), (17, True),
                            (17, 0), (17, -1), (17, 1.5)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                dataset(seed, count)

    def test_split_checks_reject_missing_groups_duplicates_and_family_leakage(self):
        leaking = copy.deepcopy(self.rows)
        leaking[-1]["family"] = leaking[0]["family"]
        invalid = copy.deepcopy(self.rows)
        invalid[0]["split"] = "unknown"
        for rows in ([], self.training, invalid, [*self.rows, self.rows[0]], leaking):
            with self.assertRaises(ValueError):
                split(rows)
        script = (
            "from mcp_shield.data import dataset,split; rows=dataset(); "
            "rows[-1]['family']=rows[0]['family']; split(rows)"
        )
        result = subprocess.run([sys.executable, "-O", "-c", script],
                                cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("families must be disjoint", result.stderr)

    def test_report_exposes_failures(self):
        report = evaluate()
        self.assertGreater(report["ml"]["fn"], 0)
        self.assertGreater(report["ml"]["fp"], 0)
        self.assertEqual(sum(report["policy_decisions"].values()), report["test_rows"])
        self.assertEqual(report, json.loads((ROOT / "reports" / "baseline.json").read_text()))

    def test_evaluation_manifest_tracks_training_and_heldout_inputs(self):
        report = evaluate()
        for key, rows in (("training_sha256", self.training), ("heldout_sha256", self.heldout)):
            expected = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertEqual(report[key], expected)
        changed = evaluate(seed=18)
        self.assertNotEqual(changed["training_sha256"], report["training_sha256"])
        self.assertNotEqual(changed["heldout_sha256"], report["heldout_sha256"])

    def test_metric_inputs_reject_empty_and_nonbinary_predictions(self):
        for pairs in (None, [], [(1,)], [(2, 0)], [(True, 1)], [(0, 0.0)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                metrics(pairs)
        self.assertEqual(metrics([(0, 0)])["accuracy"], 1)
        self.assertIsNone(metrics([(0, 0)])["recall"])
        self.assertIsNone(metrics([(0, 0)])["precision"])

    def test_heldout_family_metrics_reconcile_without_masking_group_failures(self):
        report = evaluate()
        families = report["per_family"]
        self.assertEqual(sorted(families), report["test_families"])
        self.assertEqual(sum(group["rows"] for group in families.values()), report["test_rows"])
        for field in ("tp", "fp", "tn", "fn"):
            self.assertEqual(sum(group["ml"][field] for group in families.values()), report["ml"][field])
        for decision in ("allow", "review", "deny"):
            self.assertEqual(sum(group["policy_decisions"][decision] for group in families.values()),
                             report["policy_decisions"][decision])
        self.assertTrue(any(group["ml"]["fn"] for group in families.values()))
        self.assertTrue(any(group["ml"]["fp"] for group in families.values()))

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

    def test_failed_output_replacement_preserves_artifact_and_cleans_temporary_file(self):
        from mcp_shield.cli import write_output
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self.model), encoding="utf-8")
            with patch("mcp_shield.cli.os.replace", side_effect=PermissionError("Isolated write failure")):
                with self.assertRaises(PermissionError):
                    write_output(path, '{"incomplete": true}')
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.model)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            write_output(path, '{"replacement": true}')
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"replacement": True})

    def test_cli_occupied_port_has_actionable_recovery(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self.model), encoding="utf-8")
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = subprocess.run(
                [sys.executable, "-m", "mcp_shield", "serve", "--model", str(path), "--port", str(port)],
                cwd=ROOT, text=True, capture_output=True, timeout=10,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"port {port}", result.stderr)
        self.assertIn("--port 0", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_cli_json_boundaries_match_http_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            model.write_text(json.dumps(self.model), encoding="utf-8")
            path = Path(directory) / "input.json"
            for raw, error in (
                (json.dumps(self.sample).encode("utf-16"), "Error:"),
                (json.dumps(self.sample).replace('"verified_publisher": true',
                    '"verified_publisher": false, "verified_publisher": true').encode(), "Duplicate JSON"),
                (b"x" * 16385, "1 to 16384 bytes"),
                (b"[" * 1500 + b"]" * 1500, "Error:"),
            ):
                path.write_bytes(raw)
                result = subprocess.run(
                    [sys.executable, "-m", "mcp_shield", "inspect", str(path), "--model", str(model)],
                    cwd=ROOT, text=True, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("Traceback", result.stderr)

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
            logs = io.StringIO()
            with redirect_stderr(logs), patch.object(server.RequestHandlerClass, "respond", side_effect=ConnectionResetError("client closed")):
                with self.assertRaises(http.client.RemoteDisconnected):
                    request("GET", "/health")
            self.assertEqual(logs.getvalue(), "")
            self.assertEqual(request("GET", "/health")[0], 200)
            headers = {"Content-Type": "application/json"}
            duplicated = json.dumps(self.sample).replace('"verified_publisher": true', '"verified_publisher": false, "verified_publisher": true')
            self.assertEqual(request("POST", "/api/inspect", duplicated, headers)[0], 400)
            self.assertEqual(request("POST", "/api/inspect", "{}", {**headers, "Content-Length": "+2"})[0], 400)
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
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=8)
            try:
                connection.request("POST", "/api/inspect", "{}", {**headers, "Content-Length": "100"})
                response = connection.getresponse()
                self.assertEqual(response.status, 408)
                self.assertIn("timed out", json.loads(response.read())["error"])
            finally:
                connection.close()
            self.assertEqual(request("GET", "/health")[0], 200)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
