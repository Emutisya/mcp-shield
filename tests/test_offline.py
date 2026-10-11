import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_shield.core import inspect, train
from mcp_shield.data import dataset, split
from mcp_shield.offline import inspect_batch, read_document


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train(split(dataset())[0])
        cls.sample = json.loads((Path(__file__).resolve().parents[1] / "examples" / "catalog.json").read_text())

    def test_batch_atomic_validation_and_counts(self):
        denied = copy.deepcopy(self.sample)
        denied["signals"]["credential_request"] = True
        result = inspect_batch([self.sample, denied], self.model)
        self.assertEqual(result["decisions"], {"allow": 1, "review": 0, "deny": 1})
        self.assertEqual(result["results"][0], inspect(self.sample, self.model))
        with patch("mcp_shield.offline.inspect") as inference:
            with self.assertRaisesRegex(ValueError, "item 1"):
                inspect_batch([self.sample, {}], self.model)
            inference.assert_not_called()
        for value in ([], {}, [self.sample] * 101):
            with self.assertRaises(ValueError):
                inspect_batch(value, self.model)

    def test_reader_bounds_and_duplicates(self):
        import io
        for raw in (b"", b"x" * 11, b'{"x":1,"x":2}', b"\xff"):
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                with self.assertRaises((ValueError, UnicodeError)):
                    read_document("unused", maximum=10)

    def test_report_rounds_display_not_threshold_comparisons(self):
        from mcp_shield.evaluation import evaluate
        prediction = inspect(self.sample, self.model)
        prediction["uncalibrated_ml_probability"] = .49999999999999994
        with patch("mcp_shield.evaluation.inspect", return_value=prediction):
            report = evaluate()
        self.assertEqual(report["ml"]["tp"], 0)
        self.assertEqual(report["ml"]["fn"], 184)
        self.assertEqual(report["first_12_ml_failures"][0]["score"], .5)
        self.assertEqual(report["display_score_decimal_places"], 12)

    def test_receipts_are_order_independent_and_content_sensitive(self):
        from mcp_shield.offline import fingerprint, receipt
        self.assertEqual(fingerprint({"b": 1, "a": 2}), fingerprint({"a": 2, "b": 1}))
        value = copy.deepcopy(self.sample)
        value["server"]["name"] = "Different fictional name"
        a, b = receipt(self.sample, self.model), receipt(value, self.model)
        self.assertNotEqual(a["observation_sha256"], b["observation_sha256"])
        self.assertEqual(a["result"], b["result"])
        self.assertEqual(a["model_sha256"], b["model_sha256"])
        self.assertNotIn("server", a)

    def test_model_introspection_exposes_absent_and_present_likelihoods(self):
        from mcp_shield.offline import model_info
        neutral = copy.deepcopy(self.model)
        neutral["probabilities"] = [[.5] * 10, [.5] * 10]
        result = model_info(neutral)
        self.assertTrue(all(f["present_log_odds"] == f["absent_log_odds"] == 0 for f in result["features"]))
        self.assertEqual(result["training_rows"], 800)
        with self.assertRaises(ValueError):
            model_info({**self.model, "prior": 0})

    def test_model_comparison_localizes_changed_parameters(self):
        from mcp_shield.offline import compare_models
        changed = copy.deepcopy(self.model)
        changed["probabilities"][1][0] = .8
        result = compare_models(self.model, changed)
        self.assertNotEqual(result["before_sha256"], result["after_sha256"])
        self.assertNotEqual(result["features"][0]["present_log_odds_delta"], 0)
        self.assertTrue(all(r["present_log_odds_delta"] == 0 for r in result["features"][1:]))
        self.assertEqual(compare_models(self.model, self.model)["prior_delta"], 0)

    def test_model_loader_bounds_and_unknown_fields(self):
        import io
        from mcp_shield.core import load_model, validate_model
        with self.assertRaises(ValueError):
            validate_model({**self.model, "claimed_certification": "unsupported"})
        for raw in (b"", b" " * 65537, b'{"version":1,"version":1}', b"\xff"):
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                with self.assertRaises(ValueError):
                    load_model("unused")

    def test_scenario_comparison_separates_metadata_and_feature_changes(self):
        from mcp_shield.offline import compare_observations
        changed = copy.deepcopy(self.sample)
        changed["server"]["name"] = "Another fictional server"
        result = compare_observations([self.sample, changed], self.model)
        self.assertEqual(result["changed_features"], [])
        self.assertEqual(result["probability_delta"], 0)
        changed["signals"]["scope_violation"] = True
        result = compare_observations([self.sample, changed], self.model)
        self.assertEqual(result["added_gates"], ["scope_violation"])
        self.assertEqual(result["after"]["decision"], "deny")
        with self.assertRaises(ValueError):
            compare_observations([self.sample], self.model)

    def test_boolean_sensitivity_preserves_original_and_hard_gates(self):
        from mcp_shield.offline import sensitivity
        original = copy.deepcopy(self.sample)
        result = sensitivity(self.sample, self.model)
        self.assertEqual(self.sample, original)
        self.assertEqual(len(result["variants"]), 5)
        self.assertTrue(all(v["result"]["decision"] == "deny" for v in result["variants"][:4]))
        self.assertEqual(result["variants"][-1]["result"]["decision"], "review")

    def test_count_boundaries_distinguish_policy_and_model_thresholds(self):
        from mcp_shield.offline import count_boundaries
        groups = count_boundaries(self.sample, self.model)["groups"]
        calls = groups["unexpected_calls"]
        self.assertEqual(calls[1]["result"]["decision"], "review")
        self.assertEqual(calls[1]["result"]["uncalibrated_ml_probability"],
                         calls[2]["result"]["uncalibrated_ml_probability"])
        self.assertNotEqual(calls[2]["result"]["uncalibrated_ml_probability"],
                            calls[3]["result"]["uncalibrated_ml_probability"])
        sizes = groups["response_bytes"]
        self.assertEqual(sizes[1]["result"], sizes[2]["result"])
        self.assertNotEqual(sizes[2]["result"], sizes[3]["result"])

    def test_policy_trace_exposes_all_triggers_even_under_denial(self):
        value = copy.deepcopy(self.sample)
        value["signals"].update(scope_violation=True, metadata_changes=1, unexpected_calls=1,
                                verified_publisher=False)
        result = inspect(value, {**self.model, "prior": .999999})
        self.assertEqual(result["decision"], "deny")
        self.assertEqual(result["review_triggers"],
                         ["ml_threshold", "metadata_drift", "unexpected_calls", "unverified_publisher"])

    def test_full_precision_explanation_reconstructs_posterior(self):
        import math
        result = inspect(self.sample, self.model)
        reconstructed = result["prior_log_odds"] + math.fsum(c["raw_log_odds_contribution"] for c in result["contributions"])
        self.assertAlmostEqual(reconstructed, result["posterior_log_odds"], places=12)
        self.assertAlmostEqual(1 / (1 + math.exp(-reconstructed)), result["uncalibrated_ml_probability"], places=12)

    def test_lexical_trace_identifies_inert_cues_not_substrings(self):
        value = copy.deepcopy(self.sample)
        value["tool"]["description"] = "PRIORITY archive with private annotations, tokenization."
        result = inspect(value, self.model)
        self.assertEqual(result["language_cues"],
                         {"instruction_language": ["priority"], "sensitive_language": ["private"]})
        self.assertEqual(result["gates"], [])

    def test_projection_is_model_independent_and_preserves_feature_order(self):
        from mcp_shield.core import FEATURES, features
        from mcp_shield.offline import projection
        result = projection(self.sample)
        self.assertEqual(list(result["features"]), list(FEATURES))
        self.assertEqual(list(result["features"].values()), features(self.sample))
        self.assertNotIn("decision", result)
        with self.assertRaises(ValueError):
            projection({})

    def test_validation_reports_all_bad_items_without_inference(self):
        from mcp_shield.offline import validation_report
        with patch("mcp_shield.offline.inspect") as inference:
            report = validation_report([{}, self.sample, None])
            inference.assert_not_called()
        self.assertEqual(report["valid_rows"], 1)
        self.assertEqual([e["index"] for e in report["errors"]], [0, 2])
        with patch("sys.argv", ["shield", "validate", "unused"]), patch("mcp_shield.cli.read_document", return_value=[{}]):
            from contextlib import redirect_stdout
            import io
            from mcp_shield.cli import main
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                main()
            self.assertEqual(exit.exception.code, 1)

    def test_schema_rejects_non_scalar_unicode_and_c1_controls(self):
        from mcp_shield.core import validate
        value = copy.deepcopy(self.sample)
        for text in ("bad\ud800name", "bad\udfffname", "bad\x85name", "bad\x7fname"):
            value["server"]["name"] = text
            with self.assertRaisesRegex(ValueError, "server.name"):
                validate(value)
        value["server"]["name"] = "Fictional café 🌿"
        self.assertEqual(validate(value), value)

    def test_batch_feature_groups_reconcile_and_preserve_absent_groups(self):
        denied = copy.deepcopy(self.sample)
        denied["signals"]["credential_request"] = True
        result = inspect_batch([self.sample, denied], self.model)
        groups = result["feature_groups"]
        self.assertEqual(groups["credential_request"]["1"]["decisions"]["deny"], 1)
        self.assertEqual(groups["scope_violation"]["1"]["rows"], 0)
        for group in groups.values():
            self.assertEqual(sum(g["rows"] for g in group.values()), 2)
            for d, count in result["decisions"].items():
                self.assertEqual(sum(g["decisions"][d] for g in group.values()), count)

    def test_representation_audit_detects_mixed_labels_and_overlap(self):
        from mcp_shield.offline import representation_audit
        rows = [{"id": str(i), "family": "a" if i < 2 else "b", "split": "train" if i < 2 else "test",
                 "input": self.sample, "label": i % 2} for i in range(4)]
        result = representation_audit(rows)
        self.assertEqual(result["shared_feature_vectors"], 1)
        self.assertEqual(result["training_mixed_label_vectors"], 1)
        self.assertEqual(result["heldout_mixed_label_vectors"], 1)

    def test_bounded_sample_generation_is_batch_compatible_and_reproducible(self):
        from mcp_shield.offline import sample_observations, validation_report
        samples = sample_observations("school", 5)
        self.assertEqual(samples, sample_observations("school", 5))
        self.assertEqual(validation_report(samples)["valid_rows"], 5)
        self.assertTrue(all("Synthetic case" in v["tool"]["description"] for v in samples))
        for family, count in (("unknown", 1), ("school", 0), ("school", 101), ("school", True)):
            with self.assertRaises(ValueError):
                sample_observations(family, count)

    def test_cli_rejects_aliasing_exports_before_evaluation(self):
        from contextlib import redirect_stderr
        import io
        from mcp_shield.cli import main, validate_outputs
        with self.assertRaises(ValueError):
            validate_outputs(["README.md"], [str(Path("README.md").resolve())])
        with patch("sys.argv", ["shield", "evaluate", "--output", "same.json", "--predictions", str(Path("same.json").resolve())]), \
             patch("mcp_shield.cli.evaluate") as evaluator, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                main()
            self.assertEqual(exit.exception.code, 2)
            evaluator.assert_not_called()

    def test_matched_model_replay_retains_hard_denial(self):
        from mcp_shield.offline import replay_models
        denied = copy.deepcopy(self.sample)
        denied["signals"]["hidden_instruction"] = True
        result = replay_models([self.sample, denied], self.model, {**self.model, "prior": .999999})
        self.assertEqual(result["transitions"]["allow"]["review"], 1)
        self.assertEqual(result["transitions"]["deny"]["deny"], 1)
        self.assertEqual([r["index"] for r in result["rows"]], [0, 1])

    def test_duplicate_audit_does_not_silently_reweight_batch(self):
        renamed = copy.deepcopy(self.sample)
        renamed["server"]["name"] = "Fictional second source"
        report = inspect_batch([self.sample, renamed, self.sample], self.model)
        self.assertEqual(report["duplicate_observations"]["unique_rows"], 2)
        self.assertEqual(report["duplicate_observations"]["groups"][0]["indexes"], [0, 2])
        self.assertEqual(report["decisions"]["allow"], 3)
