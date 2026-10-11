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
