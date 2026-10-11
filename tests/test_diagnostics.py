import unittest
from mcp_shield.diagnostics import threshold_sweep


class DiagnosticTests(unittest.TestCase):
    rows = [{"label": 0, "score": .2}, {"label": 1, "score": .5},
            {"label": 1, "score": .8}, {"label": 0, "score": .8}]

    def test_sweep_threshold_ties_and_monotonic_counts(self):
        result = threshold_sweep(self.rows, [1, .8, 0])
        self.assertEqual([r["tp"] for r in result], [2, 1, 0])
        self.assertEqual(result[1]["fp"], 1)
        for thresholds in ([True], [.5, .5], [], [float("nan")], [-.1]):
            with self.assertRaises(ValueError):
                threshold_sweep(self.rows, thresholds)
        with self.assertRaises(ValueError):
            threshold_sweep([{"label": True, "score": .5}])

    def test_confusion_rates_preserve_undefined_denominators(self):
        from mcp_shield.evaluation import metrics
        result = metrics([(0, 0), (0, 1), (1, 0), (1, 1)])
        for key in ("specificity", "false_positive_rate", "false_negative_rate", "f1", "balanced_accuracy"):
            self.assertEqual(result[key], .5)
        absent_positive = metrics([(0, 0)])
        self.assertIsNone(absent_positive["balanced_accuracy"])
        self.assertIsNone(absent_positive["f1"])
        self.assertIsNone(metrics([(1, 1)])["specificity"])
