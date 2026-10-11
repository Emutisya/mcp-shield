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

    def test_comparator_baselines_use_same_heldout_labels(self):
        from mcp_shield.evaluation import evaluate
        report = evaluate()
        gates = report["hard_gates_only_baseline"]
        self.assertEqual(gates["tp"] + gates["fp"], report["policy_decisions"]["deny"])
        self.assertEqual(report["always_positive_baseline"]["fn"], 0)
        self.assertEqual(report["always_positive_baseline"]["tn"], 0)
        self.assertEqual(gates["positive_labels"], report["ml"]["positive_labels"])

    def test_burden_separates_review_from_denial(self):
        from mcp_shield.diagnostics import decision_burden
        rows = [dict(r, decision=d) for r, d in zip(self.rows, ("allow", "review", "deny", "review"))]
        burden = decision_burden(rows)
        self.assertEqual(burden["review"]["benign_labels"], 1)
        self.assertEqual(burden["deny"]["positive_labels"], 1)
        self.assertEqual(sum(v["rows"] for v in burden.values()), 4)
        with self.assertRaises(ValueError):
            decision_burden(self.rows)

    def test_family_policy_metrics_reconcile(self):
        from mcp_shield.evaluation import evaluate
        report = evaluate()
        for key in ("tp", "fp", "tn", "fn"):
            self.assertEqual(sum(g["policy_nonallow_as_positive"][key] for g in report["per_family"].values()),
                             report["policy_nonallow_as_positive"][key])
        for d in ("allow", "review", "deny"):
            self.assertEqual(sum(g["decision_burden"][d]["benign_labels"] for g in report["per_family"].values()),
                             report["decision_burden"][d]["benign_labels"])
