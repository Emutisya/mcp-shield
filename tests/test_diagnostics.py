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

    def test_classifier_policy_cross_tab_keeps_disagreement_visible(self):
        from mcp_shield.diagnostics import classifier_policy_cross_tab
        rows = [dict(r, decision=d) for r, d in zip(self.rows, ("deny", "allow", "review", "deny"))]
        result = classifier_policy_cross_tab(rows)
        self.assertEqual(result["0"]["deny"]["rows"], 1)
        self.assertEqual(result["1"]["allow"]["positive_labels"], 1)
        self.assertEqual(sum(v["rows"] for g in result.values() for v in g.values()), 4)

    def test_full_prediction_export_reconciles_confusion_without_metadata(self):
        from mcp_shield.evaluation import evaluate, metrics
        report = evaluate(include_predictions=True)
        rows = report["predictions"]
        self.assertEqual(len(rows), report["test_rows"])
        self.assertEqual(metrics([(r["label"], r["ml_prediction"]) for r in rows]), report["ml"])
        self.assertTrue(all("input" not in r and "server" not in r for r in rows))
        self.assertNotIn("predictions", evaluate())
        with self.assertRaises(ValueError):
            evaluate(include_predictions=1)

    def test_gate_audit_retains_overlap_and_benign_triggered_rows(self):
        from mcp_shield.evaluation import evaluate
        report = evaluate()
        histogram = report["gate_count_histogram"]
        self.assertEqual(sum(histogram.values()), report["test_rows"])
        self.assertEqual(sum(count for gates, count in histogram.items() if int(gates) > 0),
                         report["policy_decisions"]["deny"])
        self.assertEqual(sum(int(gates) * count for gates, count in histogram.items()),
                         sum(g["rows"] for g in report["gate_diagnostics"].values()))
        self.assertTrue(any(g["benign_labels"] for g in report["gate_diagnostics"].values()))
