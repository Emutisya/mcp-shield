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

    def test_feature_distribution_reports_partition_shift_without_changing_model(self):
        import copy
        from mcp_shield.data import dataset
        from mcp_shield.diagnostics import feature_distribution
        value = dataset()[0]["input"]
        a, b = copy.deepcopy(value), copy.deepcopy(value)
        a["signals"]["scope_violation"] = False
        b["signals"]["scope_violation"] = True
        result = feature_distribution([a, a], [b])
        self.assertEqual(result["scope_violation"]["fraction_delta"], 1)
        self.assertEqual(result["scope_violation"]["training_present"], 0)
        with self.assertRaises(ValueError):
            feature_distribution([], [b])

    def test_dependence_identifies_perfect_association_and_constant_features(self):
        import copy
        from mcp_shield.data import dataset
        from mcp_shield.diagnostics import feature_dependence
        a, b = (copy.deepcopy(dataset()[0]["input"]) for _ in range(2))
        a["signals"].update(credential_request=False, scope_violation=False)
        b["signals"].update(credential_request=True, scope_violation=True)
        result = feature_dependence([{"input": a, "label": 0}, {"input": b, "label": 0}])
        pair = next(r for r in result if r["label"] == 0 and r["left"] == "credential_request" and r["right"] == "scope_violation")
        self.assertEqual(pair["phi"], 1)
        self.assertEqual(pair["cells_00_01_10_11"], [1, 0, 0, 1])
        self.assertTrue(all(r["phi"] is None for r in result if r["label"] == 1))

    def test_ranking_awards_half_credit_for_ties_and_preserves_undefined_auc(self):
        from mcp_shield.diagnostics import ranking_diagnostics
        self.assertEqual(ranking_diagnostics([{"label": 0, "score": .5}, {"label": 1, "score": .5}])["roc_auc"], .5)
        self.assertEqual(ranking_diagnostics([{"label": 0, "score": 0}, {"label": 1, "score": 1}])["roc_auc"], 1)
        self.assertEqual(ranking_diagnostics([{"label": 0, "score": 1}, {"label": 1, "score": 0}])["roc_auc"], 0)
        self.assertIsNone(ranking_diagnostics([{"label": 1, "score": .5}])["roc_auc"])

    def test_score_histogram_places_threshold_edges_once_and_keeps_empty_bins(self):
        from mcp_shield.diagnostics import score_histogram
        rows = [{"label": 1, "score": s} for s in (0, .5, .65, 1)]
        result = score_histogram(rows)
        self.assertEqual([b["rows"] for b in result], [1, 0, 1, 1, 1])
        self.assertIsNone(result[1]["mean_score"])
        self.assertEqual(sum(b["ml_errors"] for b in result), 1)

    def test_probability_losses_handle_exact_endpoint_errors_without_nan(self):
        from mcp_shield.diagnostics import probability_losses
        perfect = probability_losses([{"label": 0, "score": 0}, {"label": 1, "score": 1}])
        self.assertEqual(perfect["brier"], 0)
        self.assertEqual(perfect["log_loss"], 0)
        wrong = probability_losses([{"label": 1, "score": 0}, {"label": 0, "score": 1}])
        self.assertEqual(wrong["brier"], 1)
        self.assertIsNone(wrong["log_loss"])
        self.assertEqual(wrong["infinite_log_loss_rows"], 2)

    def test_repeated_seeds_preserve_manifests_without_selecting_a_winner(self):
        from mcp_shield.evaluation import evaluate_seeds
        report = evaluate_seeds([17, 18])
        self.assertEqual([r["seed"] for r in report["runs"]], [17, 18])
        self.assertNotEqual(report["runs"][0]["heldout_sha256"], report["runs"][1]["heldout_sha256"])
        self.assertNotIn("best_seed", report)
        for seeds in ([], [17, 17], [True], list(range(11))):
            with self.assertRaises(ValueError):
                evaluate_seeds(seeds)

    def test_report_comparison_rejects_unmatched_data_and_inconsistent_counts(self):
        import copy
        from mcp_shield.evaluation import compare_reports, evaluate
        a = evaluate()
        b = copy.deepcopy(a)
        b["ml"]["tp"] += 1
        b["ml"]["fn"] -= 1
        self.assertEqual(compare_reports(a, b)["count_deltas"]["ml"]["fn"], -1)
        with self.assertRaisesRegex(ValueError, "identical heldout"):
            compare_reports(a, evaluate(18))
        b["ml"]["tp"] += 1
        with self.assertRaisesRegex(ValueError, "reconcile"):
            compare_reports(a, b)

    def test_csv_predictions_preserve_counts_quote_text_and_neutralize_formulas(self):
        import csv
        import io
        from mcp_shield.exports import predictions_csv
        row = {"id": "=fictional", "family": "toy, family", "label": 1, "score": .5,
               "ml_prediction": 1, "decision": "deny", "gates": ["scope_violation"], "review_triggers": []}
        parsed = list(csv.DictReader(io.StringIO(predictions_csv([row]))))
        self.assertEqual(parsed[0]["id"], "'=fictional")
        self.assertEqual(parsed[0]["family"], row["family"])
        self.assertEqual(parsed[0]["label"], "1")
        self.assertIn("Synthetic", parsed[0]["boundary"])
