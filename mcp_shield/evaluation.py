"""Held-out synthetic evaluation separating ML classification from policy."""

import hashlib
import json
from .core import FLAGS, inspect, train
from .data import dataset, split


def metrics(pairs):
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("Metrics require at least one labelled prediction.")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           or any(type(value) is not int or value not in (0, 1) for value in pair)
           for pair in pairs):
        raise ValueError("Metric labels and predictions must be integers of 0 or 1.")
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "support": len(pairs), "positive_labels": tp + fn,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
            "false_positive_rate": round(fp / (tn + fp), 4) if tn + fp else None,
            "false_negative_rate": round(fn / (tp + fn), 4) if tp + fn else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
            "balanced_accuracy": round((tp / (tp + fn) + tn / (tn + fp)) / 2, 4)
            if tp + fn and tn + fp else None,
            "accuracy": round((tp + tn) / len(pairs), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def evaluate(seed=17, include_predictions=False):
    from .diagnostics import classifier_policy_cross_tab, decision_burden, feature_dependence, feature_distribution, threshold_sweep
    if type(include_predictions) is not bool:
        raise ValueError("include_predictions must be boolean.")
    training, heldout = split(dataset(seed))
    model = train(training)
    results = [(r, inspect(r["input"], model)) for r in heldout]
    gate_diagnostics = {flag: {
        "rows": sum(flag in p["gates"] for _, p in results),
        "benign_labels": sum(flag in p["gates"] and r["label"] == 0 for r, p in results),
        "positive_labels": sum(flag in p["gates"] and r["label"] == 1 for r, p in results),
        "exclusive_rows": sum(p["gates"] == [flag] for _, p in results),
    } for flag in FLAGS}
    failures = [{"id": r["id"], "label": r["label"], "decision": p["decision"],
                 "score": round(p["uncalibrated_ml_probability"], 12)}
                for r, p in results if (p["uncalibrated_ml_probability"] >= 0.5) != bool(r["label"])]
    per_family = {}
    for family in sorted({row["family"] for row, _ in results}):
        group = [(row, prediction) for row, prediction in results if row["family"] == family]
        per_family[family] = {
            "rows": len(group), "positive_labels": sum(row["label"] for row, _ in group),
            "ml": metrics([(row["label"], int(prediction["uncalibrated_ml_probability"] >= .5))
                           for row, prediction in group]),
            "policy_nonallow_as_positive": metrics([(row["label"], int(prediction["decision"] != "allow"))
                                                   for row, prediction in group]),
            "decision_burden": decision_burden([{"label": row["label"], "score": prediction["uncalibrated_ml_probability"],
                                               "decision": prediction["decision"]} for row, prediction in group]),
            "policy_decisions": {decision: sum(prediction["decision"] == decision for _, prediction in group)
                                 for decision in ("allow", "review", "deny")},
        }
    report = {"data": "Explicitly synthetic; latent toy labels, not verified attacks.",
            "training_sha256": hashlib.sha256(json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "heldout_sha256": hashlib.sha256(json.dumps(heldout, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "seed": seed, "train_rows": len(training), "test_rows": len(heldout),
            "split": "Whole families held out; description templates disjoint; names/tags excluded from features.",
            "train_families": sorted({r["family"] for r in training}),
            "test_families": sorted({r["family"] for r in heldout}),
            "ml_threshold": 0.5,
            "display_score_decimal_places": 12,
            "ml": metrics([(r["label"], int(p["uncalibrated_ml_probability"] >= 0.5)) for r, p in results]),
            "always_benign_baseline": metrics([(r["label"], 0) for r in heldout]),
            "hard_gates_only_baseline": metrics([(r["label"], int(any(r["input"]["signals"][f] for f in FLAGS)))
                                                for r in heldout]),
            "always_positive_baseline": metrics([(r["label"], 1) for r in heldout]),
            "policy_nonallow_as_positive": metrics([(r["label"], int(p["decision"] != "allow")) for r, p in results]),
            "policy_decisions": {d: sum(p["decision"] == d for _, p in results) for d in ("allow", "review", "deny")},
            "decision_burden": decision_burden([{"label": r["label"], "score": p["uncalibrated_ml_probability"],
                                                "decision": p["decision"]} for r, p in results]),
            "classifier_policy_cross_tab": classifier_policy_cross_tab([
                {"label": r["label"], "score": p["uncalibrated_ml_probability"], "decision": p["decision"]}
                for r, p in results]),
            "gate_diagnostics": gate_diagnostics,
            "feature_distribution": feature_distribution([r["input"] for r in training], [r["input"] for r in heldout]),
            "training_feature_dependence": feature_dependence(training),
            "gate_count_histogram": {str(count): sum(len(p["gates"]) == count for _, p in results)
                                     for count in range(len(FLAGS) + 1)},
            "ml_failure_count": len(failures), "first_12_ml_failures": failures[:12],
            "per_family": per_family,
            "threshold_sweep": threshold_sweep([{"label": r["label"], "score": p["uncalibrated_ml_probability"]}
                                                for r, p in results]),
            "limitations": [
                "All numbers measure toy synthetic distributions, not real-world detection.",
                "Signal provenance is trusted; unseen attacks and dishonest signals can pass.",
                "Feature vectors overlap across splits intentionally, but source templates do not.",
                "Scores are uncalibrated; no calibration or real attack corpus is available.",
                "Review is not deny; binary policy metrics count both as nonallow.",
            ]}
    if include_predictions:
        report["predictions"] = [{"id": r["id"], "family": r["family"], "label": r["label"],
                                  "score": round(p["uncalibrated_ml_probability"], 12),
                                  "ml_prediction": int(p["uncalibrated_ml_probability"] >= .5),
                                  "decision": p["decision"], "gates": p["gates"],
                                  "review_triggers": p["review_triggers"]} for r, p in results]
    return report
