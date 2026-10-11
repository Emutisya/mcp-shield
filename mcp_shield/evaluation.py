"""Held-out synthetic evaluation separating ML classification from policy."""

import hashlib
import json
from .core import inspect, train
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
            "accuracy": round((tp + tn) / len(pairs), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def evaluate(seed=17):
    training, heldout = split(dataset(seed))
    model = train(training)
    results = [(r, inspect(r["input"], model)) for r in heldout]
    failures = [{"id": r["id"], "label": r["label"], "decision": p["decision"],
                 "score": p["uncalibrated_ml_probability"]}
                for r, p in results if (p["uncalibrated_ml_probability"] >= 0.5) != bool(r["label"])]
    per_family = {}
    for family in sorted({row["family"] for row, _ in results}):
        group = [(row, prediction) for row, prediction in results if row["family"] == family]
        per_family[family] = {
            "rows": len(group), "positive_labels": sum(row["label"] for row, _ in group),
            "ml": metrics([(row["label"], int(prediction["uncalibrated_ml_probability"] >= .5))
                           for row, prediction in group]),
            "policy_decisions": {decision: sum(prediction["decision"] == decision for _, prediction in group)
                                 for decision in ("allow", "review", "deny")},
        }
    return {"data": "Explicitly synthetic; latent toy labels, not verified attacks.",
            "training_sha256": hashlib.sha256(json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "heldout_sha256": hashlib.sha256(json.dumps(heldout, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "seed": seed, "train_rows": len(training), "test_rows": len(heldout),
            "split": "Whole families held out; description templates disjoint; names/tags excluded from features.",
            "train_families": sorted({r["family"] for r in training}),
            "test_families": sorted({r["family"] for r in heldout}),
            "ml_threshold": 0.5,
            "ml": metrics([(r["label"], int(p["uncalibrated_ml_probability"] >= 0.5)) for r, p in results]),
            "always_benign_baseline": metrics([(r["label"], 0) for r in heldout]),
            "policy_nonallow_as_positive": metrics([(r["label"], int(p["decision"] != "allow")) for r, p in results]),
            "policy_decisions": {d: sum(p["decision"] == d for _, p in results) for d in ("allow", "review", "deny")},
            "ml_failure_count": len(failures), "first_12_ml_failures": failures[:12],
            "per_family": per_family,
            "limitations": [
                "All numbers measure toy synthetic distributions, not real-world detection.",
                "Signal provenance is trusted; unseen attacks and dishonest signals can pass.",
                "Feature vectors overlap across splits intentionally, but source templates do not.",
                "Scores are uncalibrated; no calibration or real attack corpus is available.",
                "Review is not deny; binary policy metrics count both as nonallow.",
            ]}
