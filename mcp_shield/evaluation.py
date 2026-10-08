"""Held-out synthetic evaluation separating ML classification from policy."""

from .core import inspect, train
from .data import dataset, split


def metrics(pairs):
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
    return {"data": "Explicitly synthetic; latent toy labels, not verified attacks.",
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
            "limitations": [
                "All numbers measure toy synthetic distributions, not real-world detection.",
                "Signal provenance is trusted; unseen attacks and dishonest signals can pass.",
                "Feature vectors overlap across splits intentionally, but source templates do not.",
                "Scores are uncalibrated; no calibration or real attack corpus is available.",
                "Review is not deny; binary policy metrics count both as nonallow.",
            ]}
