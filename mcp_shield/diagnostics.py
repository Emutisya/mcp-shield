"""Descriptive diagnostics; never tune policy using held-out labels."""

import math
from .evaluation import metrics
from .core import FEATURES, features


def labelled_scores(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Diagnostics require a nonempty list of labelled scores.")
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("label")) is not int
                or row["label"] not in (0, 1) or type(row.get("score")) not in (int, float)
                or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
            raise ValueError("Each diagnostic row needs label 0/1 and a finite score in [0,1].")
    return rows


def threshold_sweep(rows, thresholds=(0, .25, .5, .65, .75, 1)):
    labelled_scores(rows)
    if (not isinstance(thresholds, (tuple, list)) or not 1 <= len(thresholds) <= 101
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds)
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("Supply 1–101 distinct finite thresholds in [0,1].")
    return [{"threshold": t, **metrics([(r["label"], int(r["score"] >= t)) for r in rows])}
            for t in sorted(thresholds)]


def decision_burden(rows):
    labelled_scores(rows)
    if any(r.get("decision") not in ("allow", "review", "deny") for r in rows):
        raise ValueError("Each burden row requires an allow, review or deny decision.")
    return {decision: {
        "rows": sum(r["decision"] == decision for r in rows),
        "benign_labels": sum(r["decision"] == decision and r["label"] == 0 for r in rows),
        "positive_labels": sum(r["decision"] == decision and r["label"] == 1 for r in rows),
        "fraction": round(sum(r["decision"] == decision for r in rows) / len(rows), 4),
    } for decision in ("allow", "review", "deny")}


def classifier_policy_cross_tab(rows):
    decision_burden(rows)
    return {str(prediction): {decision: {
        "rows": sum(int(r["score"] >= .5) == prediction and r["decision"] == decision for r in rows),
        "positive_labels": sum(r["label"] for r in rows
                               if int(r["score"] >= .5) == prediction and r["decision"] == decision),
    } for decision in ("allow", "review", "deny")} for prediction in (0, 1)}


def feature_distribution(training, heldout):
    if any(not isinstance(group, list) or not group for group in (training, heldout)):
        raise ValueError("Feature distribution requires nonempty observation lists in both partitions.")
    a, b = ([features(value) for value in group] for group in (training, heldout))
    return {name: {"training_present": sum(v[j] for v in a), "heldout_present": sum(v[j] for v in b),
                  "training_fraction": round(sum(v[j] for v in a) / len(a), 4),
                  "heldout_fraction": round(sum(v[j] for v in b) / len(b), 4),
                  "fraction_delta": round(sum(v[j] for v in b) / len(b) - sum(v[j] for v in a) / len(a), 4)}
            for j, name in enumerate(FEATURES)}


def feature_dependence(rows):
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(r, dict) or type(r.get("label")) is not int or r["label"] not in (0, 1) for r in rows)):
        raise ValueError("Dependence diagnostics require labelled observation rows.")
    vectors = [(r["label"], features(r["input"])) for r in rows]
    result = []
    for label in (0, 1):
        group = [v for y, v in vectors if y == label]
        for j in range(len(FEATURES)):
            for k in range(j + 1, len(FEATURES)):
                cells = [sum(v[j] == a and v[k] == b for v in group) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
                n00, n01, n10, n11 = cells
                denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
                result.append({"label": label, "left": FEATURES[j], "right": FEATURES[k], "rows": len(group),
                               "cells_00_01_10_11": cells,
                               "phi": round((n11 * n00 - n10 * n01) / denominator, 4) if denominator else None})
    return result
