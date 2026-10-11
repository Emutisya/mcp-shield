"""Descriptive diagnostics; never tune policy using held-out labels."""

import math
from .evaluation import metrics


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
