"""Bounded offline analysis of synthetic caller-supplied observations."""

import json
import hashlib
import math
from pathlib import Path

from .core import inspect, unique_fields, validate, validate_model

WARNING = "Offline synthetic simulation only; observations are caller supplied, not independently verified."
MAX_BATCH = 100
MAX_DOCUMENT = 2 * 1024 * 1024


def read_document(path, maximum=MAX_DOCUMENT):
    with Path(path).open("rb") as source:
        raw = source.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError(f"JSON document must be 1 to {maximum} bytes.")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)


def inspect_batch(values, model):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_BATCH:
        raise ValueError(f"Batch requires 1 to {MAX_BATCH} observations.")
    for index, value in enumerate(values):
        try:
            validate(value)
        except ValueError as exc:
            raise ValueError(f"Batch item {index}: {exc}") from exc
    results = [inspect(value, model) for value in values]
    return {"warning": WARNING, "rows": len(results), "results": results,
            "model_sha256": fingerprint(validate_model(model)),
            "observation_sha256": [fingerprint(value) for value in values],
            "decisions": {d: sum(r["decision"] == d for r in results)
                          for d in ("allow", "review", "deny")}}


def fingerprint(value):
    """Canonical content identity, not authentication or independent verification."""
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def receipt(value, model):
    result = inspect(value, model)
    return {"warning": WARNING, "format_version": 1,
            "observation_sha256": fingerprint(value), "model_sha256": fingerprint(model),
            "result": result, "provenance": "Unsigned content hashes; not evidence of source authenticity."}


def model_info(model):
    validate_model(model)
    return {"warning": WARNING, "model_sha256": fingerprint(model),
            "version": model["version"], "training_rows": model["training_rows"],
            "synthetic_class_prior": model["prior"],
            "prior_log_odds": math.log(model["prior"]) - math.log1p(-model["prior"]),
            "features": [{
                "feature": name, "benign_present_probability": model["probabilities"][0][j],
                "positive_present_probability": model["probabilities"][1][j],
                "present_log_odds": round(math.log(model["probabilities"][1][j]) - math.log(model["probabilities"][0][j]), 12),
                "absent_log_odds": round(math.log1p(-model["probabilities"][1][j]) - math.log1p(-model["probabilities"][0][j]), 12),
            } for j, name in enumerate(model["features"])]}


def compare_models(before, after):
    a, b = model_info(before), model_info(after)
    return {"warning": WARNING, "before_sha256": a["model_sha256"], "after_sha256": b["model_sha256"],
            "training_rows_delta": after["training_rows"] - before["training_rows"],
            "prior_delta": after["prior"] - before["prior"],
            "features": [{"feature": x["feature"],
                          "present_log_odds_delta": round(y["present_log_odds"] - x["present_log_odds"], 12),
                          "absent_log_odds_delta": round(y["absent_log_odds"] - x["absent_log_odds"], 12)}
                         for x, y in zip(a["features"], b["features"])]}
