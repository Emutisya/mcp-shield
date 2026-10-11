"""Bounded offline analysis of synthetic caller-supplied observations."""

import json
import hashlib
import math
import copy
from pathlib import Path

from .core import FEATURES, FLAGS, features, inspect, language_cues, unique_fields, validate, validate_model

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
            "feature_groups": feature_groups(values, results),
            "decisions": {d: sum(r["decision"] == d for r in results)
                          for d in ("allow", "review", "deny")}}


def feature_groups(values, results):
    vectors = [features(v) for v in values]
    return {name: {str(present): {
        "rows": sum(vector[j] == present for vector in vectors),
        "decisions": {d: sum(vector[j] == present and result["decision"] == d
                             for vector, result in zip(vectors, results)) for d in ("allow", "review", "deny")},
    } for present in (0, 1)} for j, name in enumerate(FEATURES)}


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


def compare_observations(values, model):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("Comparison requires exactly two complete synthetic observations.")
    batch = inspect_batch(values, model)
    a, b = batch["results"]
    x, y = (features(v) for v in values)
    return {"warning": WARNING, "model_sha256": batch["model_sha256"],
            "observation_sha256": batch["observation_sha256"], "before": a, "after": b,
            "probability_delta": b["uncalibrated_ml_probability"] - a["uncalibrated_ml_probability"],
            "changed_features": [name for name, left, right in zip(FEATURES, x, y) if left != right],
            "added_gates": sorted(set(b["gates"]) - set(a["gates"])),
            "removed_gates": sorted(set(a["gates"]) - set(b["gates"]))}


def sensitivity(value, model):
    baseline = inspect(value, model)
    variants = []
    for signal in (*FLAGS, "verified_publisher"):
        hypothetical = copy.deepcopy(value)
        hypothetical["signals"][signal] = not hypothetical["signals"][signal]
        result = inspect(hypothetical, model)
        variants.append({"signal": signal, "hypothetical_value": hypothetical["signals"][signal],
                         "result": result,
                         "probability_delta": result["uncalibrated_ml_probability"] - baseline["uncalibrated_ml_probability"]})
    return {"warning": WARNING, "boundary": "One-variable synthetic sensitivity; not causal evidence or remediation advice. Never falsify observed signals.",
            "baseline": baseline, "variants": variants}


def count_boundaries(value, model):
    baseline = inspect(value, model)
    candidates = {"metadata_changes": (0, 1, 20), "unexpected_calls": (0, 1, 2, 3, 100),
                  "response_bytes": (0, 65535, 65536, 65537, 1000000)}
    groups = {}
    for signal, counts in candidates.items():
        groups[signal] = []
        for count in counts:
            hypothetical = copy.deepcopy(value)
            hypothetical["signals"][signal] = count
            groups[signal].append({"hypothetical_value": count, "result": inspect(hypothetical, model)})
    return {"warning": WARNING, "boundary": "Numeric boundary simulations only; not rate estimates or remediation advice.",
            "baseline": baseline, "groups": groups}


def projection(value):
    vector = features(value)
    return {"warning": WARNING, "observation_sha256": fingerprint(value),
            "features": dict(zip(FEATURES, vector)),
            "excluded_fields": ["server.name", "server.publisher", "tool.name"],
            "language_cues": language_cues(value)}


def validation_report(values):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_BATCH:
        raise ValueError(f"Validation requires 1 to {MAX_BATCH} observations.")
    errors = []
    for index, value in enumerate(values):
        try:
            validate(value)
        except ValueError as exc:
            errors.append({"index": index, "error": str(exc)})
    return {"warning": WARNING, "rows": len(values), "valid_rows": len(values) - len(errors),
            "invalid_rows": len(errors), "errors": errors,
            "boundary": "Schema validity is not safety, source verification or an inspection verdict."}


def representation_audit(rows):
    from .data import split
    training, heldout = split(rows)
    groups = []
    for partition in (training, heldout):
        vectors = {}
        for row in partition:
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("Representation audit requires binary integer labels.")
            vector = tuple(features(row["input"]))
            vectors.setdefault(vector, set()).add(row["label"])
        groups.append(vectors)
    return {"warning": WARNING, "training_unique_vectors": len(groups[0]), "heldout_unique_vectors": len(groups[1]),
            "shared_feature_vectors": len(groups[0].keys() & groups[1].keys()),
            "training_mixed_label_vectors": sum(len(labels) > 1 for labels in groups[0].values()),
            "heldout_mixed_label_vectors": sum(len(labels) > 1 for labels in groups[1].values()),
            "boundary": "Binary representation overlap is intentional; mixed toy labels show information the features cannot resolve."}
