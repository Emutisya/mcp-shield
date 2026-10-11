"""Strict inspection schema, Bernoulli naive Bayes, and deterministic policy."""

import json
import math
import re
from pathlib import Path

FLAGS = ("credential_request", "scope_violation", "hidden_instruction", "unapproved_network")
COUNTS = {"metadata_changes": 20, "unexpected_calls": 100, "response_bytes": 1000000}
SIGNALS = set(FLAGS) | set(COUNTS) | {"verified_publisher"}
FEATURES = (*FLAGS, "unverified_publisher", "metadata_drift", "call_anomaly",
            "large_response", "instruction_language", "sensitive_language")
MODEL_VERSION = 1
MAX_MODEL_BYTES = 65536
VOCABULARY = {"instruction_language": {"instruction", "override", "directive", "priority"},
              "sensitive_language": {"credential", "secret", "token", "private"}}


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not accepted.")
        result[key] = value
    return result


def validate(value):
    if not isinstance(value, dict) or set(value) != {"server", "tool", "signals"}:
        raise ValueError("Expected exactly server, tool, and signals objects.")
    for key, fields in (("server", {"name", "publisher"}), ("tool", {"name", "description"})):
        item = value[key]
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{key} must contain exactly {', '.join(sorted(fields))}.")
        for field, text in item.items():
            limit = 2000 if field == "description" else 120
            if not isinstance(text, str) or not text.strip() or len(text) > limit:
                raise ValueError(f"{key}.{field} must be nonempty text, at most {limit} characters.")
            if any((ord(c) < 32 and c not in "\n\t") or 127 <= ord(c) <= 159
                   or 0xD800 <= ord(c) <= 0xDFFF for c in text):
                raise ValueError(f"{key}.{field} contains unsupported control characters.")
    signals = value["signals"]
    if not isinstance(signals, dict) or set(signals) != SIGNALS:
        raise ValueError(f"signals must contain exactly {', '.join(sorted(SIGNALS))}.")
    for key in (*FLAGS, "verified_publisher"):
        if type(signals[key]) is not bool:
            raise ValueError(f"signals.{key} must be a JSON boolean.")
    for key, maximum in COUNTS.items():
        if type(signals[key]) is not int or not 0 <= signals[key] <= maximum:
            raise ValueError(f"signals.{key} must be an integer from 0 to {maximum}.")
    return value


def language_cues(value):
    validate(value)
    words = set(re.findall(r"[a-z]+", value["tool"]["description"].lower()))
    return {name: sorted(words & vocabulary) for name, vocabulary in VOCABULARY.items()}


def features(value):
    cues = language_cues(value)
    s = value["signals"]
    return [int(s[k]) for k in FLAGS] + [
        int(not s["verified_publisher"]), int(s["metadata_changes"] > 0),
        int(s["unexpected_calls"] > 2), int(s["response_bytes"] > 65536),
        int(bool(cues["instruction_language"])),
        int(bool(cues["sensitive_language"])),
    ]


def train(rows):
    """Laplace-smoothed Bernoulli NB; parameters learned only from supplied rows."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Training requires a nonempty sequence of labelled examples.")
    counts = [0, 0]
    positives = [[0] * len(FEATURES) for _ in range(2)]
    for row in rows:
        if (not isinstance(row, dict) or "input" not in row
                or type(row.get("label")) is not int or row["label"] not in (0, 1)):
            raise ValueError("Each training example requires an input and an integer label of 0 or 1.")
        label = row["label"]
        counts[label] += 1
        for j, x in enumerate(features(row["input"])):
            positives[label][j] += x
    if not all(counts):
        raise ValueError("Training requires examples of both classes.")
    return {
        "version": MODEL_VERSION, "features": list(FEATURES),
        "prior": (counts[1] + 1) / (sum(counts) + 2),
        "probabilities": [[(n + 1) / (counts[c] + 2) for n in positives[c]] for c in range(2)],
        "training_rows": sum(counts),
    }


def validate_model(model):
    if (not isinstance(model, dict)
            or set(model) != {"version", "features", "prior", "probabilities", "training_rows"}
            or type(model.get("version")) is not int
            or model.get("version") != MODEL_VERSION or model.get("features") != list(FEATURES)):
        raise ValueError("Unsupported model version or feature layout. Retrain the model.")
    if type(model.get("training_rows")) is not int or model["training_rows"] < 2:
        raise ValueError("Model training_rows must be an integer of at least two.")
    p = model.get("probabilities")
    if not isinstance(p, list) or len(p) != 2 or any(not isinstance(a, list) or len(a) != len(FEATURES) for a in p):
        raise ValueError("Invalid model probability dimensions.")
    for v in [model.get("prior")] + [v for a in p for v in a]:
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1:
            raise ValueError("Model probabilities must be finite and between zero and one.")
    return model


def load_model(path):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_MODEL_BYTES + 1)
    if not 0 < len(raw) <= MAX_MODEL_BYTES:
        raise ValueError(f"Model JSON must be 1 to {MAX_MODEL_BYTES} bytes. Retrain the model.")
    return validate_model(json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields))


def inspect(value, model):
    x = features(value)
    validate_model(model)
    log_odds = math.log(model["prior"]) - math.log1p(-model["prior"])
    prior_log_odds = log_odds
    explanations = []
    for j, present in enumerate(x):
        benign, risky = (a[j] for a in model["probabilities"])
        contribution = (math.log(risky) - math.log(benign) if present
                        else math.log1p(-risky) - math.log1p(-benign))
        log_odds += contribution
        explanations.append((contribution, {"feature": FEATURES[j], "present": bool(present),
                             "log_odds_contribution": round(contribution, 4),
                             "raw_log_odds_contribution": contribution}))
    probability = 1 / (1 + math.exp(-max(-700, min(700, log_odds))))
    gates = [key for key in FLAGS if value["signals"][key]]
    review_triggers = []
    if probability >= .65:
        review_triggers.append("ml_threshold")
    if value["signals"]["metadata_changes"]:
        review_triggers.append("metadata_drift")
    if value["signals"]["unexpected_calls"]:
        review_triggers.append("unexpected_calls")
    if not value["signals"]["verified_publisher"]:
        review_triggers.append("unverified_publisher")
    reasons = [f"Hard deny gate: {key}." for key in gates]
    if gates:
        decision = "deny"
    elif probability >= 0.65:
        decision = "review"
        reasons.append("ML score is above the conservative review threshold (0.65).")
    elif value["signals"]["metadata_changes"] or value["signals"]["unexpected_calls"]:
        decision = "review"
        reasons.append("Metadata drift or unexpected calls require human review.")
    elif not value["signals"]["verified_publisher"]:
        decision = "review"
        reasons.append("Publisher is not verified by the caller's observation source.")
    else:
        decision = "allow"
        reasons.append("No observed policy gate fired; this is not a safety guarantee.")
    return {"decision": decision, "uncalibrated_ml_probability": probability,
            "prior_log_odds": prior_log_odds, "posterior_log_odds": log_odds,
            "language_cues": language_cues(value),
            "reasons": reasons, "gates": gates,
            "review_triggers": review_triggers,
            "contributions": [row for _, row in sorted(explanations, key=lambda item: abs(item[0]), reverse=True)],
            "warning": "Synthetic-trained offline simulator. Caller-supplied signals are not independently verified."}
