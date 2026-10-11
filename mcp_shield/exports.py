"""Human-readable and machine-readable offline evidence exports."""

import csv
import io


def inspection_text(result):
    lines = [result["warning"], f"Advisory decision: {result['decision'].upper()}",
             f"Uncalibrated synthetic-trained score: {result['uncalibrated_ml_probability']:.12g}",
             "Hard gates: " + (", ".join(result["gates"]) or "none observed"),
             "Review triggers: " + (", ".join(result["review_triggers"]) or "none observed"),
             *result["reasons"], "Feature contributions (learned associations, not causal evidence):"]
    lines.extend(f"  {row['feature']}: {'present' if row['present'] else 'absent'} "
                 f"{row['raw_log_odds_contribution']:+.6f} log-odds" for row in result["contributions"])
    return "\n".join(lines) + "\n"


def predictions_csv(rows):
    stream = io.StringIO(newline="")
    fields = ("id", "family", "label", "score", "ml_prediction", "decision", "gates", "review_triggers", "boundary")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        record = {key: row[key] for key in fields if key != "boundary"}
        for key in ("gates", "review_triggers"):
            record[key] = ";".join(record[key])
        record["boundary"] = "Synthetic toy labels only; not independently verified attacks."
        for key, value in record.items():
            if isinstance(value, str) and value.lstrip()[:1] in ("=", "+", "-", "@"):
                record[key] = "'" + value
        writer.writerow(record)
    return stream.getvalue()


def evaluation_text(report):
    lines = [report["data"], f"Seed: {report['seed']}; train rows: {report['train_rows']}; held-out rows: {report['test_rows']}",
             f"Training SHA-256: {report['training_sha256']}", f"Held-out SHA-256: {report['heldout_sha256']}",
             "Method | TP FP TN FN | Accuracy Precision Recall"]
    for method in ("always_benign_baseline", "hard_gates_only_baseline", "ml", "policy_nonallow_as_positive"):
        group = report[method]
        counts = " ".join(str(group[k]) for k in ("tp", "fp", "tn", "fn"))
        values = " ".join("undefined" if group[k] is None else f"{group[k]:.4f}" for k in ("accuracy", "precision", "recall"))
        lines.append(f"{method} | {counts} | {values}")
    lines.append("Advisory workload (review is not denial):")
    for decision, group in report["decision_burden"].items():
        lines.append(f"  {decision}: {group['rows']} rows; {group['benign_labels']} benign toy labels; {group['positive_labels']} positive toy labels")
    lines.append("Held-out family ML false positives / false negatives:")
    lines.extend(f"  {family}: {group['ml']['fp']} / {group['ml']['fn']}" for family, group in report["per_family"].items())
    lines.extend(report["limitations"])
    return "\n".join(lines) + "\n"
