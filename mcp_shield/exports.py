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
