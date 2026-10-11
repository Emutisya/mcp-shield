"""Human-readable and machine-readable offline evidence exports."""


def inspection_text(result):
    lines = [result["warning"], f"Advisory decision: {result['decision'].upper()}",
             f"Uncalibrated synthetic-trained score: {result['uncalibrated_ml_probability']:.12g}",
             "Hard gates: " + (", ".join(result["gates"]) or "none observed"),
             "Review triggers: " + (", ".join(result["review_triggers"]) or "none observed"),
             *result["reasons"], "Feature contributions (learned associations, not causal evidence):"]
    lines.extend(f"  {row['feature']}: {'present' if row['present'] else 'absent'} "
                 f"{row['raw_log_odds_contribution']:+.6f} log-odds" for row in result["contributions"])
    return "\n".join(lines) + "\n"
