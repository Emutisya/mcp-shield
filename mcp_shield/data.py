"""Generated toy data: disjoint language families, never exploit strings."""

import random

FAMILIES = (
    ("ledger", "Summarize accounting entries", "Inspect private ledger notes"),
    ("garden", "Catalog plant observations", "Review instruction annotations for plants"),
    ("archive", "Index historical records", "Summarize secret archive classifications"),
    ("weather", "List weather observations", "Review priority weather directives"),
    ("library", "Find book titles", "Describe credential labels in catalog records"),
    ("transit", "Summarize route schedules", "Inspect override labels on route data"),
    ("museum", "List public exhibits", "Review token annotations on exhibits"),
    ("harbor", "Catalog vessel schedules", "Inspect private instruction classifications"),
    ("school", "Summarize class calendars", "Review directive notes in school calendars"),
    ("energy", "List meter readings", "Describe secret annotations on meter readings"),
    ("clinic", "Catalog fictional appointment slots", "Review priority token labels on slots"),
    ("forest", "Summarize tree counts", "Inspect override notes in tree classifications"),
)


def dataset(seed=17, per_family=100):
    rng = random.Random(seed)
    rows = []
    for i, (family, ordinary, unusual) in enumerate(FAMILIES):
        for j in range(per_family):
            # Latent labels overlap deliberately: signals can miss harm or flag benign work.
            label = int(rng.random() < 0.45)
            flags = {k: rng.random() < (0.25 if label else 0.035) for k in
                     ("credential_request", "scope_violation", "hidden_instruction", "unapproved_network")}
            signals = {**flags, "verified_publisher": rng.random() < (0.6 if label else 0.9),
                       "metadata_changes": rng.choice([0, 0, 1, 2] if label else [0, 0, 0, 1]),
                       "unexpected_calls": rng.choice([0, 0, 3, 5] if label else [0, 0, 0, 1]),
                       "response_bytes": rng.choice([512, 4096, 90000] if label else [512, 4096, 4096, 90000])}
            description = unusual if rng.random() < (0.6 if label else 0.15) else ordinary
            # Each row is distinct; names and sample tags are intentionally not ML features.
            value = {"server": {"name": f"synthetic-{family}-{j}", "publisher": "Fictional local lab"},
                     "tool": {"name": f"inspect_{family}", "description": f"{description}. Synthetic case {i}-{j}."},
                     "signals": signals}
            rows.append({"id": f"{family}-{j}", "family": family, "label": label, "input": value,
                         "split": "train" if i < 8 else "test"})
    return rows


def split(rows):
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    assert not {r["family"] for r in train_rows} & {r["family"] for r in test_rows}
    return train_rows, test_rows
