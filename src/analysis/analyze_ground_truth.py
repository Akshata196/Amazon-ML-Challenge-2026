from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

path = BASE / "dataset/train/train_ground_truth.tsv"

counts = []

for chunk in pd.read_csv(
    path,
    sep="\t",
    chunksize=200_000,
    dtype=str
):
    # Empty / missing means no match
    match_counts = (
        chunk["matched_entity_ids"]
        .fillna("")
        .str.strip()
        .apply(lambda x: 0 if not x else len(x.split(",")))
    )

    counts.extend(match_counts.tolist())

s = pd.Series(counts)

print("\n" + "=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)

print("Total Source 1 entities:", len(s))

print("\nMatch count distribution:")
print(s.value_counts().sort_index())

print("\nStatistics:")
print(s.describe())

print("\nSingletons / no-match:")
print("No matches:", (s == 0).sum())
print("Percentage:", f"{(s == 0).mean() * 100:.2f}%")

print("\nAt least one match:")
print("Entities:", (s > 0).sum())
print("Percentage:", f"{(s > 0).mean() * 100:.2f}%")

print("\nMultiple matches:")
print("Entities:", (s > 1).sum())
print("Percentage:", f"{(s > 1).mean() * 100:.2f}%")