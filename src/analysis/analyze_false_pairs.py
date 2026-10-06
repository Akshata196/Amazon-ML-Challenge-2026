from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio
import re
import unicodedata
import random

# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dataset"

SAMPLE_S1 = 5000
NEGATIVES_PER_S1 = 2
CHUNK_SIZE = 200_000
POOL_SIZE_PER_COUNTRY = 5000

random.seed(42)
np.random.seed(42)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode-safe normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_DIR = DATASET / "train"

s1_path = TRAIN_DIR / "train_source1.tsv"
s2_path = TRAIN_DIR / "train_source2.tsv"
s3_path = TRAIN_DIR / "train_source3.tsv"
gt_path = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# STEP 1: SAMPLE SOURCE 1
# ============================================================

print("=" * 70)
print("STEP 1: Sampling Source 1")
print("=" * 70)

s1_sample = pd.read_csv(
    s1_path,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_S1
)

print(f"Source-1 sample: {len(s1_sample)}")


# ============================================================
# STEP 2: GET GROUND-TRUTH MATCHES FOR SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Loading ground-truth matches")
print("=" * 70)

sample_ids = set(s1_sample["entity_id"])

true_matches = {}

for chunk in pd.read_csv(
    gt_path,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):
    filtered = chunk[
        chunk["source1_entity_id"].isin(sample_ids)
    ]

    for _, row in filtered.iterrows():
        matched = str(row["matched_entity_ids"])

        if matched == "nan" or matched.strip() == "":
            true_matches[row["source1_entity_id"]] = set()
        else:
            true_matches[row["source1_entity_id"]] = set(
                x.strip()
                for x in matched.split(",")
                if x.strip()
            )

print(f"Ground-truth entries found: {len(true_matches)}")


# ============================================================
# STEP 3: BUILD COUNTRY-SPECIFIC RANDOM NEGATIVE POOLS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Building negative candidate pools")
print("=" * 70)


def build_country_pool(file_path, source_name):
    pools = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        for country, group in chunk.groupby("country", dropna=False):

            country_key = str(country)

            if country_key not in pools:
                pools[country_key] = []

            remaining = POOL_SIZE_PER_COUNTRY - len(pools[country_key])

            if remaining <= 0:
                continue

            records = group.sample(
                n=min(remaining, len(group)),
                random_state=42
            )

            pools[country_key].extend(
                records.to_dict("records")
            )

    print(
        f"{source_name}: "
        + ", ".join(
            f"{country}={len(rows)}"
            for country, rows in pools.items()
        )
    )

    return pools


s2_pool = build_country_pool(s2_path, "Source 2")
s3_pool = build_country_pool(s3_path, "Source 3")


# ============================================================
# STEP 4: CREATE NON-MATCHING PAIRS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Creating negative pairs")
print("=" * 70)

negative_pairs = []

for _, s1 in s1_sample.iterrows():

    s1_id = s1["entity_id"]
    country = str(s1["country"])

    true_ids = true_matches.get(s1_id, set())

    # Try Source 2
    pool = s2_pool.get(country, [])

    attempts = 0

    while len(negative_pairs) < SAMPLE_S1 * NEGATIVES_PER_S1 and attempts < 20:
        attempts += 1

        if not pool:
            break

        candidate = random.choice(pool)

        if candidate["entity_id"] not in true_ids:
            negative_pairs.append({
                "source1_entity_id": s1_id,
                "source2_entity_id": candidate["entity_id"],
                "source": "S2",
                "s1_name": s1["business_name"],
                "s2_name": candidate["business_name"],
                "s1_address": s1["business_address"],
                "s2_address": candidate["business_address"],
                "s1_country": s1["country"],
                "s2_country": candidate["country"],
            })

    # Try Source 3
    pool = s3_pool.get(country, [])

    attempts = 0

    while len(negative_pairs) < SAMPLE_S1 * NEGATIVES_PER_S1 * 2 and attempts < 20:
        attempts += 1

        if not pool:
            break

        candidate = random.choice(pool)

        if candidate["entity_id"] not in true_ids:
            negative_pairs.append({
                "source1_entity_id": s1_id,
                "source3_entity_id": candidate["entity_id"],
                "source": "S3",
                "s1_name": s1["business_name"],
                "s2_name": candidate["business_name"],
                "s1_address": s1["business_address"],
                "s2_address": candidate["business_address"],
                "s1_country": s1["country"],
                "s2_country": candidate["country"],
            })


negative_df = pd.DataFrame(negative_pairs)

print(f"Negative pairs created: {len(negative_df)}")


# ============================================================
# STEP 5: CALCULATE SIMILARITY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Calculating false-pair similarities")
print("=" * 70)

results = []

for _, row in negative_df.iterrows():

    s1_name = normalize_text(row["s1_name"])
    s2_name = normalize_text(row["s2_name"])

    s1_address = normalize_text(row["s1_address"])
    s2_address = normalize_text(row["s2_address"])

    results.append({
        "source": row["source"],

        "name_ratio": ratio(
            s1_name,
            s2_name
        ) / 100,

        "name_token_ratio": token_set_ratio(
            s1_name,
            s2_name
        ) / 100,

        "address_ratio": ratio(
            s1_address,
            s2_address
        ) / 100,

        "address_token_ratio": token_set_ratio(
            s1_address,
            s2_address
        ) / 100,

        "country_match": int(
            str(row["s1_country"]) == str(row["s2_country"])
        ),

        "exact_name": int(
            s1_name == s2_name and s1_name != ""
        ),

        "exact_address": int(
            s1_address == s2_address and s1_address != ""
        ),
    })


result_df = pd.DataFrame(results)


# ============================================================
# STEP 6: PRINT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FALSE / NON-MATCH SIMILARITY STATISTICS")
print("=" * 70)

print(
    result_df[
        [
            "name_ratio",
            "name_token_ratio",
            "address_ratio",
            "address_token_ratio",
            "country_match",
            "exact_name",
            "exact_address",
        ]
    ].describe().round(3)
)


# ============================================================
# STEP 7: PERCENTILES
# ============================================================

print("\n" + "=" * 70)
print("PERCENTILES")
print("=" * 70)

features = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
]

for feature in features:

    print(f"\n{feature}")

    print(
        result_df[feature]
        .quantile(
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        .round(3)
    )


# ============================================================
# STEP 8: SAVE
# ============================================================

output_path = BASE / "false_pair_analysis.csv"

result_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Saved detailed results to:")
print(output_path)