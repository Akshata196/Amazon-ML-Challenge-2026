from pathlib import Path
import pandas as pd
import re
import unicodedata
from rapidfuzz import fuzz

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

GT_PATH = BASE / "dataset/train/train_ground_truth.tsv"
S2_PATH = BASE / "dataset/train/train_source2.tsv"
S3_PATH = BASE / "dataset/train/train_source3.tsv"

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    return set(normalize_text(text).split())


# --------------------------------------------------
# LOAD SAMPLE GROUND TRUTH
# --------------------------------------------------

print("=" * 70)
print("STEP 1: Loading ground truth sample")
print("=" * 70)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

print("Total ground truth rows:", len(gt))

# Only entities having at least one match
gt_with_matches = gt[
    gt["matched_entity_ids"].notna()
    & (gt["matched_entity_ids"].str.strip() != "")
].copy()

sample = gt_with_matches.sample(
    n=min(SAMPLE_SIZE, len(gt_with_matches)),
    random_state=RANDOM_STATE
)

print("Sampled Source-1 entities:", len(sample))


# --------------------------------------------------
# EXTRACT MATCH IDS
# --------------------------------------------------

needed_s2 = set()
needed_s3 = set()

for ids in sample["matched_entity_ids"]:
    for entity_id in ids.split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            needed_s2.add(entity_id)

        elif entity_id.startswith("S3-"):
            needed_s3.add(entity_id)

print("\nRequired true matches:")
print("S2 IDs:", len(needed_s2))
print("S3 IDs:", len(needed_s3))


# --------------------------------------------------
# FETCH ONLY REQUIRED RECORDS FROM LARGE FILES
# --------------------------------------------------

def fetch_records(path, needed_ids, chunksize=200_000):

    result = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize
    ):
        matched = chunk[
            chunk["entity_id"].isin(needed_ids)
        ]

        if len(matched):
            result.append(matched)

    if result:
        return pd.concat(result, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


print("\n" + "=" * 70)
print("STEP 2: Fetching true matching records")
print("=" * 70)

s2 = fetch_records(S2_PATH, needed_s2)
s3 = fetch_records(S3_PATH, needed_s3)

print("Fetched S2 records:", len(s2))
print("Fetched S3 records:", len(s3))


# --------------------------------------------------
# CREATE LOOKUP
# --------------------------------------------------

s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")


# --------------------------------------------------
# LOAD SOURCE 1 SAMPLE
# --------------------------------------------------

s1 = pd.read_csv(
    BASE / "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
)

s1_lookup = s1.set_index("entity_id").to_dict("index")


# --------------------------------------------------
# CALCULATE SIMILARITIES
# --------------------------------------------------

results = []

print("\n" + "=" * 70)
print("STEP 3: Calculating true-pair similarities")
print("=" * 70)

for _, row in sample.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_lookup:
        continue

    source1 = s1_lookup[s1_id]

    name1 = normalize_text(source1["business_name"])
    addr1 = normalize_text(source1["business_address"])
    country1 = normalize_text(source1["country"])

    for matched_id in row["matched_entity_ids"].split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            candidate = s2_lookup.get(matched_id)
        else:
            candidate = s3_lookup.get(matched_id)

        if candidate is None:
            continue

        name2 = normalize_text(candidate["business_name"])
        addr2 = normalize_text(candidate["business_address"])
        country2 = normalize_text(candidate["country"])

        name_ratio = fuzz.ratio(name1, name2) / 100.0
        name_token_ratio = fuzz.token_set_ratio(name1, name2) / 100.0

        if addr1 and addr2:
            address_ratio = fuzz.ratio(addr1, addr2) / 100.0
            address_token_ratio = fuzz.token_set_ratio(addr1, addr2) / 100.0
        else:
            address_ratio = 0.0
            address_token_ratio = 0.0

        country_match = int(country1 == country2)

        exact_name = int(
            name1 != "" and name1 == name2
        )

        exact_address = int(
            addr1 != "" and addr1 == addr2
        )

        results.append({
            "s1_id": s1_id,
            "matched_id": matched_id,
            "name_ratio": name_ratio,
            "name_token_ratio": name_token_ratio,
            "address_ratio": address_ratio,
            "address_token_ratio": address_token_ratio,
            "country_match": country_match,
            "exact_name": exact_name,
            "exact_address": exact_address
        })


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

pairs = pd.DataFrame(results)

print("\nTrue pairs analyzed:", len(pairs))

print("\n" + "=" * 70)
print("TRUE MATCH SIMILARITY STATISTICS")
print("=" * 70)

columns = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "country_match",
    "exact_name",
    "exact_address"
]

print(
    pairs[columns].describe().round(3).to_string()
)


# --------------------------------------------------
# PERCENTILES
# --------------------------------------------------

print("\n" + "=" * 70)
print("PERCENTILES")
print("=" * 70)

for column in [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio"
]:

    print(f"\n{column}")

    print(
        pairs[column]
        .quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .round(3)
        .to_string()
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = BASE / "true_pair_analysis.csv"

pairs.to_csv(
    output_path,
    index=False
)

print("\nSaved detailed results to:")
print(output_path)