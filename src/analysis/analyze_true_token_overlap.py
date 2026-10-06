import os
import pandas as pd
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\Projects\AmazonMLChallenge\student_resource"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

S1_SAMPLE = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in text
    )

    return " ".join(text.split())


def get_tokens(text):
    return set(
        token
        for token in normalize(text).split()
        if len(token) >= 2
    )


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("STEP 1: Loading S1 sample")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_SAMPLE,
    dtype=str
).fillna("")

s1["name_tokens"] = s1["business_name"].apply(get_tokens)
s1["address_tokens"] = s1["business_address"].apply(get_tokens)

s1_lookup = s1.set_index("entity_id").to_dict("index")

print(f"S1 sample: {len(s1):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Loading ground truth")
print("=" * 70)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(s1_lookup)
].copy()

print(f"S1 entities with ground truth: {len(gt):,}")


# ============================================================
# COLLECT REQUIRED SOURCE IDs
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Collecting true-match IDs")
print("=" * 70)

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    if not value:
        continue

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)

print(f"Required S2 records: {len(s2_ids):,}")
print(f"Required S3 records: {len(s3_ids):,}")


# ============================================================
# LOAD ONLY REQUIRED S2/S3 RECORDS
# ============================================================

def load_required_records(file_path, required_ids, source_name):

    print(f"\nLoading required {source_name} records...")

    result = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=200_000
    ):

        chunk = chunk.fillna("")

        mask = chunk["entity_id"].isin(required_ids)

        selected = chunk.loc[mask]

        for _, row in selected.iterrows():
            result[row["entity_id"]] = {
                "business_name": row["business_name"],
                "business_address": row["business_address"],
                "country": row["country"],
            }

        if len(result) == len(required_ids):
            break

    print(f"Loaded {len(result):,} {source_name} records")

    return result


s2 = load_required_records(
    S2_FILE,
    s2_ids,
    "S2"
)

s3 = load_required_records(
    S3_FILE,
    s3_ids,
    "S3"
)


# ============================================================
# ANALYZE TOKEN OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Analyzing true-pair token overlap")
print("=" * 70)


stats = {
    "total_pairs": 0,

    "name_shared": 0,
    "name_shared_2plus": 0,
    "name_jaccard_high": 0,

    "address_shared": 0,
    "address_shared_2plus": 0,
    "address_jaccard_high": 0,

    "both_name_address_shared": 0,
    "name_or_address_shared": 0,

    "number_overlap": 0,
}

name_overlap_counts = Counter()
address_overlap_counts = Counter()

examples = []


def number_tokens(tokens_set):
    return {
        token
        for token in tokens_set
        if any(ch.isdigit() for ch in token)
    }


for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_lookup:
        continue

    s1_record = s1_lookup[s1_id]

    s1_name = s1_record["name_tokens"]
    s1_address = s1_record["address_tokens"]

    matched = str(row["matched_entity_ids"]).strip()

    if not matched:
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id.startswith("S2-"):
            candidate = s2.get(candidate_id)

        elif candidate_id.startswith("S3-"):
            candidate = s3.get(candidate_id)

        else:
            continue

        if candidate is None:
            continue

        stats["total_pairs"] += 1

        c_name = get_tokens(candidate["business_name"])
        c_address = get_tokens(candidate["business_address"])

        # -------------------------
        # NAME OVERLAP
        # -------------------------

        name_common = s1_name & c_name
        name_overlap = len(name_common)

        name_overlap_counts[name_overlap] += 1

        if name_overlap >= 1:
            stats["name_shared"] += 1

        if name_overlap >= 2:
            stats["name_shared_2plus"] += 1

        # -------------------------
        # ADDRESS OVERLAP
        # -------------------------

        address_common = s1_address & c_address
        address_overlap = len(address_common)

        address_overlap_counts[address_overlap] += 1

        if address_overlap >= 1:
            stats["address_shared"] += 1

        if address_overlap >= 2:
            stats["address_shared_2plus"] += 1

        # -------------------------
        # BOTH
        # -------------------------

        if name_overlap >= 1 and address_overlap >= 1:
            stats["both_name_address_shared"] += 1

        if name_overlap >= 1 or address_overlap >= 1:
            stats["name_or_address_shared"] += 1

        # -------------------------
        # NUMBERS
        # -------------------------

        s1_numbers = number_tokens(s1_address)
        c_numbers = number_tokens(c_address)

        if s1_numbers & c_numbers:
            stats["number_overlap"] += 1

        # -------------------------
        # SAVE EXAMPLES
        # -------------------------

        if len(examples) < 20:

            examples.append({
                "s1_name": s1_record["business_name"],
                "match_name": candidate["business_name"],
                "s1_address": s1_record["business_address"],
                "match_address": candidate["business_address"],
                "shared_name_tokens": sorted(name_common),
                "shared_address_tokens": sorted(address_common),
                "shared_numbers": sorted(s1_numbers & c_numbers),
            })


# ============================================================
# PRINT RESULTS
# ============================================================

total = stats["total_pairs"]

print("\n" + "=" * 70)
print("TRUE MATCH TOKEN OVERLAP")
print("=" * 70)

print(f"\nTotal true pairs analyzed: {total:,}")

if total > 0:

    print(
        f"\nName shared >= 1 token : "
        f"{stats['name_shared']:,} "
        f"({stats['name_shared']/total:.2%})"
    )

    print(
        f"Name shared >= 2 tokens: "
        f"{stats['name_shared_2plus']:,} "
        f"({stats['name_shared_2plus']/total:.2%})"
    )

    print(
        f"\nAddress shared >= 1 token : "
        f"{stats['address_shared']:,} "
        f"({stats['address_shared']/total:.2%})"
    )

    print(
        f"Address shared >= 2 tokens: "
        f"{stats['address_shared_2plus']:,} "
        f"({stats['address_shared_2plus']/total:.2%})"
    )

    print(
        f"\nBoth name + address shared: "
        f"{stats['both_name_address_shared']:,} "
        f"({stats['both_name_address_shared']/total:.2%})"
    )

    print(
        f"Name OR address shared: "
        f"{stats['name_or_address_shared']:,} "
        f"({stats['name_or_address_shared']/total:.2%})"
    )

    print(
        f"Address number overlap: "
        f"{stats['number_overlap']:,} "
        f"({stats['number_overlap']/total:.2%})"
    )


print("\n" + "=" * 70)
print("NAME TOKEN OVERLAP DISTRIBUTION")
print("=" * 70)

for count, frequency in sorted(name_overlap_counts.items()):
    print(f"{count:2d} shared tokens : {frequency:,}")


print("\n" + "=" * 70)
print("ADDRESS TOKEN OVERLAP DISTRIBUTION")
print("=" * 70)

for count, frequency in sorted(address_overlap_counts.items()):
    print(f"{count:2d} shared tokens : {frequency:,}")


print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)

for i, example in enumerate(examples, 1):

    print(f"\nExample {i}")

    print("S1 name     :", example["s1_name"])
    print("Match name  :", example["match_name"])

    print("S1 address  :", example["s1_address"])
    print("Match addr  :", example["match_address"])

    print("Shared name :", example["shared_name_tokens"])
    print("Shared addr :", example["shared_address_tokens"])
    print("Shared nums :", example["shared_numbers"])


print("\nDONE.")