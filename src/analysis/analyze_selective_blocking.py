import os
import re
import pandas as pd
from collections import Counter, defaultdict

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
CHUNK_SIZE = 200_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Keep Unicode letters/numbers.
    text = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in text
    )

    return " ".join(text.split())


def tokens(text):
    return normalize(text).split()


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("=" * 70)
print("STEP 1: Loading Source-1 sample")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_SAMPLE,
    dtype=str
).fillna("")

print(f"S1 records: {len(s1)}")


# ============================================================
# BUILD TOKEN FREQUENCY FROM S1
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Calculating S1 token frequencies")
print("=" * 70)

name_counter = Counter()
address_counter = Counter()

for _, row in s1.iterrows():

    for token in tokens(row["business_name"]):
        if len(token) >= 3:
            name_counter[token] += 1

    for token in tokens(row["business_address"]):
        if len(token) >= 3:
            address_counter[token] += 1


# ============================================================
# SHOW COMMON TOKENS
# ============================================================

print("\nMost common NAME tokens:")

for token, count in name_counter.most_common(30):
    print(f"{token:30s} {count}")

print("\nMost common ADDRESS tokens:")

for token, count in address_counter.most_common(30):
    print(f"{token:30s} {count}")


# ============================================================
# BUILD SELECTIVE TOKENS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Selecting informative tokens")
print("=" * 70)

# We don't want extremely common tokens.
# Start conservatively for the experiment.

MAX_NAME_FREQ = 20
MAX_ADDRESS_FREQ = 20

s1_name_blocks = defaultdict(list)
s1_address_blocks = defaultdict(list)

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = normalize(row["country"])

    name_toks = set(tokens(row["business_name"]))
    address_toks = set(tokens(row["business_address"]))

    # -----------------------------
    # NAME BLOCKS
    # -----------------------------

    rare_name_tokens = [
        t for t in name_toks
        if len(t) >= 3
        and name_counter[t] <= MAX_NAME_FREQ
    ]

    for token in rare_name_tokens:
        key = (country, token)
        s1_name_blocks[key].append(s1_id)

    # -----------------------------
    # ADDRESS BLOCKS
    # -----------------------------

    rare_address_tokens = [
        t for t in address_toks
        if len(t) >= 3
        and address_counter[t] <= MAX_ADDRESS_FREQ
    ]

    for token in rare_address_tokens:
        key = (country, token)
        s1_address_blocks[key].append(s1_id)


print(f"Unique name block keys: {len(s1_name_blocks):,}")
print(f"Unique address block keys: {len(s1_address_blocks):,}")


# ============================================================
# SOURCE SCANNER
# ============================================================

def generate_candidates(source_file, source_name):

    print("\n" + "=" * 70)
    print(f"STEP 4: Testing selective blocking on {source_name}")
    print("=" * 70)

    total_records = 0
    candidate_pairs = set()

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            source_file,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        chunk = chunk.fillna("")

        for _, row in chunk.iterrows():

            total_records += 1

            source_id = row["entity_id"]
            country = normalize(row["country"])

            name_toks = set(tokens(row["business_name"]))
            address_toks = set(tokens(row["business_address"]))

            # -----------------------------
            # NAME BLOCK
            # -----------------------------

            for token in name_toks:

                if len(token) < 3:
                    continue

                if name_counter[token] > MAX_NAME_FREQ:
                    continue

                key = (country, token)

                for s1_id in s1_name_blocks.get(key, []):
                    candidate_pairs.add((s1_id, source_id))

            # -----------------------------
            # ADDRESS BLOCK
            # -----------------------------

            for token in address_toks:

                if len(token) < 3:
                    continue

                if address_counter[token] > MAX_ADDRESS_FREQ:
                    continue

                key = (country, token)

                for s1_id in s1_address_blocks.get(key, []):
                    candidate_pairs.add((s1_id, source_id))

        print(
            f"Chunk {chunk_no}: "
            f"{total_records:,} records | "
            f"candidates={len(candidate_pairs):,}"
        )

    print("\nRESULT")
    print("-" * 50)
    print(f"{source_name} records scanned : {total_records:,}")
    print(f"Candidate pairs              : {len(candidate_pairs):,}")
    print(
        f"Average candidates / S1      : "
        f"{len(candidate_pairs) / S1_SAMPLE:,.2f}"
    )

    return candidate_pairs


# ============================================================
# RUN
# ============================================================

s2_candidates = generate_candidates(
    S2_FILE,
    "Source 2"
)

s3_candidates = generate_candidates(
    S3_FILE,
    "Source 3"
)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total = len(s2_candidates) + len(s3_candidates)

print(f"S2 candidates : {len(s2_candidates):,}")
print(f"S3 candidates : {len(s3_candidates):,}")
print(f"TOTAL         : {total:,}")

