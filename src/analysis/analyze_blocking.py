from pathlib import Path
import pandas as pd
import re
import unicodedata

from normalize import normalize_text


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE / "dataset" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

SAMPLE_S1 = 2000
CHUNK_SIZE = 200_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tokens(text):
    text = normalize_text(text)

    if not text:
        return []

    return text.split()


def first_token(text):
    tokens = get_tokens(text)
    return tokens[0] if tokens else ""


def first_n_chars(text, n=3):
    text = normalize_text(text)
    text = text.replace(" ", "")

    return text[:n]


def first_n_token_chars(text, n=3):
    tokens = get_tokens(text)

    if not tokens:
        return ""

    return tokens[0][:n]


def token_set(text):
    return set(get_tokens(text))


def make_block_keys(row):
    name = normalize_text(row["business_name"])
    address = normalize_text(row["business_address"])

    name_tokens = get_tokens(name)
    address_tokens = get_tokens(address)

    name_first = name_tokens[0] if name_tokens else ""
    address_first = address_tokens[0] if address_tokens else ""

    return {
        "country": str(row["country"]),

        # Name based
        "country_name_first":
            (str(row["country"]), name_first),

        "country_name_prefix3":
            (
                str(row["country"]),
                first_n_chars(name, 3)
            ),

        "country_name_prefix4":
            (
                str(row["country"]),
                first_n_chars(name, 4)
            ),

        "country_name_token_prefix3":
            (
                str(row["country"]),
                first_n_token_chars(name, 3)
            ),

        # Address based
        "country_address_first":
            (
                str(row["country"]),
                address_first
            ),

        "country_address_prefix3":
            (
                str(row["country"]),
                first_n_chars(address, 3)
            ),
    }


# ============================================================
# STEP 1: SAMPLE SOURCE 1
# ============================================================

print("=" * 70)
print("STEP 1: Sampling Source 1")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_S1
)

print(f"Source 1 sample: {len(s1)}")


# ============================================================
# STEP 2: LOAD GROUND TRUTH FOR SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Loading ground truth")
print("=" * 70)

s1_ids = set(s1["entity_id"])

ground_truth = {}

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    chunk = chunk[
        chunk["source1_entity_id"].isin(s1_ids)
    ]

    for _, row in chunk.iterrows():

        matched = str(row["matched_entity_ids"])

        if matched == "nan" or not matched.strip():
            ground_truth[row["source1_entity_id"]] = set()
        else:
            ground_truth[row["source1_entity_id"]] = set(
                x.strip()
                for x in matched.split(",")
                if x.strip()
            )

print(f"Ground truth rows: {len(ground_truth)}")


# ============================================================
# STEP 3: CREATE SOURCE 1 BLOCK KEYS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Creating Source-1 blocking keys")
print("=" * 70)

s1_keys = {}

for _, row in s1.iterrows():

    s1_keys[row["entity_id"]] = make_block_keys(row)

print("Blocking keys created.")


# ============================================================
# STEP 4: BUILD SOURCE 2 / SOURCE 3 BLOCK INDEXES
# ============================================================

def build_indexes(file_path, source_name):

    print(f"\nBuilding indexes for {source_name}")

    indexes = {
        "country": {},
        "country_name_first": {},
        "country_name_prefix3": {},
        "country_name_prefix4": {},
        "country_name_token_prefix3": {},
        "country_address_first": {},
        "country_address_prefix3": {},
    }

    total = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        total += len(chunk)

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            keys = make_block_keys(row)

            for block_name, block_key in keys.items():

                indexes[block_name].setdefault(
                    block_key,
                    set()
                )

                indexes[block_name][block_key].add(
                    entity_id
                )

    print(f"Records indexed: {total}")

    for name, index in indexes.items():

        sizes = [
            len(ids)
            for ids in index.values()
        ]

        if sizes:
            print(
                f"{name:30s} "
                f"blocks={len(sizes):8d} "
                f"avg={sum(sizes)/len(sizes):8.2f} "
                f"max={max(sizes):8d}"
            )

    return indexes


s2_indexes = build_indexes(
    S2_PATH,
    "Source 2"
)

s3_indexes = build_indexes(
    S3_PATH,
    "Source 3"
)


# ============================================================
# STEP 5: TEST TRUE-MATCH RECALL
# ============================================================

BLOCK_NAMES = [
    "country_name_first",
    "country_name_prefix3",
    "country_name_prefix4",
    "country_name_token_prefix3",
    "country_address_first",
    "country_address_prefix3",
]


def evaluate_blocking(indexes, source_name):

    print("\n" + "=" * 70)
    print(f"BLOCKING RECALL — {source_name}")
    print("=" * 70)

    for block_name in BLOCK_NAMES:

        total_true = 0
        recovered = 0

        for s1_id, true_ids in ground_truth.items():

            if not true_ids:
                continue

            keys = s1_keys[s1_id]

            block_key = keys[block_name]

            candidates = indexes[block_name].get(
                block_key,
                set()
            )

            for true_id in true_ids:

                # Only evaluate IDs belonging to this source
                if source_name == "S2" and not true_id.startswith("S2-"):
                  continue

                if source_name == "S3" and not true_id.startswith("S3-"):
                  continue

                total_true += 1

                if true_id in candidates:
                    recovered += 1

        recall = (
            recovered / total_true
            if total_true > 0
            else 0
        )

        print(
            f"{block_name:30s} "
            f"recall={recall:.4f} "
            f"({recovered}/{total_true})"
        )


evaluate_blocking(
    s2_indexes,
    "S2"
)

evaluate_blocking(
    s3_indexes,
    "S3"
)


print("\n" + "=" * 70)
print("BLOCKING ANALYSIS COMPLETE")
print("=" * 70)