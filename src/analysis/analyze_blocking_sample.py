from pathlib import Path
import pandas as pd
import re
import unicodedata


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE / "dataset" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

SAMPLE_S1 = 5000
CHUNK_SIZE = 200_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    cleaned = []

    for char in text:

        category = unicodedata.category(char)

        if (
            category.startswith("L")
            or category.startswith("M")
            or category.startswith("N")
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokens(text):

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


# ============================================================
# BLOCKING KEY FUNCTIONS
# ============================================================

def get_keys(row):

    name = normalize_text(row["business_name"])
    address = normalize_text(row["business_address"])

    name_tokens = tokens(name)
    address_tokens = tokens(address)

    keys = {}

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    country = str(row["country"])

    keys["country"] = country

    # --------------------------------------------------------
    # Name first token
    # --------------------------------------------------------

    if name_tokens:

        keys["country_name_first"] = (
            country,
            name_tokens[0]
        )

    else:

        keys["country_name_first"] = (
            country,
            ""
        )

    # --------------------------------------------------------
    # First 3 characters of complete normalized name
    # --------------------------------------------------------

    compact_name = name.replace(" ", "")

    keys["country_name_prefix3"] = (
        country,
        compact_name[:3]
    )

    # --------------------------------------------------------
    # First 4 characters
    # --------------------------------------------------------

    keys["country_name_prefix4"] = (
        country,
        compact_name[:4]
    )

    # --------------------------------------------------------
    # First 3 characters of first token
    # --------------------------------------------------------

    if name_tokens:

        keys["country_name_token_prefix3"] = (
            country,
            name_tokens[0][:3]
        )

    else:

        keys["country_name_token_prefix3"] = (
            country,
            ""
        )

    # --------------------------------------------------------
    # Address first token
    # --------------------------------------------------------

    if address_tokens:

        keys["country_address_first"] = (
            country,
            address_tokens[0]
        )

    else:

        keys["country_address_first"] = (
            country,
            ""
        )

    # --------------------------------------------------------
    # Address prefix
    # --------------------------------------------------------

    compact_address = address.replace(" ", "")

    keys["country_address_prefix3"] = (
        country,
        compact_address[:3]
    )

    return keys


# ============================================================
# STEP 1
# SAMPLE SOURCE 1
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

print(f"Source-1 sample: {len(s1)}")


# ============================================================
# STEP 2
# GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Loading ground truth")
print("=" * 70)

sample_ids = set(s1["entity_id"])

ground_truth = {}

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    filtered = chunk[
        chunk["source1_entity_id"].isin(sample_ids)
    ]

    for _, row in filtered.iterrows():

        matched = str(row["matched_entity_ids"])

        if matched == "nan" or not matched.strip():

            ground_truth[
                row["source1_entity_id"]
            ] = []

        else:

            ground_truth[
                row["source1_entity_id"]
            ] = [
                x.strip()
                for x in matched.split(",")
                if x.strip()
            ]


print(
    f"Ground-truth Source-1 entities: "
    f"{len(ground_truth)}"
)


# ============================================================
# STEP 3
# COLLECT REQUIRED TRUE MATCH IDs
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Collecting true match IDs")
print("=" * 70)

required_s2 = set()
required_s3 = set()

for matches in ground_truth.values():

    for entity_id in matches:

        if entity_id.startswith("S2-"):

            required_s2.add(entity_id)

        elif entity_id.startswith("S3-"):

            required_s3.add(entity_id)


print(f"Required S2 records: {len(required_s2)}")
print(f"Required S3 records: {len(required_s3)}")


# ============================================================
# STEP 4
# FETCH ONLY TRUE MATCH RECORDS
# ============================================================

def fetch_records(file_path, required_ids, source_name):

    print(f"\nFetching {source_name} true records...")

    records = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if len(matched) > 0:

            records.extend(
                matched.to_dict("records")
            )

    print(
        f"Fetched {source_name}: "
        f"{len(records)}"
    )

    return {
        row["entity_id"]: row
        for row in records
    }


s2_records = fetch_records(
    S2_PATH,
    required_s2,
    "S2"
)

s3_records = fetch_records(
    S3_PATH,
    required_s3,
    "S3"
)


# ============================================================
# STEP 5
# CREATE S1 KEY CACHE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Creating blocking keys")
print("=" * 70)

s1_dict = {}

for _, row in s1.iterrows():

    s1_dict[
        row["entity_id"]
    ] = row


# ============================================================
# STEP 6
# TEST BLOCKING KEYS
# ============================================================

BLOCK_NAMES = [
    "country",
    "country_name_first",
    "country_name_prefix3",
    "country_name_prefix4",
    "country_name_token_prefix3",
    "country_address_first",
    "country_address_prefix3",
]


results = {
    name: {
        "total": 0,
        "recovered": 0
    }
    for name in BLOCK_NAMES
}


for s1_id, matches in ground_truth.items():

    if not matches:
        continue

    s1_row = s1_dict.get(s1_id)

    if s1_row is None:
        continue

    s1_keys = get_keys(s1_row)

    for true_id in matches:

        if true_id.startswith("S2-"):

            candidate = s2_records.get(true_id)

        elif true_id.startswith("S3-"):

            candidate = s3_records.get(true_id)

        else:

            continue

        if candidate is None:
            continue

        candidate_keys = get_keys(candidate)

        for block_name in BLOCK_NAMES:

            results[block_name]["total"] += 1

            if (
                s1_keys[block_name]
                ==
                candidate_keys[block_name]
            ):

                results[
                    block_name
                ]["recovered"] += 1


# ============================================================
# STEP 7
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRUE-MATCH BLOCKING RECALL")
print("=" * 70)

for block_name in BLOCK_NAMES:

    total = results[block_name]["total"]

    recovered = results[block_name]["recovered"]

    recall = (
        recovered / total
        if total
        else 0
    )

    print(
        f"{block_name:35s} "
        f"recall={recall:.4f} "
        f"({recovered}/{total})"
    )


print("\n" + "=" * 70)
print("BLOCKING SAMPLE ANALYSIS COMPLETE")
print("=" * 70)