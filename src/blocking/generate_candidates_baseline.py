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

OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_S1 = 5000
CHUNK_SIZE = 200_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
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

    return re.sub(r"\s+", " ", text).strip()


def get_tokens(text):

    text = normalize_text(text)

    return text.split() if text else []


# ============================================================
# BLOCKING SIGNATURES
# ============================================================

def make_signatures(row):

    country = str(row["country"])

    name = normalize_text(row["business_name"])
    address = normalize_text(row["business_address"])

    name_tokens = get_tokens(name)
    address_tokens = get_tokens(address)

    signatures = set()

    # --------------------------------------------------------
    # Block 1: exact normalized name
    # --------------------------------------------------------

    if name:

        signatures.add(
            (
                "NAME_EXACT",
                country,
                name
            )
        )

    # --------------------------------------------------------
    # Block 2: first + last name token
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        signatures.add(
            (
                "NAME_FIRST_LAST",
                country,
                name_tokens[0][:3],
                name_tokens[-1][:3]
            )
        )

    elif len(name_tokens) == 1:

        signatures.add(
            (
                "NAME_FIRST_LAST",
                country,
                name_tokens[0][:3],
                ""
            )
        )

    # --------------------------------------------------------
    # Block 3: first + last 3 characters of complete name
    # --------------------------------------------------------

    compact_name = name.replace(" ", "")

    if len(compact_name) >= 3:

        signatures.add(
            (
                "NAME_EDGE",
                country,
                compact_name[:3],
                compact_name[-3:]
            )
        )

    # --------------------------------------------------------
    # Block 4: address number
    # --------------------------------------------------------

    address_numbers = re.findall(
        r"\b\d+\b",
        address
    )

    if address_numbers:

        # Use first number only
        signatures.add(
            (
                "ADDRESS_NUMBER",
                country,
                address_numbers[0]
            )
        )

    # --------------------------------------------------------
    # Block 5: first + last address token
    # --------------------------------------------------------

    if len(address_tokens) >= 2:

        signatures.add(
            (
                "ADDRESS_EDGE",
                country,
                address_tokens[0][:3],
                address_tokens[-1][:3]
            )
        )

    return signatures


# ============================================================
# STEP 1: LOAD S1 SAMPLE
# ============================================================

print("=" * 70)
print("STEP 1: Loading Source-1 sample")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_S1
)

print(f"S1 records: {len(s1)}")


# ============================================================
# STEP 2: CREATE S1 SIGNATURE INDEX
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Building S1 signature index")
print("=" * 70)

signature_to_s1 = {}

for _, row in s1.iterrows():

    entity_id = row["entity_id"]

    signatures = make_signatures(row)

    for signature in signatures:

        signature_to_s1.setdefault(
            signature,
            set()
        ).add(entity_id)

print(
    f"Unique blocking signatures: "
    f"{len(signature_to_s1):,}"
)


# ============================================================
# STEP 3: SCAN SOURCE
# ============================================================

def generate_for_source(
    source_path,
    source_name
):

    print("\n" + "=" * 70)
    print(f"STEP 3: Generating candidates from {source_name}")
    print("=" * 70)

    candidate_pairs = set()

    total = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        for _, row in chunk.iterrows():

            source_id = row["entity_id"]

            signatures = make_signatures(row)

            matched_s1 = set()

            for signature in signatures:

                s1_ids = signature_to_s1.get(
                    signature,
                    set()
                )

                matched_s1.update(s1_ids)

            for s1_id in matched_s1:

                candidate_pairs.add(
                    (
                        s1_id,
                        source_id
                    )
                )

        total += len(chunk)

        print(
            f"Processed chunk {chunk_number} "
            f"({total:,} records) | "
            f"candidates={len(candidate_pairs):,}"
        )

    print(
        f"\n{source_name} records scanned: "
        f"{total:,}"
    )

    print(
        f"{source_name} candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    return candidate_pairs


# ============================================================
# STEP 4: SOURCE 2
# ============================================================

s2_candidates = generate_for_source(
    S2_PATH,
    "Source 2"
)


# ============================================================
# STEP 5: SOURCE 3
# ============================================================

s3_candidates = generate_for_source(
    S3_PATH,
    "Source 3"
)


# ============================================================
# STEP 6: SAVE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: Saving candidates")
print("=" * 70)

rows = []

for s1_id, source_id in s2_candidates:

    rows.append({
        "source1_entity_id": s1_id,
        "candidate_entity_id": source_id
    })

for s1_id, source_id in s3_candidates:

    rows.append({
        "source1_entity_id": s1_id,
        "candidate_entity_id": source_id
    })


candidate_df = pd.DataFrame(rows)

output_path = (
    OUTPUT_DIR /
    "candidate_pairs_baseline.tsv"
)

candidate_df.to_csv(
    output_path,
    sep="\t",
    index=False
)

print(
    f"Total candidate pairs: "
    f"{len(candidate_df):,}"
)

print(f"Saved to:\n{output_path}")

print("\n" + "=" * 70)
print("BASELINE CANDIDATE GENERATION COMPLETE")
print("=" * 70)