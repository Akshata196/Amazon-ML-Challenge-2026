import os
import re
import pandas as pd
from collections import Counter, defaultdict
from unidecode import unidecode


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\Projects\AmazonMLChallenge\student_resource"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

S1_SAMPLE = 5000
S2_TEST_ROWS = 100_000


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


def transliterate(text):
    return normalize(unidecode(str(text)))


def get_tokens(text):
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 2
    }


def get_translit_tokens(text):
    return {
        token
        for token in transliterate(text).split()
        if len(token) >= 2
    }


def get_numbers(text):
    return {
        token
        for token in normalize(text).split()
        if any(ch.isdigit() for ch in token)
    }


# ============================================================
# STEP 1: LOAD S1
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

print(f"S1 records: {len(s1):,}")


# ============================================================
# STEP 2: PREPARE S1 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Preparing S1 blocking features")
print("=" * 70)

name_frequency = Counter()
trans_name_frequency = Counter()
address_frequency = Counter()


for _, row in s1.iterrows():

    for token in get_tokens(row["business_name"]):
        name_frequency[token] += 1

    for token in get_translit_tokens(row["business_name"]):
        trans_name_frequency[token] += 1

    for token in get_tokens(row["business_address"]):
        address_frequency[token] += 1


# ============================================================
# BLOCK FREQUENCY LIMIT
# ============================================================

# Important:
# These limits apply only to S1 block frequency.
# We will ALSO estimate actual candidate volume.

MAX_S1_TOKEN_FREQ = 30


# ============================================================
# BUILD COMPOSITE BLOCK INDEX
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Building composite blocking index")
print("=" * 70)


# Each key maps to S1 entity IDs.
blocks = defaultdict(set)


for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = normalize(row["country"])

    name_tokens = get_tokens(row["business_name"])
    trans_name_tokens = get_translit_tokens(
        row["business_name"]
    )

    address_tokens = get_tokens(
        row["business_address"]
    )

    numbers = get_numbers(
        row["business_address"]
    )

    # --------------------------------------------------------
    # NAME + ADDRESS
    # --------------------------------------------------------

    informative_names = [
        token
        for token in name_tokens
        if name_frequency[token] <= MAX_S1_TOKEN_FREQ
    ]

    informative_addresses = [
        token
        for token in address_tokens
        if address_frequency[token] <= MAX_S1_TOKEN_FREQ
    ]

    for name_token in informative_names:

        for address_token in informative_addresses:

            blocks[
                (
                    "NAME_ADDR",
                    country,
                    name_token,
                    address_token
                )
            ].add(s1_id)

    # --------------------------------------------------------
    # TRANSLITERATED NAME + ADDRESS
    # --------------------------------------------------------

    informative_trans_names = [
        token
        for token in trans_name_tokens
        if trans_name_frequency[token] <= MAX_S1_TOKEN_FREQ
    ]

    for name_token in informative_trans_names:

        for address_token in informative_addresses:

            blocks[
                (
                    "TRANS_NAME_ADDR",
                    country,
                    name_token,
                    address_token
                )
            ].add(s1_id)

    # --------------------------------------------------------
    # ADDRESS NUMBER + ADDRESS TOKEN
    # --------------------------------------------------------

    for number in numbers:

        for address_token in informative_addresses:

            blocks[
                (
                    "NUM_ADDR",
                    country,
                    number,
                    address_token
                )
            ].add(s1_id)

    # --------------------------------------------------------
    # TWO NAME TOKENS
    # --------------------------------------------------------

    if len(informative_names) >= 2:

        for i in range(len(informative_names)):

            for j in range(i + 1, len(informative_names)):

                token1 = informative_names[i]
                token2 = informative_names[j]

                pair = tuple(sorted((token1, token2)))

                blocks[
                    (
                        "NAME_NAME",
                        country,
                        pair[0],
                        pair[1]
                    )
                ].add(s1_id)


print(f"Total block keys: {len(blocks):,}")


# ============================================================
# BLOCK SIZE ANALYSIS
# ============================================================

block_sizes = [len(v) for v in blocks.values()]

if block_sizes:

    print("\nBlock size statistics:")

    print(
        f"Smallest: {min(block_sizes)}"
    )

    print(
        f"Median:   {pd.Series(block_sizes).median():.0f}"
    )

    print(
        f"Mean:     {sum(block_sizes)/len(block_sizes):.2f}"
    )

    print(
        f"Largest:  {max(block_sizes)}"
    )


# ============================================================
# STEP 4: LOAD ONLY 100K S2
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Loading first 100K Source-2 records")
print("=" * 70)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    nrows=S2_TEST_ROWS,
    dtype=str
).fillna("")

print(f"S2 records: {len(s2):,}")


# ============================================================
# STEP 5: GENERATE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Generating V3 candidates")
print("=" * 70)

candidate_pairs = set()

candidate_by_method = Counter()


for idx, row in s2.iterrows():

    if idx > 0 and idx % 10_000 == 0:

        print(
            f"Processed {idx:,} S2 records | "
            f"candidates={len(candidate_pairs):,}"
        )

    source_id = row["entity_id"]

    country = normalize(row["country"])

    name_tokens = get_tokens(
        row["business_name"]
    )

    trans_name_tokens = get_translit_tokens(
        row["business_name"]
    )

    address_tokens = get_tokens(
        row["business_address"]
    )

    numbers = get_numbers(
        row["business_address"]
    )

    # --------------------------------------------------------
    # NAME + ADDRESS
    # --------------------------------------------------------

    for name_token in name_tokens:

        for address_token in address_tokens:

            key = (
                "NAME_ADDR",
                country,
                name_token,
                address_token
            )

            matches = blocks.get(key)

            if matches:

                for s1_id in matches:

                    candidate_pairs.add(
                        (s1_id, source_id)
                    )

                    candidate_by_method[
                        "NAME_ADDR"
                    ] += 1

    # --------------------------------------------------------
    # TRANSLITERATED NAME + ADDRESS
    # --------------------------------------------------------

    for name_token in trans_name_tokens:

        for address_token in address_tokens:

            key = (
                "TRANS_NAME_ADDR",
                country,
                name_token,
                address_token
            )

            matches = blocks.get(key)

            if matches:

                for s1_id in matches:

                    candidate_pairs.add(
                        (s1_id, source_id)
                    )

                    candidate_by_method[
                        "TRANS_NAME_ADDR"
                    ] += 1

    # --------------------------------------------------------
    # NUMBER + ADDRESS
    # --------------------------------------------------------

    for number in numbers:

        for address_token in address_tokens:

            key = (
                "NUM_ADDR",
                country,
                number,
                address_token
            )

            matches = blocks.get(key)

            if matches:

                for s1_id in matches:

                    candidate_pairs.add(
                        (s1_id, source_id)
                    )

                    candidate_by_method[
                        "NUM_ADDR"
                    ] += 1

    # --------------------------------------------------------
    # TWO NAME TOKENS
    # --------------------------------------------------------

    informative_name_tokens = list(name_tokens)

    if len(informative_name_tokens) >= 2:

        for i in range(len(informative_name_tokens)):

            for j in range(i + 1, len(informative_name_tokens)):

                pair = tuple(
                    sorted(
                        (
                            informative_name_tokens[i],
                            informative_name_tokens[j]
                        )
                    )
                )

                key = (
                    "NAME_NAME",
                    country,
                    pair[0],
                    pair[1]
                )

                matches = blocks.get(key)

                if matches:

                    for s1_id in matches:

                        candidate_pairs.add(
                            (s1_id, source_id)
                        )

                        candidate_by_method[
                            "NAME_NAME"
                        ] += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V3 BLOCKING RESULTS")
print("=" * 70)

print(
    f"\nS1 records              : {len(s1):,}"
)

print(
    f"S2 records tested       : {len(s2):,}"
)

print(
    f"Candidate pairs         : {len(candidate_pairs):,}"
)

print(
    f"Average candidates/S1   : "
    f"{len(candidate_pairs) / len(s1):,.2f}"
)

print("\nCandidate generation by block method:")

for method, count in candidate_by_method.items():

    print(
        f"{method:20s}: {count:,}"
    )


# ============================================================
# STEP 6: CHECK TRUE MATCH RECALL
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: Checking true-match recall")
print("=" * 70)


# Ground truth for the first 5K S1
gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(
        set(s1["entity_id"])
    )
]


# Only S2 IDs that are actually inside our 100K sample
s2_test_ids = set(s2["entity_id"])

true_pairs_in_sample = set()


for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched = str(
        row["matched_entity_ids"]
    ).strip()

    if not matched:
        continue

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id in s2_test_ids:

            true_pairs_in_sample.add(
                (s1_id, entity_id)
            )


recovered = (
    true_pairs_in_sample
    & candidate_pairs
)

total_true = len(true_pairs_in_sample)

total_recovered = len(recovered)


print(
    f"\nTrue S1-S2 pairs inside 100K S2:"
    f" {total_true:,}"
)

print(
    f"Recovered by V3:"
    f" {total_recovered:,}"
)

if total_true > 0:

    print(
        f"Blocking recall:"
        f" {total_recovered / total_true:.2%}"
    )


print("\nDONE.")