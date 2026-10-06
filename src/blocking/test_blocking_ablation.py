import os
import pandas as pd
from collections import defaultdict

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

MAX_S1_TOKEN_FREQ = 30


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


def tokens(text):
    return {
        x for x in normalize(text).split()
        if len(x) >= 2
    }


def translit_tokens(text):
    return {
        x for x in transliterate(text).split()
        if len(x) >= 2
    }


def numbers(text):
    return {
        x for x in normalize(text).split()
        if any(ch.isdigit() for ch in x)
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_SAMPLE,
    dtype=str
).fillna("")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    nrows=S2_TEST_ROWS,
    dtype=str
).fillna("")

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")


# ============================================================
# FREQUENCIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Building S1 token frequencies")
print("=" * 70)

name_freq = {}
trans_name_freq = {}
addr_freq = {}

for _, row in s1.iterrows():

    for t in tokens(row["business_name"]):
        name_freq[t] = name_freq.get(t, 0) + 1

    for t in translit_tokens(row["business_name"]):
        trans_name_freq[t] = trans_name_freq.get(t, 0) + 1

    for t in tokens(row["business_address"]):
        addr_freq[t] = addr_freq.get(t, 0) + 1


# ============================================================
# BUILD SEPARATE INDICES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Building separate block indices")
print("=" * 70)

indices = {
    "NAME_ADDR": defaultdict(set),
    "TRANS_NAME_ADDR": defaultdict(set),
    "NUM_ADDR": defaultdict(set),
    "NAME_NAME": defaultdict(set),
}


for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = normalize(row["country"])

    name = tokens(row["business_name"])
    trans_name = translit_tokens(row["business_name"])
    address = tokens(row["business_address"])
    nums = numbers(row["business_address"])

    informative_name = [
        x for x in name
        if name_freq.get(x, 0) <= MAX_S1_TOKEN_FREQ
    ]

    informative_trans_name = [
        x for x in trans_name
        if trans_name_freq.get(x, 0) <= MAX_S1_TOKEN_FREQ
    ]

    informative_address = [
        x for x in address
        if addr_freq.get(x, 0) <= MAX_S1_TOKEN_FREQ
    ]

    # NAME + ADDRESS
    for n in informative_name:
        for a in informative_address:

            key = (country, n, a)

            indices["NAME_ADDR"][key].add(s1_id)

    # TRANSLITERATED NAME + ADDRESS
    for n in informative_trans_name:
        for a in informative_address:

            key = (country, n, a)

            indices["TRANS_NAME_ADDR"][key].add(s1_id)

    # NUMBER + ADDRESS
    for num in nums:
        for a in informative_address:

            key = (country, num, a)

            indices["NUM_ADDR"][key].add(s1_id)

    # NAME + NAME
    if len(informative_name) >= 2:

        for i in range(len(informative_name)):

            for j in range(i + 1, len(informative_name)):

                pair = tuple(
                    sorted(
                        (
                            informative_name[i],
                            informative_name[j]
                        )
                    )
                )

                key = (country, pair[0], pair[1])

                indices["NAME_NAME"][key].add(s1_id)


for method in indices:

    print(
        f"{method:20s}: "
        f"{len(indices[method]):,} block keys"
    )


# ============================================================
# GENERATE CANDIDATES PER METHOD
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Generating candidates separately")
print("=" * 70)


candidate_sets = {
    method: set()
    for method in indices
}


for idx, row in s2.iterrows():

    if idx > 0 and idx % 20_000 == 0:

        print(
            f"Processed {idx:,}/{len(s2):,}"
        )

    source_id = row["entity_id"]
    country = normalize(row["country"])

    name = tokens(row["business_name"])
    trans_name = translit_tokens(row["business_name"])
    address = tokens(row["business_address"])
    nums = numbers(row["business_address"])

    # --------------------------------------------------------
    # NAME + ADDRESS
    # --------------------------------------------------------

    for n in name:

        for a in address:

            key = (country, n, a)

            for s1_id in indices["NAME_ADDR"].get(key, []):

                candidate_sets["NAME_ADDR"].add(
                    (s1_id, source_id)
                )

    # --------------------------------------------------------
    # TRANS NAME + ADDRESS
    # --------------------------------------------------------

    for n in trans_name:

        for a in address:

            key = (country, n, a)

            for s1_id in indices["TRANS_NAME_ADDR"].get(key, []):

                candidate_sets["TRANS_NAME_ADDR"].add(
                    (s1_id, source_id)
                )

    # --------------------------------------------------------
    # NUMBER + ADDRESS
    # --------------------------------------------------------

    for num in nums:

        for a in address:

            key = (country, num, a)

            for s1_id in indices["NUM_ADDR"].get(key, []):

                candidate_sets["NUM_ADDR"].add(
                    (s1_id, source_id)
                )

    # --------------------------------------------------------
    # NAME + NAME
    # --------------------------------------------------------

    name_list = list(name)

    if len(name_list) >= 2:

        for i in range(len(name_list)):

            for j in range(i + 1, len(name_list)):

                pair = tuple(
                    sorted(
                        (
                            name_list[i],
                            name_list[j]
                        )
                    )
                )

                key = (
                    country,
                    pair[0],
                    pair[1]
                )

                for s1_id in indices["NAME_NAME"].get(key, []):

                    candidate_sets["NAME_NAME"].add(
                        (s1_id, source_id)
                    )


# ============================================================
# GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Loading ground truth")
print("=" * 70)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

s1_ids = set(s1["entity_id"])
s2_ids = set(s2["entity_id"])

true_pairs = set()

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_ids:
        continue

    value = str(
        row["matched_entity_ids"]
    ).strip()

    if not value:
        continue

    for candidate_id in value.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id in s2_ids:

            true_pairs.add(
                (s1_id, candidate_id)
            )


print(
    f"True pairs inside 100K S2: "
    f"{len(true_pairs):,}"
)


# ============================================================
# ABLATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING ABLATION RESULTS")
print("=" * 70)


for method, candidates in candidate_sets.items():

    recovered = candidates & true_pairs

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0
    )

    print("\n" + "-" * 60)

    print(f"METHOD: {method}")

    print(
        f"Candidates : {len(candidates):,}"
    )

    print(
        f"Recovered  : {len(recovered):,}"
    )

    print(
        f"Recall     : {recall:.2%}"
    )


# ============================================================
# COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("BLOCK COMBINATIONS")
print("=" * 70)


combinations = [
    ["NAME_ADDR", "TRANS_NAME_ADDR"],
    ["NAME_ADDR", "NAME_NAME"],
    ["NAME_ADDR", "NUM_ADDR"],
    ["NAME_ADDR", "TRANS_NAME_ADDR", "NAME_NAME"],
    ["NAME_ADDR", "TRANS_NAME_ADDR", "NUM_ADDR"],
    ["NAME_ADDR", "TRANS_NAME_ADDR", "NAME_NAME", "NUM_ADDR"],
]


for methods in combinations:

    combined = set()

    for method in methods:
        combined.update(
            candidate_sets[method]
        )

    recovered = combined & true_pairs

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0
    )

    print("\n" + "+".join(methods))

    print(
        f"Candidates : {len(combined):,}"
    )

    print(
        f"Recovered  : {len(recovered):,}"
    )

    print(
        f"Recall     : {recall:.2%}"
    )


print("\nDONE.")