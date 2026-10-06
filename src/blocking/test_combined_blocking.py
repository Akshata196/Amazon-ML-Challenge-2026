import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter
from itertools import combinations
from unidecode import unidecode


# ============================================================
# CONFIG
# ============================================================

BASE = r"D:\Projects\AmazonMLChallenge\student_resource\dataset"

S1_FILE = BASE + r"\train\train_source1.tsv"
S2_FILE = BASE + r"\train\train_source2.tsv"
GT_FILE = BASE + r"\train\train_ground_truth.tsv"

S1_LIMIT = 5000
S2_LIMIT = 100000

MAX_S1_TOKEN_FREQ = 30


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()

    result = []

    for ch in text:

        cat = unicodedata.category(ch)

        if cat[0] in ("L", "M", "N"):
            result.append(ch)

        elif ch.isspace():
            result.append(" ")

        else:
            result.append(" ")

    return " ".join("".join(result).split())


def transliterate(text):
    return normalize(unidecode(text))


def tokens(text):
    return set(text.split())


def numbers(text):
    return set(re.findall(r"\d+", text))


# ============================================================
# LOAD
# ============================================================

print("Loading data...")

s1_df = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_LIMIT
)

s2_df = pd.read_csv(
    S2_FILE,
    sep="\t",
    nrows=S2_LIMIT
)

gt_df = pd.read_csv(
    GT_FILE,
    sep="\t"
)

print(f"S1 sample       : {len(s1_df):,}")
print(f"S2 sample       : {len(s2_df):,}")
print(f"Ground truth    : {len(gt_df):,}")


# ============================================================
# PREPARE
# ============================================================

def prepare(df):

    records = []

    for _, r in df.iterrows():

        name = normalize(r["business_name"])
        address = normalize(r["business_address"])
        trans_name = transliterate(name)

        records.append({

            "id": str(r["entity_id"]).strip(),

            "country": normalize(r["country"]),

            "name": name,
            "address": address,

            "name_tokens": tokens(name),
            "address_tokens": tokens(address),

            "numbers": numbers(address),

            "trans_name": trans_name,
            "trans_name_tokens": tokens(trans_name),
        })

    return records


print("Preparing records...")

S1 = prepare(s1_df)
S2 = prepare(s2_df)


# ============================================================
# ID MAPS
# ============================================================

s2_ids = {
    r["id"]
    for r in S2
}


# ============================================================
# GROUND TRUTH
# ============================================================

gt_map = {}

for _, row in gt_df.iterrows():

    s1_id = str(
        row["source1_entity_id"]
    ).strip()

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":

        gt_map[s1_id] = set()

    else:

        gt_map[s1_id] = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }


# ============================================================
# TRUE PAIRS IN S2 SAMPLE
# ============================================================

true_pairs = defaultdict(set)

for r in S1:

    s1_id = r["id"]

    for entity_id in gt_map.get(
        s1_id,
        set()
    ):

        if entity_id in s2_ids:

            true_pairs[s1_id].add(
                entity_id
            )


total_true_pairs = sum(
    len(v)
    for v in true_pairs.values()
)


print(
    "\nTrue S1-S2 pairs inside sample:",
    total_true_pairs
)


# ============================================================
# S1 TOKEN FREQUENCY
# ============================================================

name_freq = Counter()
addr_freq = Counter()

for r in S1:

    country = r["country"]

    for t in r["name_tokens"]:

        name_freq[
            (country, t)
        ] += 1

    for t in r["address_tokens"]:

        addr_freq[
            (country, t)
        ] += 1


# ============================================================
# INFORMATIVE TOKENS
# ============================================================

def informative_name_tokens(r):

    vals = [

        t

        for t in r["name_tokens"]

        if name_freq[
            (r["country"], t)
        ] <= MAX_S1_TOKEN_FREQ

    ]

    return sorted(
        vals,
        key=lambda x:
        name_freq[
            (r["country"], x)
        ]
    )


def informative_address_tokens(r):

    vals = [

        t

        for t in r["address_tokens"]

        if addr_freq[
            (r["country"], t)
        ] <= MAX_S1_TOKEN_FREQ

    ]

    return sorted(
        vals,
        key=lambda x:
        addr_freq[
            (r["country"], x)
        ]
    )


# ============================================================
# BLOCK INDICES
# ============================================================

indices = {

    "NAME_ADDR":
        defaultdict(list),

    "TRANS_NAME_ADDR":
        defaultdict(list),

    "NUM_ADDR":
        defaultdict(list),

    "NAME_NAME":
        defaultdict(list),
}


print("\nBuilding block indices...")


for idx, r in enumerate(S2):

    country = r["country"]

    name_tokens = informative_name_tokens(r)
    addr_tokens = informative_address_tokens(r)

    # --------------------------------------------------------
    # 1. NAME + ADDRESS
    # --------------------------------------------------------

    for name_token in name_tokens:

        for addr_token in addr_tokens:

            key = (
                country,
                name_token,
                addr_token
            )

            indices[
                "NAME_ADDR"
            ][key].append(idx)

    # --------------------------------------------------------
    # 2. TRANSLITERATED NAME + ADDRESS
    # --------------------------------------------------------

    trans_tokens = [
        t
        for t in r["trans_name_tokens"]
        if name_freq[
            (country, t)
        ] <= MAX_S1_TOKEN_FREQ
    ]

    for name_token in trans_tokens:

        for addr_token in addr_tokens:

            key = (
                country,
                name_token,
                addr_token
            )

            indices[
                "TRANS_NAME_ADDR"
            ][key].append(idx)

    # --------------------------------------------------------
    # 3. NUMBER + ADDRESS
    # --------------------------------------------------------

    for num in r["numbers"]:

        for addr_token in addr_tokens:

            key = (
                country,
                num,
                addr_token
            )

            indices[
                "NUM_ADDR"
            ][key].append(idx)

    # --------------------------------------------------------
    # 4. TWO NAME TOKENS
    # --------------------------------------------------------

    for name1, name2 in combinations(
        name_tokens,
        2
    ):

        key = (
            country,
            name1,
            name2
        )

        indices[
            "NAME_NAME"
        ][key].append(idx)


# ============================================================
# GENERATE CANDIDATES FOR ONE METHOD
# ============================================================

def generate_method_candidates(
    method
):

    index = indices[method]

    candidates = set()

    for r in S1:

        country = r["country"]

        name_tokens = informative_name_tokens(r)
        addr_tokens = informative_address_tokens(r)

        if method == "NAME_ADDR":

            for nt in name_tokens:

                for at in addr_tokens:

                    key = (
                        country,
                        nt,
                        at
                    )

                    for idx in index.get(
                        key,
                        []
                    ):

                        candidates.add(
                            (
                                r["id"],
                                idx
                            )
                        )

        elif method == "TRANS_NAME_ADDR":

            trans_tokens = [
                t
                for t in r["trans_name_tokens"]
                if name_freq[
                    (country, t)
                ] <= MAX_S1_TOKEN_FREQ
            ]

            for nt in trans_tokens:

                for at in addr_tokens:

                    key = (
                        country,
                        nt,
                        at
                    )

                    for idx in index.get(
                        key,
                        []
                    ):

                        candidates.add(
                            (
                                r["id"],
                                idx
                            )
                        )

        elif method == "NUM_ADDR":

            for num in r["numbers"]:

                for at in addr_tokens:

                    key = (
                        country,
                        num,
                        at
                    )

                    for idx in index.get(
                        key,
                        []
                    ):

                        candidates.add(
                            (
                                r["id"],
                                idx
                            )
                        )

        elif method == "NAME_NAME":

            for name1, name2 in combinations(
                name_tokens,
                2
            ):

                key = (
                    country,
                    name1,
                    name2
                )

                for idx in index.get(
                    key,
                    []
                ):

                    candidates.add(
                        (
                            r["id"],
                            idx
                        )
                    )

    return candidates


# ============================================================
# EVALUATE
# ============================================================

def evaluate(candidates):

    candidate_lookup = defaultdict(set)

    for s1_id, s2_idx in candidates:

        candidate_lookup[
            s1_id
        ].add(
            S2[s2_idx]["id"]
        )

    recovered = 0

    for s1_id, true_ids in true_pairs.items():

        recovered += len(
            true_ids &
            candidate_lookup.get(
                s1_id,
                set()
            )
        )

    recall = (
        recovered / total_true_pairs
        if total_true_pairs
        else 0
    )

    return (
        len(candidates),
        recovered,
        recall
    )


# ============================================================
# RUN INDIVIDUAL METHODS
# ============================================================

methods = [

    "NAME_ADDR",
    "TRANS_NAME_ADDR",
    "NUM_ADDR",
    "NAME_NAME",
]


method_candidates = {}

print("\n" + "=" * 75)
print("INDIVIDUAL BLOCKING RESULTS")
print("=" * 75)


for method in methods:

    print(
        f"\nRunning {method}..."
    )

    candidates = generate_method_candidates(
        method
    )

    method_candidates[method] = candidates

    count, recovered, recall = evaluate(
        candidates
    )

    print(
        f"Candidates : {count:,}"
    )

    print(
        f"Recovered  : {recovered:,}"
    )

    print(
        f"Recall     : {recall:.2%}"
    )


# ============================================================
# COMBINATIONS
# ============================================================

combinations_to_test = [

    (
        "NAME_ADDR + NUM_ADDR",
        [
            "NAME_ADDR",
            "NUM_ADDR"
        ]
    ),

    (
        "NAME_ADDR + TRANS_NAME_ADDR + NUM_ADDR",
        [
            "NAME_ADDR",
            "TRANS_NAME_ADDR",
            "NUM_ADDR"
        ]
    ),

    (
        "ALL FOUR",
        [
            "NAME_ADDR",
            "TRANS_NAME_ADDR",
            "NUM_ADDR",
            "NAME_NAME"
        ]
    ),
]


print("\n" + "=" * 75)
print("COMBINED BLOCKING RESULTS")
print("=" * 75)


for label, methods_to_union in combinations_to_test:

    combined = set()

    for method in methods_to_union:

        combined.update(
            method_candidates[method]
        )

    count, recovered, recall = evaluate(
        combined
    )

    print(
        f"\n{label}"
    )

    print(
        f"Candidates : {count:,}"
    )

    print(
        f"Recovered  : {recovered:,}"
    )

    print(
        f"Recall     : {recall:.2%}"
    )


print("\n" + "=" * 75)
print("DONE")
print("=" * 75)