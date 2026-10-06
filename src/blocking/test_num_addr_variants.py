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
MAX_S1_NUM_FREQ = 50


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

# IMPORTANT:
# Read GT independently and create an ID-based lookup.
gt_df = pd.read_csv(
    GT_FILE,
    sep="\t"
)

print(f"S1 sample: {len(s1_df):,}")
print(f"S2 sample: {len(s2_df):,}")
print(f"Ground truth rows: {len(gt_df):,}")


# ============================================================
# PREPARE RECORDS
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

s1_by_id = {
    r["id"]: r
    for r in S1
}

s2_by_id = {
    r["id"]: r
    for r in S2
}

s2_ids = set(s2_by_id.keys())


# ============================================================
# GROUND TRUTH MAP
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
# SELECT S1 SAMPLE
# ============================================================

selected_s1_ids = set(
    r["id"]
    for r in S1
)

print(
    "\nS1 IDs found in ground truth:",
    len(selected_s1_ids & set(gt_map.keys()))
)


# ============================================================
# BUILD TRUE PAIRS INSIDE S2 SAMPLE
# ============================================================

true_pairs = defaultdict(set)

for s1_id in selected_s1_ids:

    gt_matches = gt_map.get(
        s1_id,
        set()
    )

    for entity_id in gt_matches:

        # Only S2 entities that exist
        # in our first 100K S2 sample.
        if entity_id in s2_ids:

            true_pairs[s1_id].add(
                entity_id
            )


total_true_pairs = sum(
    len(x)
    for x in true_pairs.values()
)


print(
    "True S1-S2 pairs inside S2 sample:",
    total_true_pairs
)


# ============================================================
# TOKEN FREQUENCIES IN S1
# ============================================================

name_freq = Counter()
addr_freq = Counter()
num_freq = Counter()

for r in S1:

    country = r["country"]

    for t in r["name_tokens"]:
        name_freq[(country, t)] += 1

    for t in r["address_tokens"]:
        addr_freq[(country, t)] += 1

    for n in r["numbers"]:
        num_freq[(country, n)] += 1


# ============================================================
# INFORMATIVE TOKENS
# ============================================================

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
        addr_freq[(r["country"], x)]
    )


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
        name_freq[(r["country"], x)]
    )


def informative_numbers(r):

    return [

        n

        for n in r["numbers"]

        if num_freq[
            (r["country"], n)
        ] <= MAX_S1_NUM_FREQ

    ]


# ============================================================
# BUILD INDICES
# ============================================================

indices = {

    "NUM_ADDR_BASE":
        defaultdict(list),

    "NUM_ADDR_RARENUM":
        defaultdict(list),

    "NUM_ADDR_2ADDR":
        defaultdict(list),

    "NUM_ADDR_NAME":
        defaultdict(list),
}


print("\nBuilding blocking indices...")


for idx, r in enumerate(S2):

    country = r["country"]

    addr = informative_address_tokens(r)
    name = informative_name_tokens(r)
    nums = informative_numbers(r)

    # --------------------------------------------------------
    # A: current NUM + ADDRESS
    # --------------------------------------------------------

    for n in r["numbers"]:

        for a in addr:

            indices[
                "NUM_ADDR_BASE"
            ][
                (country, n, a)
            ].append(idx)

    # --------------------------------------------------------
    # B: rare NUM + ADDRESS
    # --------------------------------------------------------

    for n in nums:

        for a in addr:

            indices[
                "NUM_ADDR_RARENUM"
            ][
                (country, n, a)
            ].append(idx)

    # --------------------------------------------------------
    # C: number + two address tokens
    # --------------------------------------------------------

    addr_top = addr[:4]

    for n in nums:

        for a1, a2 in combinations(
            addr_top,
            2
        ):

            indices[
                "NUM_ADDR_2ADDR"
            ][
                (
                    country,
                    n,
                    a1,
                    a2
                )
            ].append(idx)

    # --------------------------------------------------------
    # D: number + address + name
    # --------------------------------------------------------

    addr_top = addr[:3]
    name_top = name[:3]

    for n in nums:

        for a in addr_top:

            for nm in name_top:

                indices[
                    "NUM_ADDR_NAME"
                ][
                    (
                        country,
                        n,
                        a,
                        nm
                    )
                ].append(idx)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    s1_records,
    method
):

    index = indices[method]

    candidates = set()

    for r in s1_records:

        country = r["country"]

        addr = informative_address_tokens(r)
        name = informative_name_tokens(r)

        if method == "NUM_ADDR_BASE":

            nums_use = r["numbers"]

            for n in nums_use:

                for a in addr:

                    key = (
                        country,
                        n,
                        a
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

        elif method == "NUM_ADDR_RARENUM":

            nums_use = informative_numbers(r)

            for n in nums_use:

                for a in addr:

                    key = (
                        country,
                        n,
                        a
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

        elif method == "NUM_ADDR_2ADDR":

            nums_use = informative_numbers(r)

            addr_top = addr[:4]

            for n in nums_use:

                for a1, a2 in combinations(
                    addr_top,
                    2
                ):

                    key = (
                        country,
                        n,
                        a1,
                        a2
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

        elif method == "NUM_ADDR_NAME":

            nums_use = informative_numbers(r)

            addr_top = addr[:3]
            name_top = name[:3]

            for n in nums_use:

                for a in addr_top:

                    for nm in name_top:

                        key = (
                            country,
                            n,
                            a,
                            nm
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
# EVALUATION
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
# RUN
# ============================================================

print("\n" + "=" * 70)
print("FINAL NUM_ADDR BLOCKING EXPERIMENT")
print("=" * 70)

results = []

methods = [

    "NUM_ADDR_BASE",

    "NUM_ADDR_RARENUM",

    "NUM_ADDR_2ADDR",

    "NUM_ADDR_NAME",
]


for method in methods:

    print(
        f"\nRunning: {method}"
    )

    candidates = generate_candidates(
        S1,
        method
    )

    candidate_count, recovered, recall = evaluate(
        candidates
    )

    print(
        f"Candidates : {candidate_count:,}"
    )

    print(
        f"True pairs : {total_true_pairs:,}"
    )

    print(
        f"Recovered  : {recovered:,}"
    )

    print(
        f"Recall     : {recall:.2%}"
    )

    results.append({

        "method": method,

        "candidates":
            candidate_count,

        "recovered":
            recovered,

        "recall":
            recall,
    })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for r in results:

    print(

        f"{r['method']:20s} | "

        f"{r['candidates']:10,d} candidates | "

        f"{r['recovered']:6,d} recovered | "

        f"{r['recall']:7.2%} recall"

    )

print("\nDone.")