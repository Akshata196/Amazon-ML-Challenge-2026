import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from rapidfuzz.fuzz import ratio

DB_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\amazon_ml_final.duckdb"

MODEL_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\model\entity_matcher.json"

V2_MATCHING = r"D:\Projects\AmazonMLChallenge\final_submission\output\matching_results.tsv"
V2_CANDIDATE = r"D:\Projects\AmazonMLChallenge\final_submission\output\candidate_pairs.tsv"

V3_MATCHING = r"D:\Projects\AmazonMLChallenge\final_submission\output\matching_results_v3.tsv"
V3_CANDIDATE = r"D:\Projects\AmazonMLChallenge\final_submission\output\candidate_pairs_v3.tsv"

THRESHOLD = 0.99

# Process unmatched S1 in chunks.
CHUNK_SIZE = 5000

# Same bounded strategy validated on training.
TOP_K = 20


# ============================================================
# Helpers
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def token_overlap(a, b):
    A = set(safe_str(a).split())
    B = set(safe_str(b).split())

    if not A or not B:
        return 0.0

    return len(A & B) / min(len(A), len(B))


def jaccard(a, b):
    A = set(safe_str(a).split())
    B = set(safe_str(b).split())

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def number_overlap(a, b):
    A = set(safe_str(a).split())
    B = set(safe_str(b).split())

    if not A or not B:
        return 0.0

    return len(A & B) / min(len(A), len(B))


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("CONSERVATIVE V3 GENERATION")
print("=" * 70)

print(f"Threshold: {THRESHOLD}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Top K per source: {TOP_K}")

con = duckdb.connect(DB_PATH)

# ------------------------------------------------------------
# Load V2
# ------------------------------------------------------------

print("\nLoading V2 submission...")

v2_match = pd.read_csv(
    V2_MATCHING,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

v2_candidates = pd.read_csv(
    V2_CANDIDATE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"V2 matching rows: {len(v2_match):,}")
print(f"V2 candidate rows: {len(v2_candidates):,}")

# Existing V2 matched IDs.
v2_matched = {}

for _, row in v2_match.iterrows():

    s1_id = row["source1_entity_id"]
    ids = row["matched_entity_ids"]

    if ids.strip():
        v2_matched[s1_id] = [
            x.strip()
            for x in ids.split(",")
            if x.strip()
        ]
    else:
        v2_matched[s1_id] = []


unmatched_ids = [
    s1_id
    for s1_id, ids in v2_matched.items()
    if len(ids) == 0
]

print(f"V2 unmatched S1: {len(unmatched_ids):,}")


# ============================================================
# Model
# ============================================================

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("Model loaded.")


# ============================================================
# Candidate processing
# ============================================================

new_matches = {}

total_candidates = 0
total_scored = 0
total_accepted = 0
s1_with_candidate = 0
s1_with_new_match = 0

# Convert unmatched IDs to dataframe for chunking.
unmatched_df = pd.DataFrame({
    "s1_id": unmatched_ids
})


for chunk_start in range(
    0,
    len(unmatched_df),
    CHUNK_SIZE
):

    chunk = unmatched_df.iloc[
        chunk_start:
        chunk_start + CHUNK_SIZE
    ]

    ids = chunk["s1_id"].tolist()

    print(
        f"\nProcessing "
        f"{chunk_start + 1:,}-"
        f"{min(chunk_start + CHUNK_SIZE, len(unmatched_df)):,}"
        f" / {len(unmatched_df):,}"
    )

    # --------------------------------------------------------
    # Load S1 records
    # --------------------------------------------------------

    placeholders = ",".join(["?"] * len(ids))

    s1 = con.execute(
        f"""
        SELECT
            entity_id AS s1_id,
            business_name,
            business_address,
            country,
            name_norm,
            address_norm,
            name_trans,
            address_trans,
            address_numbers
        FROM test_s1_norm
        WHERE entity_id IN ({placeholders})
        """,
        ids
    ).df()

    if s1.empty:
        continue

    candidates = []

    # --------------------------------------------------------
    # Generate bounded candidates
    # --------------------------------------------------------

    for _, row in s1.iterrows():

        tokens = safe_str(
            row["name_norm"]
        ).split()

        tokens = [
            t for t in tokens
            if len(t) >= 5
        ]

        if not tokens:
            continue

        blocking_token = max(
            tokens,
            key=len
        )

        # S2
        s2 = con.execute(
            """
            SELECT
                entity_id AS candidate_id,
                business_name,
                business_address,
                country,
                name_norm,
                address_norm,
                name_trans,
                address_trans,
                address_numbers
            FROM test_s2_norm
            WHERE list_contains(
                string_split(name_norm, ' '),
                ?
            )
            LIMIT ?
            """,
            [blocking_token, TOP_K]
        ).df()

        # S3
        s3 = con.execute(
            """
            SELECT
                entity_id AS candidate_id,
                business_name,
                business_address,
                country,
                name_norm,
                address_norm,
                name_trans,
                address_trans,
                address_numbers
            FROM test_s3_norm
            WHERE list_contains(
                string_split(name_norm, ' '),
                ?
            )
            LIMIT ?
            """,
            [blocking_token, TOP_K]
        ).df()

        if not s2.empty or not s3.empty:
            s1_with_candidate += 1

        for cdf in [s2, s3]:

            for _, c in cdf.iterrows():

                candidates.append({
                    "s1_id": row["s1_id"],
                    "candidate_id": c["candidate_id"],

                    "s1_name": row["business_name"],
                    "candidate_name": c["business_name"],

                    "s1_address": row["business_address"],
                    "candidate_address": c["business_address"],

                    "s1_country": row["country"],
                    "candidate_country": c["country"],

                    "name_norm": row["name_norm"],
                    "candidate_name_norm": c["name_norm"],

                    "address_norm": row["address_norm"],
                    "candidate_address_norm": c["address_norm"],

                    "name_trans": row["name_trans"],
                    "candidate_name_trans": c["name_trans"],

                    "address_trans": row["address_trans"],
                    "candidate_address_trans": c["address_trans"],

                    "address_numbers": row["address_numbers"],
                    "candidate_address_numbers": c["address_numbers"],
                })

    if not candidates:
        continue

    cand = pd.DataFrame(candidates)

    total_candidates += len(cand)

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = []

    for _, r in cand.iterrows():

        name = safe_str(r["name_norm"])
        cname = safe_str(r["candidate_name_norm"])

        addr = safe_str(r["address_norm"])
        caddr = safe_str(r["candidate_address_norm"])

        tname = safe_str(r["name_trans"])
        ctname = safe_str(r["candidate_name_trans"])

        taddr = safe_str(r["address_trans"])
        ctaddr = safe_str(r["candidate_address_trans"])

        features.append([
            ratio(name, cname) / 100,
            token_overlap(name, cname),
            jaccard(name, cname),

            ratio(tname, ctname) / 100,
            token_overlap(tname, ctname),

            ratio(addr, caddr) / 100,
            token_overlap(addr, caddr),
            jaccard(addr, caddr),

            ratio(taddr, ctaddr) / 100,

            number_overlap(
                r["address_numbers"],
                r["candidate_address_numbers"]
            ),

            int(name == cname),
            int(addr == caddr),

            int(
                safe_str(r["s1_country"]).lower()
                ==
                safe_str(r["candidate_country"]).lower()
            ),

            abs(len(name) - len(cname)),
            abs(len(addr) - len(caddr)),
        ])

    feature_names = [
        "name_ratio",
        "name_token_overlap",
        "name_jaccard",
        "trans_name_ratio",
        "trans_name_overlap",
        "address_ratio",
        "address_token_overlap",
        "address_jaccard",
        "trans_address_ratio",
        "number_overlap",
        "exact_name",
        "exact_address",
        "country_match",
        "name_length_diff",
        "address_length_diff",
    ]

    X = pd.DataFrame(
        features,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scores = model.predict_proba(X)[:, 1]

    cand["score"] = scores

    total_scored += len(cand)

    # --------------------------------------------------------
    # Best candidate per S1
    # --------------------------------------------------------

    best = (
        cand
        .sort_values(
            ["s1_id", "score"],
            ascending=[True, False]
        )
        .groupby("s1_id", as_index=False)
        .head(1)
    )

    # --------------------------------------------------------
    # Strict threshold
    # --------------------------------------------------------

    accepted = best[
        best["score"] >= THRESHOLD
    ]

    total_accepted += len(accepted)

    for _, r in accepted.iterrows():

        s1_id = r["s1_id"]
        candidate_id = r["candidate_id"]

        new_matches[s1_id] = candidate_id

    if len(accepted) > 0:
        s1_with_new_match += len(accepted)

    print(
        f"Candidates: {len(cand):,} | "
        f"Accepted: {len(accepted):,}"
    )


# ============================================================
# Create V3 matching file
# ============================================================

print("\n" + "=" * 70)
print("CREATING V3 OUTPUT")
print("=" * 70)

v3_match = v2_match.copy()

added = 0

for idx, row in v3_match.iterrows():

    s1_id = row["source1_entity_id"]

    # Only add to previously unmatched S1.
    if (
        not row["matched_entity_ids"].strip()
        and s1_id in new_matches
    ):

        v3_match.at[
            idx,
            "matched_entity_ids"
        ] = new_matches[s1_id]

        added += 1

v3_match.to_csv(
    V3_MATCHING,
    sep="\t",
    index=False
)

print(f"New V3 matches added: {added:,}")

# ============================================================
# Candidate file
#
# For validator, include the new accepted candidates in the
# candidate set while preserving every V2 candidate.
# ============================================================

print("\nCreating V3 candidate file...")

v3_candidates = v2_candidates.copy()

existing_pairs = set()

for _, row in v3_candidates.iterrows():

    s1_id = row["source1_entity_id"]
    ids = row["candidate_entity_ids"]

    if ids.strip():

        for cid in ids.split(","):

            cid = cid.strip()

            if cid:
                existing_pairs.add(
                    (s1_id, cid)
                )


# Add newly accepted candidates.
additional = {}

for s1_id, cid in new_matches.items():

    if (s1_id, cid) in existing_pairs:
        continue

    additional.setdefault(
        s1_id,
        []
    ).append(cid)

for idx, row in v3_candidates.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in additional:
        continue

    current = row["candidate_entity_ids"].strip()

    new_ids = additional[s1_id]

    if current:
        combined = (
            current.split(",")
            + new_ids
        )
    else:
        combined = new_ids

    # Remove duplicates while preserving order.
    seen = set()
    result = []

    for cid in combined:

        cid = cid.strip()

        if cid and cid not in seen:
            seen.add(cid)
            result.append(cid)

    v3_candidates.at[
        idx,
        "candidate_entity_ids"
    ] = ",".join(result)

v3_candidates.to_csv(
    V3_CANDIDATE,
    sep="\t",
    index=False
)

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("V3 SUMMARY")
print("=" * 70)

print(
    f"V2 unmatched S1: "
    f"{len(unmatched_ids):,}"
)

print(
    f"S1 with bounded candidates: "
    f"{s1_with_candidate:,}"
)

print(
    f"Total candidates scored: "
    f"{total_scored:,}"
)

print(
    f"Accepted >= {THRESHOLD}: "
    f"{total_accepted:,}"
)

print(
    f"New V3 matches: "
    f"{added:,}"
)

print("\nFiles created:")

print(V3_MATCHING)
print(V3_CANDIDATE)

print("\nV2 files were NOT modified.")

print("=" * 70)
print("V3 GENERATION COMPLETE")
print("=" * 70)