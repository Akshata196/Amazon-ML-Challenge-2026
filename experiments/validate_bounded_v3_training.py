import duckdb
import pandas as pd
import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio
import xgboost as xgb

DB_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\amazon_ml_final.duckdb"
MODEL_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\model\entity_matcher.json"

SAMPLE_SIZE = 500
TOP_K = 20

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("BOUNDED V3 TRAINING VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Sample training S1 records
# ------------------------------------------------------------

s1 = con.execute(f"""
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
    FROM train_s1_norm
    USING SAMPLE {SAMPLE_SIZE} ROWS
""").df()

print(f"Training S1 sample: {len(s1)}")

# ------------------------------------------------------------
# 2. Ground truth
# ------------------------------------------------------------

gt = con.execute("""
    SELECT
        source1_entity_id,
        matched_entity_ids
    FROM train_gt
""").df()

gt_map = {}

for _, row in gt.iterrows():
    ids = str(row["matched_entity_ids"])

    if ids == "nan" or ids.strip() == "":
        gt_map[row["source1_entity_id"]] = set()
    else:
        gt_map[row["source1_entity_id"]] = set(
            x.strip()
            for x in ids.split(",")
            if x.strip()
        )

# ------------------------------------------------------------
# 3. Generate bounded candidates
#    Same strategy as V3:
#    longest name token >= 5 characters
# ------------------------------------------------------------

candidates = []

print("\nGenerating bounded candidates...")

for idx, row in s1.iterrows():

    tokens = str(row["name_norm"]).split()

    tokens = [
        t for t in tokens
        if len(t) >= 5
    ]

    if not tokens:
        continue

    blocking_token = max(tokens, key=len)

    # S2
    q = """
        SELECT
            entity_id AS candidate_id,
            business_name,
            business_address,
            country,
            name_norm,
            address_norm,
            name_trans,
            address_trans,
            address_numbers,
            'S2' AS source
        FROM train_s2_norm
        WHERE list_contains(
            string_split(name_norm, ' '),
            ?
        )
        LIMIT ?
    """

    s2 = con.execute(
        q,
        [blocking_token, TOP_K]
    ).df()

    # S3
    q = """
        SELECT
            entity_id AS candidate_id,
            business_name,
            business_address,
            country,
            name_norm,
            address_norm,
            name_trans,
            address_trans,
            address_numbers,
            'S3' AS source
        FROM train_s3_norm
        WHERE list_contains(
            string_split(name_norm, ' '),
            ?
        )
        LIMIT ?
    """

    s3 = con.execute(
        q,
        [blocking_token, TOP_K]
    ).df()

    for df in [s2, s3]:

        for _, c in df.iterrows():

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

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(s1)}")

cand = pd.DataFrame(candidates)

print(f"\nCandidate pairs: {len(cand)}")

if cand.empty:
    print("No candidates generated.")
    raise SystemExit

# ------------------------------------------------------------
# 4. Feature calculation
# ------------------------------------------------------------

print("\nBuilding model features...")

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

X = pd.DataFrame(features, columns=feature_names)

# ------------------------------------------------------------
# 5. Score
# ------------------------------------------------------------

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

scores = model.predict_proba(X)[:, 1]

cand["score"] = scores

# ------------------------------------------------------------
# 6. Ground truth labels
# ------------------------------------------------------------

cand["is_true"] = cand.apply(
    lambda r:
        r["candidate_id"]
        in gt_map.get(r["s1_id"], set()),
    axis=1
)

# ------------------------------------------------------------
# 7. Best candidate per S1
# ------------------------------------------------------------

best = (
    cand
    .sort_values("score", ascending=False)
    .groupby("s1_id")
    .head(1)
    .copy()
)

print("\n" + "=" * 70)
print("V3 TRAINING VALIDATION RESULTS")
print("=" * 70)

print(f"Candidate pairs: {len(cand)}")
print(f"S1 with candidates: {len(best)}")
print(f"True candidate pairs: {cand['is_true'].sum()}")
print(f"True best candidates: {best['is_true'].sum()}")

# ------------------------------------------------------------
# 8. Precision at thresholds
# ------------------------------------------------------------

for threshold in [0.78, 0.90, 0.95, 0.98, 0.99]:

    selected = best[best["score"] >= threshold]

    if len(selected) == 0:
        precision = 0
    else:
        precision = selected["is_true"].mean()

    print(
        f"\nThreshold >= {threshold:.2f}"
    )
    print(
        f"Selected: {len(selected)}"
    )
    print(
        f"Correct: {selected['is_true'].sum()}"
    )
    print(
        f"Precision: {precision:.4f}"
    )

# ------------------------------------------------------------
# 9. Print high-confidence candidates
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CANDIDATES")
print("=" * 70)

high = (
    best[best["score"] >= 0.98]
    [
        [
            "s1_id",
            "candidate_id",
            "s1_name",
            "candidate_name",
            "score",
            "is_true",
        ]
    ]
    .sort_values("score", ascending=False)
)

print(high.to_string(index=False))

# ------------------------------------------------------------
# 10. Save result
# ------------------------------------------------------------

out_path = (
    r"D:\Projects\AmazonMLChallenge\final_submission"
    r"\work\bounded_v3_training_results.csv"
)

high.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
