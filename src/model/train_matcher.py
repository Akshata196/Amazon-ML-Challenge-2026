import duckdb
import pandas as pd
import numpy as np
import os
import time

from rapidfuzz.fuzz import ratio
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from xgboost import XGBClassifier


DB_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\amazon_ml_final.duckdb"

MODEL_DIR = r"D:\Projects\AmazonMLChallenge\final_submission\work\model"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PARQUET = (
    r"D:\Projects\AmazonMLChallenge\final_submission"
    r"\work\training\train_labeled.parquet"
)

con = duckdb.connect(DB_PATH)

con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='5GB'")
con.execute(
    "SET temp_directory='D:/Projects/AmazonMLChallenge/final_submission/work/duckdb_tmp'"
)

print("=" * 70)
print("TRAINING ENTITY MATCHING MODEL")
print("=" * 70)


# =========================================================
# 1. Load balanced training sample
# =========================================================

print("\nCreating training sample...")

start = time.time()

# Keep all positives.
# Sample negatives at approximately 2:1 relative to positives.
con.execute("""
CREATE OR REPLACE TABLE model_pairs AS

SELECT
    s1_entity_id,
    matched_entity_id,
    label

FROM read_parquet(?)

WHERE label = 1

UNION ALL

SELECT
    s1_entity_id,
    matched_entity_id,
    label

FROM (
    SELECT
        s1_entity_id,
        matched_entity_id,
        label,
        ROW_NUMBER() OVER (
            PARTITION BY s1_entity_id
            ORDER BY hash(matched_entity_id)
        ) AS rn

    FROM read_parquet(?)

    WHERE label = 0
)

WHERE rn <= 6
""", [TRAIN_PARQUET, TRAIN_PARQUET])

pair_count = con.execute(
    "SELECT COUNT(*) FROM model_pairs"
).fetchone()[0]

positive_count = con.execute(
    "SELECT COUNT(*) FROM model_pairs WHERE label=1"
).fetchone()[0]

negative_count = con.execute(
    "SELECT COUNT(*) FROM model_pairs WHERE label=0"
).fetchone()[0]

print(f"Training pairs:  {pair_count:,}")
print(f"Positive pairs:  {positive_count:,}")
print(f"Negative pairs:  {negative_count:,}")

print(
    f"Sample creation time: "
    f"{(time.time() - start) / 60:.2f} minutes"
)


# =========================================================
# 2. Join entity information
# =========================================================

print("\nJoining normalized entity information...")

con.execute("""
CREATE OR REPLACE TABLE model_data AS

SELECT
    p.s1_entity_id,
    p.matched_entity_id,
    p.label,

    s1.business_name AS s1_name,
    s1.business_address AS s1_address,
    s1.name_norm AS s1_name_norm,
    s1.address_norm AS s1_address_norm,
    s1.name_trans AS s1_name_trans,
    s1.address_trans AS s1_address_trans,
    s1.address_numbers AS s1_numbers,
    s1.country AS s1_country,

    COALESCE(
        s2.business_name,
        s3.business_name
    ) AS s2_name,

    COALESCE(
        s2.business_address,
        s3.business_address
    ) AS s2_address,

    COALESCE(
        s2.name_norm,
        s3.name_norm
    ) AS s2_name_norm,

    COALESCE(
        s2.address_norm,
        s3.address_norm
    ) AS s2_address_norm,

    COALESCE(
        s2.name_trans,
        s3.name_trans
    ) AS s2_name_trans,

    COALESCE(
        s2.address_trans,
        s3.address_trans
    ) AS s2_address_trans,

    COALESCE(
        s2.address_numbers,
        s3.address_numbers
    ) AS s2_numbers,

    COALESCE(
        s2.country,
        s3.country
    ) AS s2_country

FROM model_pairs p

INNER JOIN train_s1_norm s1
    ON s1.entity_id = p.s1_entity_id

LEFT JOIN train_s2_norm s2
    ON s2.entity_id = p.matched_entity_id

LEFT JOIN train_s3_norm s3
    ON s3.entity_id = p.matched_entity_id
""")

print("Entity join complete.")


# =========================================================
# 3. Load to pandas
# =========================================================

print("\nLoading model data into pandas...")

df = con.execute("""
SELECT *
FROM model_data
""").fetchdf()

print(f"Rows loaded: {len(df):,}")


# =========================================================
# 4. Feature functions
# =========================================================

def safe_text(x):
    if x is None:
        return ""
    return str(x)


def fuzzy(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def token_jaccard(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def token_overlap(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / min(len(sa), len(sb))


def number_overlap(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / min(len(sa), len(sb))


# =========================================================
# 5. Feature engineering
# =========================================================

print("\nCreating similarity features...")

start = time.time()

features = pd.DataFrame(index=df.index)

# -------------------------
# Name
# -------------------------

features["name_ratio"] = [
    fuzzy(a, b)
    for a, b in zip(
        df["s1_name_norm"],
        df["s2_name_norm"]
    )
]

features["name_token_overlap"] = [
    token_overlap(a, b)
    for a, b in zip(
        df["s1_name_norm"],
        df["s2_name_norm"]
    )
]

features["name_jaccard"] = [
    token_jaccard(a, b)
    for a, b in zip(
        df["s1_name_norm"],
        df["s2_name_norm"]
    )
]

# -------------------------
# Transliteration
# -------------------------

features["trans_name_ratio"] = [
    fuzzy(a, b)
    for a, b in zip(
        df["s1_name_trans"],
        df["s2_name_trans"]
    )
]

features["trans_name_overlap"] = [
    token_overlap(a, b)
    for a, b in zip(
        df["s1_name_trans"],
        df["s2_name_trans"]
    )
]

# -------------------------
# Address
# -------------------------

features["address_ratio"] = [
    fuzzy(a, b)
    for a, b in zip(
        df["s1_address_norm"],
        df["s2_address_norm"]
    )
]

features["address_token_overlap"] = [
    token_overlap(a, b)
    for a, b in zip(
        df["s1_address_norm"],
        df["s2_address_norm"]
    )
]

features["address_jaccard"] = [
    token_jaccard(a, b)
    for a, b in zip(
        df["s1_address_norm"],
        df["s2_address_norm"]
    )
]

# -------------------------
# Transliteration address
# -------------------------

features["trans_address_ratio"] = [
    fuzzy(a, b)
    for a, b in zip(
        df["s1_address_trans"],
        df["s2_address_trans"]
    )
]

# -------------------------
# Numbers
# -------------------------

features["number_overlap"] = [
    number_overlap(a, b)
    for a, b in zip(
        df["s1_numbers"],
        df["s2_numbers"]
    )
]

# -------------------------
# Exact indicators
# -------------------------

features["exact_name"] = (
    df["s1_name_norm"].fillna("") ==
    df["s2_name_norm"].fillna("")
).astype(np.int8)

features["exact_address"] = (
    df["s1_address_norm"].fillna("") ==
    df["s2_address_norm"].fillna("")
).astype(np.int8)

features["country_match"] = (
    df["s1_country"].fillna("") ==
    df["s2_country"].fillna("")
).astype(np.int8)

# -------------------------
# Length features
# -------------------------

features["name_length_diff"] = (
    df["s1_name_norm"].fillna("").str.len() -
    df["s2_name_norm"].fillna("").str.len()
).abs()

features["address_length_diff"] = (
    df["s1_address_norm"].fillna("").str.len() -
    df["s2_address_norm"].fillna("").str.len()
).abs()

print(
    f"Feature creation time: "
    f"{(time.time() - start) / 60:.2f} minutes"
)

print(f"Features: {features.shape[1]}")


# =========================================================
# 6. Prepare X/y
# =========================================================

X = features.astype(np.float32)
y = df["label"].astype(np.int8)


# =========================================================
# 7. Split by S1 entity
# =========================================================

print("\nCreating train/validation split...")

unique_s1 = df["s1_entity_id"].drop_duplicates()

train_s1, val_s1 = train_test_split(
    unique_s1,
    test_size=0.20,
    random_state=42
)

train_mask = df["s1_entity_id"].isin(set(train_s1))
val_mask = df["s1_entity_id"].isin(set(val_s1))

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

print(f"Train rows: {len(X_train):,}")
print(f"Val rows:   {len(X_val):,}")


# =========================================================
# 8. Train XGBoost
# =========================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=42
)

start = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(
    f"Training time: "
    f"{(time.time() - start) / 60:.2f} minutes"
)


# =========================================================
# 9. Validation predictions
# =========================================================

print("\nEvaluating thresholds...")

val_prob = model.predict_proba(X_val)[:, 1]

results = []

for threshold in np.arange(0.50, 0.991, 0.02):

    pred = (val_prob >= threshold).astype(np.int8)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        pred,
        beta=0.5,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f05
        )
    )

print("\nThreshold results:")
print("-" * 60)

for threshold, precision, recall, f05 in results:
    print(
        f"{threshold:.2f}  "
        f"P={precision:.4f}  "
        f"R={recall:.4f}  "
        f"F0.5={f05:.4f}"
    )

best = max(
    results,
    key=lambda x: x[3]
)

best_threshold = best[0]

print("\nSelected threshold:")
print(f"{best_threshold:.2f}")


# =========================================================
# 10. Save model
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "entity_matcher.json"
)

model.save_model(model_path)

# Save feature names
feature_path = os.path.join(
    MODEL_DIR,
    "feature_names.txt"
)

with open(feature_path, "w", encoding="utf-8") as f:
    for name in X.columns:
        f.write(name + "\n")

# Save threshold
threshold_path = os.path.join(
    MODEL_DIR,
    "threshold.txt"
)

with open(threshold_path, "w") as f:
    f.write(str(best_threshold))

print("\nModel saved:")
print(model_path)

print("Feature names saved:")
print(feature_path)

print("Threshold saved:")
print(threshold_path)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

con.close()