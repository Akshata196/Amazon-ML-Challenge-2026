import duckdb
import os
import time

DB_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\amazon_ml_final.duckdb"

OUT_DIR = r"D:\Projects\AmazonMLChallenge\final_submission\work\training"
os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect(DB_PATH)

con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='5GB'")
con.execute(
    "SET temp_directory='D:/Projects/AmazonMLChallenge/final_submission/work/duckdb_tmp'"
)

print("=" * 70)
print("PREPARING TRAINING PAIRS")
print("=" * 70)

# ---------------------------------------------------------
# 1. Ground truth
# ---------------------------------------------------------

print("\nLoading ground truth...")

con.execute("""
CREATE OR REPLACE TABLE gt_pairs AS
SELECT
    source1_entity_id AS s1_entity_id,
    TRIM(matched_entity_id) AS matched_entity_id
FROM (
    SELECT
        source1_entity_id,
        UNNEST(
            STRING_SPLIT(
                COALESCE(matched_entity_ids, ''),
                ','
            )
        ) AS matched_entity_id
    FROM train_gt
)
WHERE TRIM(matched_entity_id) <> ''
""")

gt_count = con.execute(
    "SELECT COUNT(*) FROM gt_pairs"
).fetchone()[0]

print(f"Positive GT pairs: {gt_count:,}")


# ---------------------------------------------------------
# 2. Exact normalized-name candidates
# ---------------------------------------------------------

print("\nGenerating exact NAME candidates...")

start = time.time()

con.execute("""
CREATE OR REPLACE TABLE train_exact_name AS

SELECT DISTINCT
    s1.entity_id AS s1_entity_id,
    s2.entity_id AS matched_entity_id

FROM train_s1_norm s1

INNER JOIN train_s2_norm s2
    ON s1.country = s2.country
   AND s1.name_norm <> ''
   AND s1.name_norm = s2.name_norm
""")

print(
    f"Created in {(time.time() - start) / 60:.2f} minutes"
)

count = con.execute(
    "SELECT COUNT(*) FROM train_exact_name"
).fetchone()[0]

print(f"S1-S2 exact-name candidates: {count:,}")


# ---------------------------------------------------------
# 3. Exact normalized-name S3
# ---------------------------------------------------------

print("\nGenerating exact NAME candidates for S3...")

start = time.time()

con.execute("""
CREATE OR REPLACE TABLE train_exact_name_s3 AS

SELECT DISTINCT
    s1.entity_id AS s1_entity_id,
    s3.entity_id AS matched_entity_id

FROM train_s1_norm s1

INNER JOIN train_s3_norm s3
    ON s1.country = s3.country
   AND s1.name_norm <> ''
   AND s1.name_norm = s3.name_norm
""")

print(
    f"Created in {(time.time() - start) / 60:.2f} minutes"
)

count = con.execute(
    "SELECT COUNT(*) FROM train_exact_name_s3"
).fetchone()[0]

print(f"S1-S3 exact-name candidates: {count:,}")


# ---------------------------------------------------------
# 4. Exact normalized address
# ---------------------------------------------------------

print("\nGenerating exact ADDRESS candidates...")

start = time.time()

con.execute("""
CREATE OR REPLACE TABLE train_exact_addr AS

SELECT DISTINCT
    s1.entity_id AS s1_entity_id,
    s2.entity_id AS matched_entity_id

FROM train_s1_norm s1

INNER JOIN train_s2_norm s2
    ON s1.country = s2.country
   AND s1.address_norm <> ''
   AND s1.address_norm = s2.address_norm
""")

print(
    f"Created in {(time.time() - start) / 60:.2f} minutes"
)

count = con.execute(
    "SELECT COUNT(*) FROM train_exact_addr"
).fetchone()[0]

print(f"S1-S2 exact-address candidates: {count:,}")


# ---------------------------------------------------------
# 5. Exact address S3
# ---------------------------------------------------------

print("\nGenerating exact ADDRESS candidates for S3...")

start = time.time()

con.execute("""
CREATE OR REPLACE TABLE train_exact_addr_s3 AS

SELECT DISTINCT
    s1.entity_id AS s1_entity_id,
    s3.entity_id AS matched_entity_id

FROM train_s1_norm s1

INNER JOIN train_s3_norm s3
    ON s1.country = s3.country
   AND s1.address_norm <> ''
   AND s1.address_norm = s3.address_norm
""")

print(
    f"Created in {(time.time() - start) / 60:.2f} minutes"
)

count = con.execute(
    "SELECT COUNT(*) FROM train_exact_addr_s3"
).fetchone()[0]

print(f"S1-S3 exact-address candidates: {count:,}")


# ---------------------------------------------------------
# 6. Combine candidates
# ---------------------------------------------------------

print("\nCombining candidates...")

con.execute("""
CREATE OR REPLACE TABLE train_candidates AS

SELECT s1_entity_id, matched_entity_id
FROM train_exact_name

UNION

SELECT s1_entity_id, matched_entity_id
FROM train_exact_addr

UNION

SELECT s1_entity_id, matched_entity_id
FROM train_exact_name_s3

UNION

SELECT s1_entity_id, matched_entity_id
FROM train_exact_addr_s3
""")

candidate_count = con.execute(
    "SELECT COUNT(*) FROM train_candidates"
).fetchone()[0]

print(f"Total candidates: {candidate_count:,}")


# ---------------------------------------------------------
# 7. Label candidates
# ---------------------------------------------------------

print("\nLabelling candidates...")

con.execute("""
CREATE OR REPLACE TABLE train_labeled AS

SELECT
    c.s1_entity_id,
    c.matched_entity_id,

    CASE
        WHEN g.s1_entity_id IS NOT NULL
        THEN 1
        ELSE 0
    END AS label

FROM train_candidates c

LEFT JOIN gt_pairs g
    ON c.s1_entity_id = g.s1_entity_id
   AND c.matched_entity_id = g.matched_entity_id
""")

positive_candidates = con.execute("""
SELECT COUNT(*)
FROM train_labeled
WHERE label = 1
""").fetchone()[0]

negative_candidates = con.execute("""
SELECT COUNT(*)
FROM train_labeled
WHERE label = 0
""").fetchone()[0]

print(f"Positive candidates: {positive_candidates:,}")
print(f"Negative candidates: {negative_candidates:,}")


# ---------------------------------------------------------
# 8. Export
# ---------------------------------------------------------

output_file = os.path.join(
    OUT_DIR,
    "train_labeled.parquet"
)

con.execute(f"""
COPY train_labeled
TO '{output_file}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print(f"\nSaved:")
print(output_file)

print("\n" + "=" * 70)
print("TRAINING PAIR PREPARATION COMPLETE")
print("=" * 70)

con.close()