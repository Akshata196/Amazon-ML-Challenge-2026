import duckdb
import os
import time

DB_PATH = r"D:\Projects\AmazonMLChallenge\final_submission\work\amazon_ml_final.duckdb"
OUTPUT_DIR = r"D:\Projects\AmazonMLChallenge\final_submission\work\candidates"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 2_000

con = duckdb.connect(DB_PATH)

# Safer settings for 8 GB RAM laptop
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='5GB'")
con.execute(
    "SET temp_directory='D:/Projects/AmazonMLChallenge/final_submission/work/duckdb_tmp'"
)

os.makedirs(
    r"D:\Projects\AmazonMLChallenge\final_submission\work\duckdb_tmp",
    exist_ok=True
)

# ---------------------------------------------------------
# Stable S1 order
# ---------------------------------------------------------

print("Preparing S1 processing order...")

con.execute("""
    CREATE OR REPLACE TABLE s1_processing_order AS
    SELECT
        entity_id,
        ROW_NUMBER() OVER (ORDER BY entity_id) AS rn
    FROM train_s1_norm
""")

total_s1 = con.execute(
    "SELECT COUNT(*) FROM s1_processing_order"
).fetchone()[0]

print(f"Total S1 entities: {total_s1:,}")
print(f"Chunk size:        {CHUNK_SIZE:,}")


# ---------------------------------------------------------
# Generate candidates for one source
# ---------------------------------------------------------

def generate_source_candidates(
    source,
    chunk_start,
    chunk_end
):

    if source == "s2":
        name_table = "s2_name_tokens"
        trans_name_table = "s2_trans_name_tokens"
        addr_table = "s2_addr_tokens"
        num_table = "s2_addr_numbers"
    else:
        name_table = "s3_name_tokens"
        trans_name_table = "s3_trans_name_tokens"
        addr_table = "s3_addr_tokens"
        num_table = "s3_addr_numbers"

    # -----------------------------------------------------
    # S1 chunk
    # -----------------------------------------------------

    con.execute("DROP TABLE IF EXISTS chunk_s1_ids")

    con.execute(f"""
        CREATE TEMP TABLE chunk_s1_ids AS
        SELECT entity_id
        FROM s1_processing_order
        WHERE rn > {chunk_start}
          AND rn <= {chunk_end}
    """)

    con.execute("DROP TABLE IF EXISTS chunk_name")
    con.execute("DROP TABLE IF EXISTS chunk_trans_name")
    con.execute("DROP TABLE IF EXISTS chunk_addr")
    con.execute("DROP TABLE IF EXISTS chunk_num")

    con.execute("""
        CREATE TEMP TABLE chunk_name AS
        SELECT t.entity_id, t.country, t.token
        FROM s1_name_tokens t
        INNER JOIN chunk_s1_ids s
            ON t.entity_id = s.entity_id
    """)

    con.execute("""
        CREATE TEMP TABLE chunk_trans_name AS
        SELECT t.entity_id, t.country, t.token
        FROM s1_trans_name_tokens t
        INNER JOIN chunk_s1_ids s
            ON t.entity_id = s.entity_id
    """)

    con.execute("""
        CREATE TEMP TABLE chunk_addr AS
        SELECT t.entity_id, t.country, t.token
        FROM s1_addr_tokens t
        INNER JOIN chunk_s1_ids s
            ON t.entity_id = s.entity_id
    """)

    con.execute("""
        CREATE TEMP TABLE chunk_num AS
        SELECT t.entity_id, t.country, t.number
        FROM s1_addr_numbers t
        INNER JOIN chunk_s1_ids s
            ON t.entity_id = s.entity_id
    """)

    # -----------------------------------------------------
    # NAME ONLY
    #
    # This is intentionally generated first.
    # -----------------------------------------------------

    print(f"    {source}: NAME")

    con.execute("DROP TABLE IF EXISTS cand_name")

    con.execute(f"""
        CREATE TEMP TABLE cand_name AS
        SELECT DISTINCT
            a.entity_id AS s1_entity_id,
            b.entity_id AS matched_entity_id
        FROM chunk_name a
        INNER JOIN {name_table} b
            ON a.country = b.country
           AND a.token = b.token
    """)

    name_count = con.execute(
        "SELECT COUNT(*) FROM cand_name"
    ).fetchone()[0]

    print(f"        NAME candidates: {name_count:,}")

    # -----------------------------------------------------
    # NAME + ADDRESS
    #
    # Instead of one 4-way join, first obtain NAME pairs.
    # Then check address overlap against those pairs.
    # -----------------------------------------------------

    print(f"    {source}: NAME + ADDRESS")

    con.execute("DROP TABLE IF EXISTS cand_name_addr")

    con.execute(f"""
        CREATE TEMP TABLE cand_name_addr AS
        SELECT DISTINCT
            c.s1_entity_id,
            c.matched_entity_id
        FROM cand_name c
        INNER JOIN chunk_addr sa
            ON sa.entity_id = c.s1_entity_id
        INNER JOIN {addr_table} ta
            ON ta.entity_id = c.matched_entity_id
           AND ta.country = sa.country
           AND ta.token = sa.token
    """)

    name_addr_count = con.execute(
        "SELECT COUNT(*) FROM cand_name_addr"
    ).fetchone()[0]

    print(
        f"        NAME+ADDRESS candidates: "
        f"{name_addr_count:,}"
    )

    # -----------------------------------------------------
    # TRANSLITERATED NAME + ADDRESS
    # -----------------------------------------------------

    print(f"    {source}: TRANS NAME + ADDRESS")

    con.execute("DROP TABLE IF EXISTS cand_trans_name")

    con.execute(f"""
        CREATE TEMP TABLE cand_trans_name AS
        SELECT DISTINCT
            a.entity_id AS s1_entity_id,
            b.entity_id AS matched_entity_id
        FROM chunk_trans_name a
        INNER JOIN {trans_name_table} b
            ON a.country = b.country
           AND a.token = b.token
    """)

    con.execute("DROP TABLE IF EXISTS cand_trans_name_addr")

    con.execute(f"""
        CREATE TEMP TABLE cand_trans_name_addr AS
        SELECT DISTINCT
            c.s1_entity_id,
            c.matched_entity_id
        FROM cand_trans_name c
        INNER JOIN chunk_addr sa
            ON sa.entity_id = c.s1_entity_id
        INNER JOIN {addr_table} ta
            ON ta.entity_id = c.matched_entity_id
           AND ta.country = sa.country
           AND ta.token = sa.token
    """)

    trans_count = con.execute(
        "SELECT COUNT(*) FROM cand_trans_name_addr"
    ).fetchone()[0]

    print(
        f"        TRANS+ADDRESS candidates: "
        f"{trans_count:,}"
    )

    # -----------------------------------------------------
    # NUMBER + ADDRESS
    # -----------------------------------------------------

    print(f"    {source}: NUMBER + ADDRESS")

    con.execute("DROP TABLE IF EXISTS cand_num")

    con.execute(f"""
        CREATE TEMP TABLE cand_num AS
        SELECT DISTINCT
            a.entity_id AS s1_entity_id,
            b.entity_id AS matched_entity_id
        FROM chunk_num a
        INNER JOIN {num_table} b
            ON a.country = b.country
           AND a.number = b.number
    """)

    con.execute("DROP TABLE IF EXISTS cand_num_addr")

    con.execute(f"""
        CREATE TEMP TABLE cand_num_addr AS
        SELECT DISTINCT
            c.s1_entity_id,
            c.matched_entity_id
        FROM cand_num c
        INNER JOIN chunk_addr sa
            ON sa.entity_id = c.s1_entity_id
        INNER JOIN {addr_table} ta
            ON ta.entity_id = c.matched_entity_id
           AND ta.country = sa.country
           AND ta.token = sa.token
    """)

    num_count = con.execute(
        "SELECT COUNT(*) FROM cand_num_addr"
    ).fetchone()[0]

    print(
        f"        NUMBER+ADDRESS candidates: "
        f"{num_count:,}"
    )

    # -----------------------------------------------------
    # Final union
    # -----------------------------------------------------

    con.execute("DROP TABLE IF EXISTS chunk_candidates")

    con.execute("""
        CREATE TEMP TABLE chunk_candidates AS

        SELECT s1_entity_id, matched_entity_id
        FROM cand_name_addr

        UNION

        SELECT s1_entity_id, matched_entity_id
        FROM cand_trans_name_addr

        UNION

        SELECT s1_entity_id, matched_entity_id
        FROM cand_num_addr

        UNION

        SELECT s1_entity_id, matched_entity_id
        FROM cand_name
    """)

    total = con.execute(
        "SELECT COUNT(*) FROM chunk_candidates"
    ).fetchone()[0]

    print(
        f"        FINAL {source} candidates: "
        f"{total:,}"
    )

    # -----------------------------------------------------
    # Write Parquet
    # -----------------------------------------------------

    filename = os.path.join(
        OUTPUT_DIR,
        f"candidates_{source}_{chunk_start + 1}_{chunk_end}.parquet"
    )

    con.execute(f"""
        COPY chunk_candidates
        TO '{filename}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    return total


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

overall_start = time.time()
total_candidates = 0

for chunk_start in range(0, total_s1, CHUNK_SIZE):

    chunk_end = min(
        chunk_start + CHUNK_SIZE,
        total_s1
    )

    print()
    print("=" * 70)
    print(
        f"S1 {chunk_start + 1:,} - {chunk_end:,}"
    )
    print("=" * 70)

    chunk_start_time = time.time()

    s2_count = generate_source_candidates(
        "s2",
        chunk_start,
        chunk_end
    )

    s3_count = generate_source_candidates(
        "s3",
        chunk_start,
        chunk_end
    )

    total_candidates += s2_count + s3_count

    elapsed = time.time() - chunk_start_time

    print()
    print(
        f"Chunk time: {elapsed / 60:.2f} minutes"
    )

    print(
        f"Candidates generated so far: "
        f"{total_candidates:,}"
    )


elapsed_total = time.time() - overall_start

print()
print("=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Total candidates: {total_candidates:,}"
)

print(
    f"Total time: {elapsed_total / 3600:.2f} hours"
)

con.close()