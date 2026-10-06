import os
import pandas as pd
from unidecode import unidecode
from rapidfuzz.fuzz import ratio, token_set_ratio


BASE_DIR = r"D:\Projects\AmazonMLChallenge\student_resource"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

S1_SAMPLE = 5000


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


# ------------------------------------------------------------
# LOAD S1
# ------------------------------------------------------------

print("=" * 70)
print("STEP 1: Loading S1")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_SAMPLE,
    dtype=str
).fillna("")

s1_lookup = s1.set_index("entity_id").to_dict("index")


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(s1_lookup)
]

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    if not value:
        continue

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


# ------------------------------------------------------------
# LOAD ONLY TRUE MATCH RECORDS
# ------------------------------------------------------------

def load_required(file_path, required_ids):

    result = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=200_000
    ):

        chunk = chunk.fillna("")

        selected = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in selected.iterrows():

            result[row["entity_id"]] = {
                "business_name": row["business_name"],
                "business_address": row["business_address"],
                "country": row["country"]
            }

        if len(result) == len(required_ids):
            break

    return result


print("\nLoading true S2 records...")
s2 = load_required(S2_FILE, s2_ids)

print(f"S2 loaded: {len(s2):,}")

print("\nLoading true S3 records...")
s3 = load_required(S3_FILE, s3_ids)

print(f"S3 loaded: {len(s3):,}")


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

total = 0

name_improved = 0
address_improved = 0

original_name_scores = []
translit_name_scores = []

original_address_scores = []
translit_address_scores = []

cross_script_cases = 0
cross_script_improved = 0


for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    s1_record = s1_lookup[s1_id]

    s1_name = normalize(s1_record["business_name"])
    s1_addr = normalize(s1_record["business_address"])

    s1_name_trans = transliterate(
        s1_record["business_name"]
    )

    s1_addr_trans = transliterate(
        s1_record["business_address"]
    )

    matched = str(row["matched_entity_ids"]).strip()

    if not matched:
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id.startswith("S2-"):
            candidate = s2.get(candidate_id)
        else:
            candidate = s3.get(candidate_id)

        if candidate is None:
            continue

        total += 1

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        c_name = normalize(candidate["business_name"])
        c_name_trans = transliterate(
            candidate["business_name"]
        )

        original_name = ratio(
            s1_name,
            c_name
        ) / 100.0

        translit_name = ratio(
            s1_name_trans,
            c_name_trans
        ) / 100.0

        original_name_scores.append(original_name)
        translit_name_scores.append(translit_name)

        if translit_name > original_name:
            name_improved += 1

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        c_addr = normalize(candidate["business_address"])
        c_addr_trans = transliterate(
            candidate["business_address"]
        )

        original_addr = ratio(
            s1_addr,
            c_addr
        ) / 100.0

        translit_addr = ratio(
            s1_addr_trans,
            c_addr_trans
        ) / 100.0

        original_address_scores.append(original_addr)
        translit_address_scores.append(translit_addr)

        if translit_addr > original_addr:
            address_improved += 1

        # ----------------------------------------------------
        # CROSS-SCRIPT DETECTION
        # ----------------------------------------------------

        original_name_ascii = all(
            ord(ch) < 128
            for ch in str(s1_record["business_name"])
            + str(candidate["business_name"])
        )

        if not original_name_ascii:

            cross_script_cases += 1

            if translit_name > original_name:
                cross_script_improved += 1


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRANSLITERATION RESULTS")
print("=" * 70)

print(f"\nTotal true pairs: {total:,}")

print(
    f"\nName similarity improved in:"
    f" {name_improved:,} / {total:,}"
    f" ({name_improved / total:.2%})"
)

print(
    f"Address similarity improved in:"
    f" {address_improved:,} / {total:,}"
    f" ({address_improved / total:.2%})"
)

print(
    f"\nAverage original name similarity:"
    f" {sum(original_name_scores) / total:.4f}"
)

print(
    f"Average transliterated name similarity:"
    f" {sum(translit_name_scores) / total:.4f}"
)

print(
    f"\nAverage original address similarity:"
    f" {sum(original_address_scores) / total:.4f}"
)

print(
    f"Average transliterated address similarity:"
    f" {sum(translit_address_scores) / total:.4f}"
)

print(
    f"\nCross-script cases detected:"
    f" {cross_script_cases:,}"
)

if cross_script_cases:
    print(
        f"Cross-script cases improved:"
        f" {cross_script_improved:,}"
        f" ({cross_script_improved / cross_script_cases:.2%})"
    )

print("\nDONE.")