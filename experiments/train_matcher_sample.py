import pandas as pd
import numpy as np
import re
import unicodedata

from collections import defaultdict, Counter
from itertools import combinations

from rapidfuzz.fuzz import ratio, token_set_ratio
from unidecode import unidecode

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from xgboost import XGBClassifier


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

RANDOM_STATE = 42

MAX_NEGATIVES_PER_S1 = 10


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    ).lower()

    result = []

    for ch in text:

        cat = unicodedata.category(ch)

        if cat[0] in ("L", "M", "N"):
            result.append(ch)

        elif ch.isspace():
            result.append(" ")

        else:
            result.append(" ")

    return " ".join(
        "".join(result).split()
    )


def transliterate(text):

    return normalize(
        unidecode(text)
    )


def tokens(text):

    return set(
        text.split()
    )


def numbers(text):

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


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

print(
    f"S1: {len(s1_df):,}"
)

print(
    f"S2: {len(s2_df):,}"
)

print(
    f"GT: {len(gt_df):,}"
)


# ============================================================
# PREPARE RECORDS
# ============================================================

def prepare(df):

    records = []

    for _, r in df.iterrows():

        name = normalize(
            r["business_name"]
        )

        address = normalize(
            r["business_address"]
        )

        trans_name = transliterate(
            name
        )

        records.append({

            "id":
                str(r["entity_id"]).strip(),

            "country":
                normalize(r["country"]),

            "name":
                name,

            "address":
                address,

            "name_tokens":
                tokens(name),

            "address_tokens":
                tokens(address),

            "numbers":
                numbers(address),

            "trans_name":
                trans_name,

            "trans_name_tokens":
                tokens(trans_name),
        })

    return records


print("Preparing records...")

S1 = prepare(s1_df)
S2 = prepare(s2_df)


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
# TOKEN FREQUENCY FROM S1
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
# BUILD BLOCK INDICES
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


print("Building blocking indices...")


for idx, r in enumerate(S2):

    country = r["country"]

    name_tokens = informative_name_tokens(r)

    addr_tokens = informative_address_tokens(r)

    # --------------------------------------------------------
    # NAME + ADDRESS
    # --------------------------------------------------------

    for nt in name_tokens:

        for at in addr_tokens:

            indices[
                "NAME_ADDR"
            ][
                (
                    country,
                    nt,
                    at
                )
            ].append(idx)

    # --------------------------------------------------------
    # TRANSLITERATED NAME + ADDRESS
    # --------------------------------------------------------

    trans_tokens = [

        t

        for t in r["trans_name_tokens"]

        if name_freq[
            (country, t)
        ] <= MAX_S1_TOKEN_FREQ

    ]

    for nt in trans_tokens:

        for at in addr_tokens:

            indices[
                "TRANS_NAME_ADDR"
            ][
                (
                    country,
                    nt,
                    at
                )
            ].append(idx)

    # --------------------------------------------------------
    # NUMBER + ADDRESS
    # --------------------------------------------------------

    for num in r["numbers"]:

        for at in addr_tokens:

            indices[
                "NUM_ADDR"
            ][
                (
                    country,
                    num,
                    at
                )
            ].append(idx)

    # --------------------------------------------------------
    # NAME + NAME
    # --------------------------------------------------------

    for n1, n2 in combinations(
        name_tokens,
        2
    ):

        indices[
            "NAME_NAME"
        ][
            (
                country,
                n1,
                n2
            )
        ].append(idx)


# ============================================================
# GENERATE ALL BLOCKING CANDIDATES
# ============================================================

print("Generating candidates...")


candidate_pairs = set()


for r in S1:

    s1_id = r["id"]

    country = r["country"]

    name_tokens = informative_name_tokens(r)

    addr_tokens = informative_address_tokens(r)

    # --------------------------------------------------------
    # NAME_ADDR
    # --------------------------------------------------------

    for nt in name_tokens:

        for at in addr_tokens:

            key = (
                country,
                nt,
                at
            )

            for idx in indices[
                "NAME_ADDR"
            ].get(key, []):

                candidate_pairs.add(
                    (
                        s1_id,
                        idx
                    )
                )

    # --------------------------------------------------------
    # TRANS_NAME_ADDR
    # --------------------------------------------------------

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

            for idx in indices[
                "TRANS_NAME_ADDR"
            ].get(key, []):

                candidate_pairs.add(
                    (
                        s1_id,
                        idx
                    )
                )

    # --------------------------------------------------------
    # NUM_ADDR
    # --------------------------------------------------------

    for num in r["numbers"]:

        for at in addr_tokens:

            key = (
                country,
                num,
                at
            )

            for idx in indices[
                "NUM_ADDR"
            ].get(key, []):

                candidate_pairs.add(
                    (
                        s1_id,
                        idx
                    )
                )

    # --------------------------------------------------------
    # NAME_NAME
    # --------------------------------------------------------

    for n1, n2 in combinations(
        name_tokens,
        2
    ):

        key = (
            country,
            n1,
            n2
        )

        for idx in indices[
            "NAME_NAME"
        ].get(key, []):

            candidate_pairs.add(
                (
                    s1_id,
                    idx
                )
            )


print(
    "Total candidates:",
    f"{len(candidate_pairs):,}"
)


# ============================================================
# ORGANIZE CANDIDATES BY S1
# ============================================================

candidates_by_s1 = defaultdict(list)

for s1_id, s2_idx in candidate_pairs:

    candidates_by_s1[
        s1_id
    ].append(s2_idx)


# ============================================================
# CREATE LOOKUP FOR S1
# ============================================================

s1_by_id = {
    r["id"]: r
    for r in S1
}


# ============================================================
# LABEL CANDIDATES
# ============================================================

print("Creating positive and hard-negative examples...")


positive_count = 0
negative_count = 0

training_rows = []

rng = np.random.default_rng(
    RANDOM_STATE
)


for s1_id, candidate_indices in candidates_by_s1.items():

    true_ids = gt_map.get(
        s1_id,
        set()
    )

    positive_indices = []
    negative_indices = []

    for idx in candidate_indices:

        s2_id = S2[idx]["id"]

        if s2_id in true_ids:

            positive_indices.append(idx)

        else:

            negative_indices.append(idx)

    # --------------------------------------------------------
    # ALL POSITIVES
    # --------------------------------------------------------

    for idx in positive_indices:

        training_rows.append({

            "s1_id":
                s1_id,

            "s2_idx":
                idx,

            "label":
                1
        })

        positive_count += 1

    # --------------------------------------------------------
    # SAMPLE HARD NEGATIVES
    # --------------------------------------------------------

    if len(negative_indices) > MAX_NEGATIVES_PER_S1:

        negative_indices = rng.choice(
            negative_indices,
            size=MAX_NEGATIVES_PER_S1,
            replace=False
        )

    for idx in negative_indices:

        training_rows.append({

            "s1_id":
                s1_id,

            "s2_idx":
                int(idx),

            "label":
                0
        })

        negative_count += 1


print(
    "Positive examples:",
    f"{positive_count:,}"
)

print(
    "Negative examples:",
    f"{negative_count:,}"
)

print(
    "Total training examples:",
    f"{len(training_rows):,}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def pair_features(
    s1,
    s2
):

    name = s1["name"]
    s2_name = s2["name"]

    address = s1["address"]
    s2_address = s2["address"]

    trans_name = s1["trans_name"]
    s2_trans_name = s2["trans_name"]

    name_tokens = s1["name_tokens"]
    s2_name_tokens = s2["name_tokens"]

    addr_tokens = s1["address_tokens"]
    s2_addr_tokens = s2["address_tokens"]

    nums1 = s1["numbers"]
    nums2 = s2["numbers"]

    shared_name = (
        name_tokens &
        s2_name_tokens
    )

    shared_address = (
        addr_tokens &
        s2_addr_tokens
    )

    shared_numbers = (
        nums1 &
        nums2
    )

    # Jaccard helpers

    name_union = (
        name_tokens |
        s2_name_tokens
    )

    addr_union = (
        addr_tokens |
        s2_addr_tokens
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    return [

        # 0
        ratio(
            name,
            s2_name
        ) / 100.0,

        # 1
        token_set_ratio(
            name,
            s2_name
        ) / 100.0,

        # 2
        ratio(
            trans_name,
            s2_trans_name
        ) / 100.0,

        # 3
        token_set_ratio(
            trans_name,
            s2_trans_name
        ) / 100.0,

        # 4
        ratio(
            address,
            s2_address
        ) / 100.0,

        # 5
        token_set_ratio(
            address,
            s2_address
        ) / 100.0,

        # 6
        len(
            shared_name
        ),

        # 7
        len(
            shared_address
        ),

        # 8
        len(
            shared_numbers
        ),

        # 9
        (
            len(shared_name) /
            max(
                1,
                len(name_union)
            )
        ),

        # 10
        (
            len(shared_address) /
            max(
                1,
                len(addr_union)
            )
        ),

        # 11
        int(
            name == s2_name
        ),

        # 12
        int(
            address == s2_address
        ),

        # 13
        int(
            s1["country"] ==
            s2["country"]
        ),

        # 14
        abs(
            len(name) -
            len(s2_name)
        ),

        # 15
        abs(
            len(address) -
            len(s2_address)
        ),

        # 16
        len(name_tokens),

        # 17
        len(s2_name_tokens),

        # 18
        len(addr_tokens),

        # 19
        len(s2_addr_tokens),

        # 20
        len(nums1),

        # 21
        len(nums2),

        # 22
        int(
            bool(shared_numbers)
        ),

    ]


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

print("Computing features...")

X = []
y = []
s1_ids_for_split = []

for row in training_rows:

    s1_id = row["s1_id"]

    s2_idx = row["s2_idx"]

    label = row["label"]

    features = pair_features(
        s1_by_id[s1_id],
        S2[s2_idx]
    )

    X.append(features)

    y.append(label)

    s1_ids_for_split.append(
        s1_id
    )


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int8
)


print(
    "Feature matrix:",
    X.shape
)

print(
    "Positive:",
    int(y.sum())
)

print(
    "Negative:",
    int((y == 0).sum())
)


# ============================================================
# SPLIT BY S1 ENTITY
# ============================================================

unique_s1 = np.array(
    list(
        set(s1_ids_for_split)
    )
)

train_s1, val_s1 = train_test_split(
    unique_s1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_s1 = set(train_s1)
val_s1 = set(val_s1)


train_mask = np.array([
    sid in train_s1
    for sid in s1_ids_for_split
])

val_mask = ~train_mask


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]


print("\nSplit:")

print(
    "Train examples:",
    len(y_train)
)

print(
    "Validation examples:",
    len(y_val)
)

print(
    "Train positives:",
    int(y_train.sum())
)

print(
    "Validation positives:",
    int(y_val.sum())
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")


scale_pos_weight = (
    (y_train == 0).sum() /
    max(
        1,
        (y_train == 1).sum()
    )
)


model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    scale_pos_weight=scale_pos_weight,
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_val,
            y_val
        )
    ],

    verbose=False
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("Predicting validation set...")

probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# F0.5
# ============================================================

def f05(
    precision,
    recall
):

    if (
        precision == 0
        and recall == 0
    ):

        return 0.0

    return (
        1.25 *
        precision *
        recall
    ) / (
        0.25 *
        precision +
        recall
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nThreshold search:")

best_threshold = None
best_score = -1

threshold_results = []


for threshold in np.arange(
    0.30,
    0.96,
    0.02
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    score = f05(
        precision,
        recall
    )

    threshold_results.append(
        (
            threshold,
            precision,
            recall,
            score
        )
    )

    if score > best_score:

        best_score = score

        best_threshold = threshold


for threshold, precision, recall, score in threshold_results:

    print(
        f"{threshold:.2f} | "
        f"P={precision:.4f} | "
        f"R={recall:.4f} | "
        f"F0.5={score:.4f}"
    )


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)


precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

score = f05(
    precision,
    recall
)


print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(
    f"Best threshold : {best_threshold:.2f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

print(
    f"F0.5           : {score:.4f}"
)

print("=" * 70)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    "student_resource/matcher_sample.json"
)

print(
    "\nModel saved to:"
)

print(
    "student_resource/matcher_sample.json"
)