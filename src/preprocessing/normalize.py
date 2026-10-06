import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Unicode-safe normalization for business names and addresses.

    Preserves:
    - Latin characters
    - Devanagari and other scripts
    - Unicode combining marks
    - Numbers

    Removes/replaces punctuation and extra whitespace.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Keep Unicode letters, marks, numbers and whitespace.
    # Replace punctuation/symbols with spaces.
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if (
            category.startswith("L")   # Letter
            or category.startswith("M")  # Mark, e.g. Devanagari vowel signs
            or category.startswith("N")  # Number
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_series(series):
    return series.fillna("").map(normalize_text)


def normalize_dataframe(df):
    df = df.copy()

    df["name_norm"] = normalize_series(
        df["business_name"]
    )

    df["address_norm"] = normalize_series(
        df["business_address"]
    )

    return df

