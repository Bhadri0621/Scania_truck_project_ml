import pandas as pd
import numpy as np
import joblib
from io import StringIO


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

l_object = joblib.load(
    "artifacts/model_things.joblib"
)

model = l_object["model"]
feature_columns = l_object["features"]
threshold = l_object["threshold"]
medians = l_object["medians"]
missing_columns = l_object["missing_names"]


# =========================================================
# READ SCANIA CSV
# =========================================================

def read_scaniaguard_csv(input_file):

    # ---------------------------------------------
    # Read uploaded file as raw text
    # ---------------------------------------------

    input_file.seek(0)

    raw_bytes = input_file.read()

    text = raw_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    # ---------------------------------------------
    # Find actual CSV header
    # ---------------------------------------------

    lines = text.splitlines()

    header_index = None

    for i, line in enumerate(lines):

        # The actual Scania header contains aa_000
        # and the target column "class"

        if (
            "class" in line
            and "aa_000" in line
            and "ab_000" in line
        ):

            header_index = i
            break

    # ---------------------------------------------
    # Header not found
    # ---------------------------------------------

    if header_index is None:

        raise ValueError(
            "Could not find the Scania CSV header. "
            "Make sure the file contains columns such as "
            "'class', 'aa_000', 'ab_000'."
        )

    # ---------------------------------------------
    # Keep only actual CSV data
    # ---------------------------------------------

    csv_text = "\n".join(
        lines[header_index:]
    )

    # ---------------------------------------------
    # Read clean CSV
    # ---------------------------------------------

    df = pd.read_csv(
        StringIO(csv_text)
    )

    return df


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def opener(input_file):

    # ---------------------------------------------
    # Read CSV
    # ---------------------------------------------

    df = read_scaniaguard_csv(
        input_file
    )

    # ---------------------------------------------
    # Convert "na" to NaN
    # ---------------------------------------------

    df = df.replace(
        "na",
        np.nan
    )

    # ---------------------------------------------
    # Check whether labels exist
    # ---------------------------------------------

    has_labels = "class" in df.columns

    if has_labels:

        y_true = (
            df["class"]
            .map({
                "pos": 1,
                "neg": 0
            })
        )

        x = df.drop(
            columns=["class"]
        )

    else:

        y_true = None

        x = df.copy()

    # ---------------------------------------------
    # Create missingness indicators
    # ---------------------------------------------

    for col in missing_columns:

        indicator_name = (
            col + "_is_missing"
        )

        x[indicator_name] = (
            x[col]
            .isna()
            .astype(int)
        )

    # ---------------------------------------------
    # Convert features to numeric
    # ---------------------------------------------

    x = x.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ---------------------------------------------
    # Median imputation
    # ---------------------------------------------

    for name, value in medians.items():

        x[name] = x[name].fillna(
            value
        )

    # ---------------------------------------------
    # Check required columns
    # ---------------------------------------------

    missing_features = list(
        set(feature_columns)
        - set(x.columns)
    )

    if missing_features:

        raise ValueError(
            f"Missing required features: "
            f"{missing_features[:10]}"
        )

    # ---------------------------------------------
    # Exact feature order
    # ---------------------------------------------

    x = x[
        feature_columns
    ]

    # ---------------------------------------------
    # Predict probability
    # ---------------------------------------------

    probability = (
        model
        .predict_proba(x)[:, 1]
    )

    # ---------------------------------------------
    # Cost-sensitive threshold
    # ---------------------------------------------

    prediction = (
        probability >= threshold
    ).astype(int)

    return (
        prediction,
        probability,
        y_true
    )