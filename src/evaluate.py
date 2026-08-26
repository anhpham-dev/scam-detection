from pathlib import Path
import sys

import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


sys.path.append(
    str(Path(__file__).resolve().parent)
)

from features import (
    normalize_url,
    extract_feature_dataframe,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "sid321axn"
    / "malicious-urls-dataset"
    / "versions"
    / "1"
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "url_classifier.pkl"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

MAX_ROWS = 100_000


# ============================================================
# FIND CSV
# ============================================================

def find_csv():

    files = list(
        DATASET_DIR.glob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            "No CSV found."
        )

    return files[0]


# ============================================================
# LOAD
# ============================================================

def load_data():

    csv_path = find_csv()

    print(
        f"Loading {csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    df = df[
        ["url", "type"]
    ]

    df = df.dropna()

    df["url"] = (
        df["url"]
        .astype(str)
        .apply(normalize_url)
    )

    df["type"] = (
        df["type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df.drop_duplicates(
        subset=["url"]
    )

    if MAX_ROWS is not None:

        df = df.sample(
            n=MAX_ROWS,
            random_state=RANDOM_STATE,
            stratify=df["type"]
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading dataset..."
    )

    df = load_data()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "\nExtracting features..."
    )

    X = extract_feature_dataframe(
        df["url"]
    )

    y = df["type"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "\nLoading model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    predictions = model.predict(
        X_test
    )

    labels = sorted(
        y.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\nConfusion matrix:")

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels
        )
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Malicious URL Classifier - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        Path(__file__).resolve().parent.parent
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    print(
        f"\nSaved confusion matrix to:"
        f"\n{output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()