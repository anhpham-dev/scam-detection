from pathlib import Path
import sys
import time

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

sys.path.append(str(Path(__file__).resolve().parent)) # to import features.py

from features import (
    normalize_url,
    extract_feature_dataframe
)

# CONFIG
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
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "url_classifier.pkl"

MAX_ROWS = None # use None for all datasets (maybe)
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200 # num of trees

# Find csv
def find_csv():
    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: \{DATASET_DIR}")

    print("\nCSV file found:")
    for file in csv_files:
        print(f" - {file.name}")

    return csv_files[0]

# Load data
def load_dataset():
    csv_path = find_csv()

    print("\nLoading:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("\nOriginal dataset:")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    return df

# Clean data
def clean_dataset(df):
    required_columns = {"url", "type"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[["url", "type"]].copy()

    df = df.dropna(
        subset=["url", "type"]
    )

    df["url"] = df["url"].astype(str)
    df["type"] = df["type"].astype(str).str.lower().str.strip()

    df["url"] = df["url"].apply(
        normalize_url
    )

    df = df[df["url"].str.len()>0]

    # remove duplicates url
    before = len(df)
    df = df.drop_duplicates(
        subset=["url"]
    )

    after= len(df)

    print(f"\nRemoved duplicates: {before-after:,}")

    print("Class distribution:")
    print(df["type"].value_counts())

    return df

# Sample data
def sample_dataset(df):
    if MAX_ROWS is None:
        print(
            "\nUsing the entire dataset."
        )
        return df

    if MAX_ROWS>=len(df):
        print("\nMAX_ROWS in larger than dataset, using entire dataset")
        return df

    print(f"\nSampling {MAX_ROWS:,} rows...")

    df = df.sample(
        n=MAX_ROWS,
        random_state=RANDOM_STATE,
    )

    print("\nSampled class distribution:")
    print(df["type"].value_counts())
    print("\nSampled class percentages:")
    print(df["type"].value_counts(normalize=True).mul(100).round(2))

    return df

# Train model
def train_model(X_train, y_train):
    print("\nCreating Random Forest...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1, # all cpu core
        max_depth=None, # prevents unnecessary deep trees
        min_samples_split=2, # min samples required to split
        min_samples_leaf=1, # min samples in leaf
    )

    print("\nTraining model...")

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\nTraining finished in {elapsed:.2f} secs")

    return model

# Evaluate
def evaluate_model(
        model,
        X_test,
        y_test,
):
    print("\nMaking predictions...")

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    macro_f1 = f1_score(
        y_test, prediction, average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        prediction,
        average="weighted"
    )

    print("\n" + "="*60)
    print("Results")
    print("=====================================")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nClassification Report:")
    print(
        classification_report(
            y_test,
            prediction,
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        prediction,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual: {x}" for x in labels
        ],
        columns=[
            f"Predicted: {x}" for x in labels
        ]
    )
    print(matrix_df)
    print("=====================================")

    return prediction

# Feature importance
def show_feature_importance(model, X):
    importance = pd.DataFrame({"features": X.columns, "importance": model.feature_importances_})
    importance = importance.sort_values(
        "importance",
        ascending=False
    )
    print("\nTop Feature Importance:")
    print(importance.head(20).to_string(index=False))

# Save za model
def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def main():
    df = load_dataset()
    df = clean_dataset(df)
    df = sample_dataset(df)
    print(f"\nFinal dataset size: {len(df):,}")

    # Feature extraction

    print("\nExtracting url features...")
    start_time = time.time()

    X = extract_feature_dataframe(
        df["url"]
    )

    elapsed = time.time() - start_time

    print(f"Feature extraction finished in {elapsed:.2f} seconds")

    print(f"\nFeature matrix: {X.shape}")
    y = df["type"]

    # Train test split
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    show_feature_importance(model, X)

    # Save
    save_model(model)

    print("\nTraining completed.")

if __name__ == "__main__":
    main()