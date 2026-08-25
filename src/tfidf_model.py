from pathlib import Path
import sys
import time
import numpy as np

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

sys.path.append(str(Path(__file__).resolve().parent))

from features import normalize_url

# Config
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

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "url_tfidf_classifier.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "url_char_tfidf.pkl"
)

MAX_ROWS = None

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Find CSV
def find_csv():
    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {DATASET_DIR}")

    print("\nCSV file found:")

    for file in csv_files:
        print(f"- {file.name}")

    return csv_files[0]

# Load dataset
def load_dataset():
    csv_path = find_csv()

    print("\nLoading dataset:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("\nOriginal dataset:")
    print(f"Rows: {len(df):,}")

    print(f"Columns: {list(df.columns)}")

    return df

# Clean dataset
def clean_dataset(df):
    required_columns = {"url", "type"}

    missing = (
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[["url", "type"]].copy()

    df = df.dropna(
        subset=[
            "url",
            "type"
        ]
    )

    df["url"] = (
        df["url"].astype(str).apply(normalize_url)
    )

    df["type"] = (
        df["type"].astype(str).str.lower().str.strip()
    )

    df = df[df["url"].str.len() > 0] # invalid empty URLS

    before = len(df)

    df = df.drop_duplicates(subset=["url"])

    after = len(df)

    print(f"\nRemoved duplicates: {before - after:,}")

    print("\nClass distribution:")

    print(df["type"].value_counts())

    return df

# Sample data
def sample_dataset(df):
    if MAX_ROWS is None:
        print("\nUsing entire dataset.")
        return df

    if MAX_ROWS >= len(df):
        print("\nMAX_ROWS is larger than dataset.")
        return df

    print(f"\nSampling {MAX_ROWS:,} rows...")

    df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE, stratify=df["type"])

    print("\nSampled class distribution:")
    print(df["type".value_counts()])

    return df

# Create the character tf-idf
def create_vectorizer():
    print("\nCreating character-level TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5), # char sequence
        max_features=500_000,
        min_df=4, #ignore extremely rare one
        max_df=.95, # remove extremely common patterns
        sublinear_tf=True, # limit mem usage
        lowercase=True,
        dtype=np.float32
        )
    return vectorizer

# Train
def train_model(X_train, y_train):
    print("\nCreating Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=4.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        # n_jobs=-1
    )

    print("\nTraining model...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    elapsed = time.time() - start_time

    print(f"\nTraining finished in {elapsed:.2f} seconds")

    return model

# Evaluate
def evaluate_model(model, X_test, y_test):
    print("\nMaking predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(y_test, predictions, average="macro")

    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print("==============================\nCharacter TF-IDF Results\n==============================")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))

    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {x}" for x in labels],
        columns=[f"Predicted: {x}" for x in labels]
    )

    print(f"\nConfusion Matrix:")

    print(matrix_df)

    return predictions

# save model
def save_models(
        model,
        vectorizer
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print(f"\nClassifier saved to {MODEL_PATH}")
    print(f"\nVectorizer saved to {VECTORIZER_PATH}")

# Main
def main():
    # Load
    df = load_dataset()
    # Clean
    df = clean_dataset(df)
    # Sample
    df = sample_dataset(df)

    print(f"Final dataset size: {len(df):,}")

    print("\nSplitting dataset...")

    urls = df["url"]
    y = df["type"]
    url_train, url_test, y_train, y_test = train_test_split(urls, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Training rows: {len(url_train):,}")
    print(f"Testing rows: {len(url_test):,}")

    # TF-IDF
    vectorizer = create_vectorizer()
    print("\nFitting TF-IDF...")

    start_time = time.time()

    X_train = vectorizer.fit_transform(url_train)
    X_test = vectorizer.transform(url_test)

    elapsed = time.time() - start_time

    print(f"TF-IDF extraction finished in {elapsed:.2f} seconds")

    print("\nTF-IDF matrix:")

    print(f"Training: {X_train.shape}")
    print(f"Testing: {X_test.shape}")

    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Train
    model = train_model(X_train, y_train)

    print("\nModel classes:")
    print(model.classes_)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # save
    save_models(model, vectorizer)

    print(f"\nCharacter TF-IDF training completed.")

if __name__ == "__main__":
    main()
