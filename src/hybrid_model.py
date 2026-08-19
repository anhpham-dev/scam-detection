from pathlib import Path
import sys
import time

import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix

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

from features import normalize_url, extract_feature_dataframe

# Configuration
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
    / "url_hybrid_classifier.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "url_hybrid_tfidf.pkl"
)

MAX_ROWS = None

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Find csv
def find_csv():
    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {DATASET_DIR}")

    return csv_files[0]

# Load dataset
def load_dataset():
    csv_path = find_csv()

    print("\nFinding dataset:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print(f"\nOriginal rows: {len(df):,}")

    return df

# Clean dataset
def clean_dataset(df):
    required_columns = {"url", "type"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[["url", "type"]].copy()

    df = df.dropna(subset=["url", "type"])

    df["url"] = df["url"].astype(str).apply(normalize_url)

    df["type"] = df["type"].astype(str).str.lower().str.strip()

    df = df[df["url"].str.len() > 0]

    before = len(df)

    df = df.drop_duplicates(subset = ["url"])

    print(f"\nRemoved duplicates: {before - len(df):,}")

    print("\nClass distribution:")
    print(df["type"].value_counts())

    return df

# TF-IDF
def create_vectorizer():
    print("\nCreating character TF-IDF...")

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        min_df=2,
        max_df=.95,
        sublinear_tf=True,
        lowercase=True,
        dtype="float32"
    )

# Model
def create_model():
    print("\nCreating Logistic Regression...")

    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=4.0,
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# Eval
def evaluate_model(model, X_test, y_test):
    print("\nMaking prediction...")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print("==============================\nHybrid Model Results\n==============================")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))

    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=[f"Actual: {x}" for x in labels], columns=[f"Predicted: {x}" for x in labels])

    print("\nconfusion Matrix:")
    print(matrix_df)

# Save
def save_model(model, vectorizer):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nClassifier saved to {MODEL_PATH}")
    print(f"\nVectorizer saved to {VECTORIZER_PATH}")

# Main
def main():
    # load
    df = load_dataset()
    # clean
    df = clean_dataset(df)
    print(f"\nFinal dataset size: {len(df):,}")

    # split raw url first
    urls = df["url"]
    y = df["type"]

    print("\nSplitting dataset...")

    url_train, url_test, y_train, y_test = train_test_split(urls, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Training rows: {len(url_train):,}")
    print(f"Testing rows: {len(url_test):,}")

    # TF-IDF
    vectorizer = create_vectorizer()

    print("\nFitting TF-IDF....")

    start_time = time.time()
    X_tfidf_train = vectorizer.fit_transform(url_train)
    X_tfidf_test = vectorizer.transform(url_test)
    print(f"TF-IDF finished in {time.time() - start_time:.2f} seconds")

    print("\nTF-IDF matrix:")
    print(f"Training: {X_tfidf_train.shape}")
    print(f"Testing: {X_tfidf_test.shape}")
    print(f"Vocabulary: {len(vectorizer.vocabulary_):,}")

    # Hand crafted features
    print("\nExtracting handcrafted features...")
    X_features_train = extract_feature_dataframe(url_train)
    X_features_test = extract_feature_dataframe(url_test)

    print(f"Handcrafted features: {X_features_train.shape[1]}")

    # Convert to sparse
    X_features_train = csr_matrix(X_features_train.astype("float32").values)
    X_features_test = csr_matrix(X_features_test.astype("float32").values)

    # Combine
    print("\nCombining TF-IDF + handcrafted features...")

    X_train = hstack([X_tfidf_train, X_features_train], format="csr")
    X_test = hstack([X_tfidf_test, X_features_test], format="csr")

    print(f"Hybrid training matrix: {X_train.shape}")
    print(f"Hybrid testing matrix: {X_test.shape}")

    # Train
    model = create_model()

    print("\nTraining hybrid model...")

    start_time = time.time()
    model.fit(X_train, y_train)

    print(f"\nTraining finished in {time.time() - start_time:.2f}")

    print("\nModel classes:")
    print(model.classes_)

    # Eval
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, vectorizer)

    print("\nHybrid Training completed.")

if __name__ == "__main__":
    main()