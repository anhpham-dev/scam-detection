from pathlib import Path
import sys
import time

import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

sys.path.append(str(Path(__file__).resolve().parent))

from features import normalize_url, extract_feature_dataframe, FEATURE_GROUPS
import shutil

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

EXPERIMENTS = [
    ("tfidf_only", []),
    ("tfidf_length", FEATURE_GROUPS["length"]),
    ("tfidf_structural", FEATURE_GROUPS["structural"]),
    ("tfidf_suspicious", FEATURE_GROUPS["suspicious"]),
    ("tfidf_all", FEATURE_GROUPS["all"]),
    ("tfidf_all_minus_redundant", FEATURE_GROUPS["all_minus_redundant"]),
]

PROMOTE_BEST = True

MAX_ROWS = None

TEST_SIZE = 0.20
RANDOM_STATE = 42

def variant_paths(variant):
    suffix = "" if variant == "default" else f"_{variant}"
    return {
        "model": MODEL_DIR / f"url_hybrid_classifier{suffix}.pkl", 
        "vectorizer": MODEL_DIR / f"url_hybrid_tfidf{suffix}.pkl",
        "scaler": MODEL_DIR / f"url_hybrid_scaler{suffix}.pkl"
    }

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
        random_state=RANDOM_STATE
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

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

# Save
def save_model(model, vectorizer, scaler, variant):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = variant_paths(variant)

    joblib.dump(model, paths["model"])
    joblib.dump(vectorizer, paths["vectorizer"])

    if scaler is not None:
        joblib.dump(scaler, paths["scaler"])

    print(f"\nClassifier saved to {paths['model']}")
    print(f"\nVectorizer saved to {paths['vectorizer']}")
    if scaler is not None:
        print(f"\nScaler saved to {paths['scaler']}")

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

    # Run experiments
    results = []

    for name, feature_columns in EXPERIMENTS:
        print(f"==============================\nExperiment: {name}\n==============================")
        scaler = None

        if not feature_columns:
            print("\nUsing TF-IDF only")

            X_train = X_tfidf_train
            X_test = X_tfidf_test

        else:
            print("\nFeatures:")
            print(feature_columns)

            print("\nExtracting training features...")
            X_feature_train = extract_feature_dataframe(url_train, columns=feature_columns)

            print("Extracting test features...")
            X_feature_test = extract_feature_dataframe(url_test, columns=feature_columns)

            print(f"\nHandcrafted feature count: {X_feature_train.shape[1]}")

            # Handcrafted features
            print("\nFitting StandardScaler...")
            scaler = StandardScaler()

            # Fit on DataFrame (not .values) to preserve feature_names_in_
            scaler.fit(X_feature_train)

            X_feature_train = scaler.transform(X_feature_train).astype("float32")
            X_feature_test = scaler.transform(X_feature_test).astype("float32")

            # Convert to sparse
            X_feature_train = csr_matrix(X_feature_train)
            X_feature_test = csr_matrix(X_feature_test)

            # Combine TF-IDF + Handcrafted features
            print("\nCombining TF-IDF + handcrafted features...")
            X_train = hstack(
                [
                    X_tfidf_train,
                    X_feature_train
                ],
                format="csr"
            )

            X_test = hstack([
                X_tfidf_test,
                X_feature_test
            ], format="csr")

        # Train
        print("\nTraining model...")

        model = create_model()
        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        print(f"Training finished in {training_time:.2f} seconds")
        print("\nModel classes:")
        print(model.classes_)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "experiment": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "training_time": training_time
        })

        # Save
        save_model(model, vectorizer, scaler, name)

    # Experiment summary
    results_df = pd.DataFrame(results)
    print("==============================\nExperiment Summary\n==============================")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save experiment results
    results_path = MODEL_DIR / "experiment_results.csv"

    results_df.to_csv(results_path, index=False)

    print(f"\nExperiment results saved to: {results_path}")

    # Find best experiment
    best_index = results_df["macro_f1"].idxmax()
    best_result = results_df.loc[best_index]

    print("==============================\nBest Experiment\n==============================")
    print(f"Experiment: {best_result['experiment']}")
    print(f"Accuracy: {best_result['accuracy']}")
    print(f"Macro F1: {best_result['macro_f1']}")
    print(f"Weighted F1: {best_result['weighted_f1']}")

    # Promote best hybrid variant to canonical paths
    if PROMOTE_BEST:
        hybrids = results_df[results_df["experiment"] != "tfidf_only"]

        if not hybrids.empty:
            best_hybrid = hybrids.loc[hybrids["macro_f1"].idxmax(), "experiment"]

            print("==============================\nPromoting Best Hybrid\n==============================")
            print(f"Variant: {best_hybrid}")

            for key in ("model", "vectorizer", "scaler"):
                src = variant_paths(best_hybrid)[key]
                dst = variant_paths("default")[key]

                if src.exists():
                    shutil.copyfile(src, dst)
                    print(f"Promoted {key} -> {dst}")
                else:
                    print(f"Skipped {key}: source not found {src}")

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()