from pathlib import Path
import sys
import time

import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))

from hybrid_model import MODEL_DIR, RANDOM_STATE, TEST_SIZE, clean_dataset, create_model, create_vectorizer, evaluate_model, load_dataset
from features import FEATURE_GROUPS, extract_feature_dataframe
from domain_features import DOMAIN_FEATURE_GROUPS, extract_domain_feature_dataframe

V3_FEATURES = FEATURE_GROUPS["all_minus_redundant"]
V4_FEATURES = DOMAIN_FEATURE_GROUPS["all"]

EXPERIMENTS = [
    ("v4_tfidf_only", [], []), # TF-IDF baseline
    ("v4_tfidf_v3", V3_FEATURES, []), # TF-IDF + V3 handcrafted
    ("v4_tfidf_domain", [], V4_FEATURES), # TF-IDF + V4 domain
    ("v4_tfidf_v3_domain", V3_FEATURES, V4_FEATURES) # TF-IDF + V3 + V4
]

def variant_paths(variant):
    suffix = "" if variant == "default" else f"_{variant}"
    return {
        "model": MODEL_DIR / f"url_hybrid_v4_classier{suffix}.pkl",
        "vectorizer": MODEL_DIR / f"url_hybrid_v4_tfidf{suffix}.pkl",
        "scalar": MODEL_DIR / f"url_hybrid_v4_scaler{suffix}.pkl",
    }

def save_model(model, vectorizer, scaler, variant):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = variant_paths(variant)

    joblib.dump(model, paths["model"])
    joblib.dump(vectorizer, paths["vectorizer"])

    if scaler is not None:
        joblib.dump(scaler, paths["scaler"])

    print(f"\nClassifer saved to {paths['model']}")
    print(f"Vectorizer saved to {paths['vectorizer']}")
    if scaler is not None:
        print(f"Scaler saved to {paths['scaler']}")

def main():
    # Load
    df = load_dataset()
    # Clean dataset
    df = clean_dataset(df)
    print(f"\nFinal dataset size: {len(df):,}")

    urls = df["url"]
    y = df["type"]

    print("\nSplitting dataset...")
    url_train, url_test, y_train, y_test = train_test_split(urls, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Training rows: {len(url_train):,}")
    print(f"Testing rows: {len(url_test):,}")

    # TF-IDF shared across all experiments
    vectorizer = create_vectorizer()
    print("\nFitting TF-IDF")
    start_time = time.time()
    X_tfidf_train = vectorizer.fit_transform(url_train)
    X_tfidf_test = vectorizer.transform(url_test)
    print(f"TF-IDF finished in {time.time() - start_time:.2f} seconds")
    print(f"Training: {X_tfidf_train.shape}\nTesting: {X_tfidf_test.shape}\nVocab: {len(vectorizer.vocabulary_):,}")

    # V3 handcrafted features
    print("\nExtracting V3 handcrafted features...")
    start_time = time.time()
    v3_train_full = extract_feature_dataframe(url_train)
    v3_test_full = extract_feature_dataframe(url_test)
    print(f"V3 features finished in {time.time() - start_time:.2f} seconds")
    print(f"Training: {v3_train_full.shape}\nTesting: {v3_test_full.shape}")

    # V4 domain features
    print("\nExtracting V4 domain features...")
    start_time = time.time()
    v4_train_full = extract_domain_feature_dataframe(url_train)
    v4_test_full = extract_domain_feature_dataframe(url_test)
    print(f"V4 features finished in {time.time() - start_time:.2f} seconds")
    print(f"Training: {v4_train_full.shape}\nTesting: {v4_test_full.shape}")

    results = []

    for name, v3_columns, v4_columns in EXPERIMENTS:
        print(f"\n==============================\nExperiment: {name}\n==============================")
        v3_columns = [col for col in v3_columns if col not in set(v4_columns)]

        train_parts = []
        test_parts = []

        if v3_columns:
            train_parts.append(v3_train_full[list(v3_columns)])
            test_parts.append(v3_test_full[list(v3_columns)])

        if v4_columns:
            train_parts.append(v4_train_full[list(v4_columns)])
            test_parts.append(v4_test_full[list(v4_columns)])

        if not train_parts:
            print("\nUsing TF-IDF only")
            X_train = X_tfidf_train
            X_test = X_tfidf_test
            scaler= None
        else:
            combined_columns = list(v3_columns) + list(v4_columns)
            print(f"\nCombined handcrafted features ({len(combined_columns)}):")
            print(combined_columns)

            X_feature_train = pd.concat(train_parts, axis=1)
            X_feature_test = pd.concat(test_parts, axis=1)

            print("\nFitting StandardScaler...")
            scaler = StandardScaler()
            scaler.fit(X_feature_train)

            X_feature_train = scaler.transform(X_feature_train).astype("float32")
            X_feature_test = scaler.transform(X_feature_test)

            X_feature_train = csr_matrix(X_feature_train)
            X_feature_test = csr_matrix(X_feature_test)

            print("\nCombining TF-IDF handcrafted features...")
            X_train = hstack([X_tfidf_train, X_feature_train], format="csr")
            X_test = hstack([X_tfidf_test, X_feature_test], format="csr")

        print("\nTraining model...")
        model = create_model()
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f} seconds")
        print(model.classes_)

        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "experiment": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "training_time": training_time,
        })

        save_model(model, vectorizer, scaler, name)

    results_df = pd.DataFrame(results)
    print("\n==============================\nExperiment Summary\n==============================")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_path = MODEL_DIR / "experiment_results_v4.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nExperiment results saved to: {results_path}")

if __name__ == "__main__":
    main()