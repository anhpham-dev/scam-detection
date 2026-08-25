import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from domain_features import TLD_EXTRACT, extract_domain_feature_dataframe
from features import FEATURE_GROUPS, extract_feature_dataframe, normalize_url
from hybrid_model import clean_dataset, load_dataset, RANDOM_STATE


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT_DIR / "models" / "experiment_results_v4_robust.csv"
ERRORS_PATH = ROOT_DIR / "models" / "v4_false_predictions.csv"


def registered_domain(url):
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except ValueError:
        hostname = ""
    if not hostname:
        return "__invalid__"
    result = TLD_EXTRACT(hostname)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return result.domain or hostname


def remove_scheme_and_domain(url):
    """Keep path/query text while preventing host and scheme memorization."""
    try:
        parsed = urlparse(normalize_url(url))
    except ValueError:
        return "/"
    value = parsed.path or "/"
    if parsed.query:
        value += f"?{parsed.query}"
    if parsed.fragment:
        value += f"#{parsed.fragment}"
    return value


def create_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=500_000,
        min_df=4,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )


def make_model():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=4.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate leakage-resistant V4 URL representations.")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    frame = clean_dataset(load_dataset()).copy()
    if args.max_rows is not None and len(frame) > args.max_rows:
        frame, _ = train_test_split(
            frame,
            train_size=args.max_rows,
            random_state=RANDOM_STATE,
            stratify=frame["type"],
        )
    frame["registered_domain"] = frame["url"].map(registered_domain)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(frame["url"], frame["type"], groups=frame["registered_domain"])
    )
    train = frame.iloc[train_indices].copy()
    test = frame.iloc[test_indices].copy()
    assert set(train["registered_domain"]).isdisjoint(test["registered_domain"])

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Train/test domain overlap: {len(set(train['registered_domain']) & set(test['registered_domain']))}")

    v3_columns = FEATURE_GROUPS["all_minus_redundant"]
    train_features = extract_feature_dataframe(train["url"])[v3_columns]
    test_features = extract_feature_dataframe(test["url"])[v3_columns]
    scaler = StandardScaler().fit(train_features)
    train_features = csr_matrix(scaler.transform(train_features).astype("float32"))
    test_features = csr_matrix(scaler.transform(test_features).astype("float32"))

    raw_vectorizer = create_vectorizer()
    raw_train = raw_vectorizer.fit_transform(train["url"])
    raw_test = raw_vectorizer.transform(test["url"])

    normalized_vectorizer = create_vectorizer()
    normalized_train_urls = train["url"].map(remove_scheme_and_domain)
    normalized_test_urls = test["url"].map(remove_scheme_and_domain)
    normalized_train = normalized_vectorizer.fit_transform(normalized_train_urls)
    normalized_test = normalized_vectorizer.transform(normalized_test_urls)

    experiments = {
        "tfidf": (raw_train, raw_test),
        "handcrafted": (train_features, test_features),
        "tfidf_handcrafted": (hstack([raw_train, train_features]), hstack([raw_test, test_features])),
        "normalized_tfidf_handcrafted": (
            hstack([normalized_train, train_features]),
            hstack([normalized_test, test_features]),
        ),
    }

    results = []
    errors = []
    for name, (x_train, x_test) in experiments.items():
        print(f"\n=== {name} ===")
        started = time.time()
        model = make_model().fit(x_train, train["type"])
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test).max(axis=1)
        print(classification_report(test["type"], predictions, digits=4))
        results.append({
            "experiment": name,
            "accuracy": accuracy_score(test["type"], predictions),
            "macro_f1": f1_score(test["type"], predictions, average="macro"),
            "weighted_f1": f1_score(test["type"], predictions, average="weighted"),
            "training_time": time.time() - started,
        })
        mismatches = test["type"].to_numpy() != predictions
        errors.append(pd.DataFrame({
            "experiment": name,
            "url": test.loc[mismatches, "url"].to_numpy(),
            "registered_domain": test.loc[mismatches, "registered_domain"].to_numpy(),
            "actual": test.loc[mismatches, "type"].to_numpy(),
            "predicted": predictions[mismatches],
            "confidence": probabilities[mismatches],
        }))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    pd.concat(errors, ignore_index=True).to_csv(ERRORS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"False predictions saved to: {ERRORS_PATH}")


if __name__ == "__main__":
    main()