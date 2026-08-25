from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))

from domain_features import DOMAIN_FEATURE_GROUPS, extract_domain_feature_dataframe
from features import extract_feature_dataframe, remove_scheme
from hybrid_model import RANDOM_STATE, TEST_SIZE, clean_dataset, create_model, load_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
V5_PREFIX = "url_hybrid_v5_scheme_removed"


def create_v5_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(4, 6),
        max_features=500_000,
        min_df=4,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )


def main():
    frame = clean_dataset(load_dataset())
    urls = frame["url"]
    labels = frame["type"]
    train_urls, test_urls, y_train, y_test = train_test_split(
        urls,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_text = train_urls.map(remove_scheme)
    test_text = test_urls.map(remove_scheme)
    vectorizer = create_v5_vectorizer()
    started = time.time()
    x_tfidf_train = vectorizer.fit_transform(train_text)
    x_tfidf_test = vectorizer.transform(test_text)
    print(f"TF-IDF finished in {time.time() - started:.2f} seconds")
    print(f"Vocabulary: {len(vectorizer.vocabulary_):,}")

    domain_columns = DOMAIN_FEATURE_GROUPS["all"]
    train_features = extract_domain_feature_dataframe(train_urls)[domain_columns]
    test_features = extract_domain_feature_dataframe(test_urls)[domain_columns]
    scaler = StandardScaler().fit(train_features)
    x_features_train = csr_matrix(scaler.transform(train_features).astype("float32"))
    x_features_test = csr_matrix(scaler.transform(test_features).astype("float32"))

    x_train = hstack([x_tfidf_train, x_features_train], format="csr")
    x_test = hstack([x_tfidf_test, x_features_test], format="csr")
    model = create_model().fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, predictions, average='weighted'):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{V5_PREFIX}_classifier.pkl")
    joblib.dump(vectorizer, MODEL_DIR / f"{V5_PREFIX}_tfidf.pkl")
    joblib.dump(scaler, MODEL_DIR / f"{V5_PREFIX}_scaler.pkl")
    print(f"Saved V5 artifacts with prefix: {V5_PREFIX}")


if __name__ == "__main__":
    main()