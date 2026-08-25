from pathlib import Path
import sys

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from domain_features import extract_domain_feature_dataframe
from features import extract_feature_dataframe
from hybrid_model import TEST_SIZE, RANDOM_STATE, clean_dataset, load_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
VARIANT = "v4_tfidf_domain"
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def load_artifacts():
    return (
        joblib.load(MODEL_DIR / f"url_hybrid_v4_classifier_{VARIANT}.pkl"),
        joblib.load(MODEL_DIR / f"url_hybrid_v4_tfidf_{VARIANT}.pkl"),
        joblib.load(MODEL_DIR / f"url_hybrid_v4_scaler_{VARIANT}.pkl"),
    )


def build_features(urls, vectorizer, scaler):
    tfidf = vectorizer.transform(urls)
    handcrafted = extract_feature_dataframe(urls)
    domain = extract_domain_feature_dataframe(urls)
    features = pd.concat([handcrafted, domain], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features = features[list(scaler.feature_names_in_)]
    scaled = csr_matrix(scaler.transform(features).astype("float32"))
    return hstack([tfidf, scaled], format="csr")


def main():
    model, vectorizer, scaler = load_artifacts()
    frame = clean_dataset(load_dataset())
    _, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["type"],
    )

    x_test = build_features(test["url"], vectorizer, scaler)
    probabilities = model.predict_proba(x_test)
    phishing_index = list(model.classes_).index("phishing")
    phishing_probability = probabilities[:, phishing_index]
    actual_phishing = test["type"].to_numpy() == "phishing"

    rows = []
    print(f"Model variant: {VARIANT}")
    print(f"Validation rows: {len(test):,}")
    print("\nThreshold metrics (positive class: phishing):")
    for threshold in THRESHOLDS:
        predicted_phishing = phishing_probability >= threshold
        tn, fp, fn, tp = confusion_matrix(
            actual_phishing,
            predicted_phishing,
            labels=[False, True],
        ).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_phishing,
            predicted_phishing,
            average="binary",
            zero_division=0,
        )
        rows.append({
            "threshold": threshold,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    output_path = MODEL_DIR / "phishing_threshold_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved threshold results to: {output_path}")


if __name__ == "__main__":
    main()