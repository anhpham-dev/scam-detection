from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
import tldextract

from features import extract_feature_dataframe, normalize_url, remove_scheme
from domain_features import extract_domain_feature_dataframe


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = (
    MODEL_DIR /
    "url_hybrid_v5_scheme_removed_classifier.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIR /
    "url_hybrid_v5_scheme_removed_tfidf.pkl"
)

SCALER_PATH = (
    MODEL_DIR /
    "url_hybrid_v5_scheme_removed_scaler.pkl"
)

SUSPICIOUS_THRESHOLD = 0.60
MALICIOUS_THRESHOLD = 0.90
TRUSTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "wikipedia.org",
    "paypal.com",
    "microsoft.com",
    "apple.com",
    "github.com",
    "linkedin.com",
}
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def get_registered_domain(url):
    try:
        hostname = (urlparse(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""
    result = TLD_EXTRACT(hostname)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return result.domain or hostname


print("Loading model artifacts...")
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)


def predict_url(url: str) -> dict:

    # Same normalization used during training
    url = normalize_url(url)

    registered_domain = get_registered_domain(url)
    if registered_domain in TRUSTED_DOMAINS:
        return {
            "prediction": "benign",
            "confidence": 1.0,
            "probabilities": {"benign": 1.0},
            "phishing_probability": 0.0,
            "risk": "LOW",
            "risk_level": "LOW",
            "trusted_domain": True,
        }

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    x_tfidf = vectorizer.transform([remove_scheme(url)])

    # --------------------------------------------------------
    # V3 handcrafted features
    # --------------------------------------------------------

    v3_features = extract_feature_dataframe([url])

    # --------------------------------------------------------
    # V4 domain features
    # --------------------------------------------------------

    v4_features = extract_domain_feature_dataframe([url])

    # --------------------------------------------------------
    # Reproduce training feature selection
    # --------------------------------------------------------

    v4_columns = list(v4_features.columns)

    v3_columns = [
        col
        for col in v3_features.columns
        if col not in set(v4_columns)
    ]

    v3_features = v3_features[v3_columns]
    v4_features = v4_features[v4_columns]

    # Combine in EXACT same order as training:
    #
    # V3 features
    # +
    # V4 domain features

    features = v3_features.copy()

    for column in v4_columns:
        features[column] = v4_features[column]

    # Make sure scaler receives exactly the training columns
    if hasattr(scaler, "feature_names_in_"):
        features = features[
            list(scaler.feature_names_in_)
        ]

    x_features = csr_matrix(
        scaler.transform(features).astype("float32")
    )

    # --------------------------------------------------------
    # Combine TF-IDF + handcrafted/domain features
    # --------------------------------------------------------

    x = hstack(
        [x_tfidf, x_features],
        format="csr"
    )

    # Safety check
    if x.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch!\n"
            f"Prediction matrix: {x.shape[1]}\n"
            f"Model expects:     {model.n_features_in_}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(x)[0]

    probabilities = model.predict_proba(x)[0]

    probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(
            model.classes_,
            probabilities
        )
    }

    phishing_probability = probability_dict.get("phishing", 0.0)
    if phishing_probability >= MALICIOUS_THRESHOLD:
        risk_level = "HIGH"
    elif phishing_probability >= SUSPICIOUS_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": probability_dict,
        "phishing_probability": phishing_probability,
        "risk": risk_level,
        "risk_level": risk_level,
        "trusted_domain": False,
    }

if __name__ == "__main__":

    while True:

        url = input("Enter URL (type 'exit' to quit): ")

        if url.lower() == "exit":
            break

        if not url.strip():
            continue

        result = predict_url(url)

        print(
            f"\nPrediction: "
            f"{result['prediction']}"
        )

        print(
            f"Risk level: "
            f"{result['risk_level']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2%}"
        )

        print("\nProbabilities:")

        for class_name, probability in sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"  {class_name}: "
                f"{probability:.2%}"
            )