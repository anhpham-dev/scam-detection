from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
import tldextract

from features import extract_feature_dataframe, remove_scheme
from domain_features import extract_domain_feature_dataframe
from features import normalize_url


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


print("Loading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading classifier...")
model = joblib.load(MODEL_PATH)

print("Model loaded.")

print("\n=== HAS_HTTPS COEFFICIENT ===")

if "has_https" in scaler.feature_names_in_:
    feature_names = list(scaler.feature_names_in_)
    https_index = feature_names.index("has_https")
    tfidf_size = len(vectorizer.vocabulary_)
    model_index = tfidf_size + https_index

    print("TF-IDF size:", tfidf_size)
    print("has_https feature index:", model_index)

    for class_name, coefficient in zip(
        model.classes_,
        model.coef_[:, model_index]
    ):
        print(
            f"{class_name:<12}: "
            f"{coefficient:.6f}"
        )
else:
    print("has_https is excluded from the handcrafted features.")


print("\nClassifier classes:")
print(model.classes_)

print("\nModel information:")
print("Model:", type(model).__name__)
print("TF-IDF vocabulary:", len(vectorizer.vocabulary_))
print("Model features:", model.n_features_in_)
print("Scaler features:", scaler.n_features_in_)

if hasattr(scaler, "feature_names_in_"):
    print("\nScaler feature names:")
    print(list(scaler.feature_names_in_))


def explain_tfidf(url, top_n=30):
    url = normalize_url(url)

    x = vectorizer.transform([remove_scheme(url)])

    print("\n========================================")
    print("TF-IDF EXPLANATION")
    print("========================================")
    print("URL:", url)
    print("Active features:", x.nnz)

    feature_names = vectorizer.get_feature_names_out()

    # Phishing class index
    phishing_index = list(model.classes_).index("phishing")

    # Logistic regression coefficients for phishing
    coefficients = model.coef_[phishing_index]

    # Only active TF-IDF features
    active_indices = x.indices
    active_values = x.data

    contributions = []

    for index, value in zip(active_indices, active_values):
        contribution = value * coefficients[index]

        contributions.append(
            (
                feature_names[index],
                float(value),
                float(coefficients[index]),
                float(contribution),
            )
        )

    # Largest positive contributions toward phishing
    contributions.sort(
        key=lambda x: x[3],
        reverse=True
    )

    print("\nTop features pushing toward PHISHING:")

    for feature, tfidf, coefficient, contribution in contributions[:top_n]:
        print(
            f"{feature!r:<15} "
            f"tfidf={tfidf:.6f} "
            f"coef={coefficient:+.6f} "
            f"contribution={contribution:+.6f}"
        )

    print("\nTop features pushing AWAY from PHISHING:")

    negative = sorted(
        contributions,
        key=lambda x: x[3]
    )

    for feature, tfidf, coefficient, contribution in negative[:top_n]:
        print(
            f"{feature!r:<15} "
            f"tfidf={tfidf:.6f} "
            f"coef={coefficient:+.6f} "
            f"contribution={contribution:+.6f}"
        )


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

    print("\n=== DEBUG FEATURES ===")
    print("URL:", url)

    print("\nRaw features:")
    print(features.to_dict(orient="records")[0])

    print("\nScaled features:")
    scaled = scaler.transform(features)

    for name, value in zip(
        scaler.feature_names_in_,
        scaled[0]
    ):
        print(f"{name:<30} {value: .4f}")

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

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

# if __name__ == "__main__":
#     explain_tfidf("https://google.com")
#     explain_tfidf("https://example.com")
#     explain_tfidf("https://wikipedia.org")
#     explain_tfidf("https://youtube.com")

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