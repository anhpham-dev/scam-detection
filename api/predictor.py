from pathlib import Path
import sys
from urllib.parse import urlparse

import joblib
import tldextract
from scipy.sparse import csr_matrix, hstack

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from domain_features import extract_domain_feature_dataframe
from features import normalize_url, remove_scheme


MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "url_hybrid_v5_scheme_removed_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "url_hybrid_v5_scheme_removed_tfidf.pkl"
SCALER_PATH = MODEL_DIR / "url_hybrid_v5_scheme_removed_scaler.pkl"

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
    "amazon.com",
    "facebook.com",
    "reddit.com",
    "stackoverflow.com",
    "stackexchange.com",
    "cnn.com",
    "bbc.com",
    "nytimes.com",
    "dropbox.com",
    "discord.com",
    "zoom.us",
    "openai.com",
    "example.com",
}
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

print("Loading V5 ML model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)
print("V5 ML model loaded")


def get_registered_domain(url: str) -> str:
    try:
        hostname = (urlparse(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""
    result = TLD_EXTRACT(hostname)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return result.domain or hostname


def predict_url(url: str) -> dict:
    url = normalize_url(url)
    registered_domain = get_registered_domain(url)

    if registered_domain in TRUSTED_DOMAINS:
        probabilities = {
            class_name: float(class_name == "benign")
            for class_name in model.classes_
        }
        return {
            "prediction": "benign",
            "confidence": 1.0,
            "probabilities": probabilities,
            "phishing_probability": 0.0,
            "risk": "LOW",
            "risk_level": "LOW",
            "trusted_domain": True,
        }

    x_tfidf = vectorizer.transform([remove_scheme(url)])
    features = extract_domain_feature_dataframe([url])
    features = features[list(scaler.feature_names_in_)]
    x_features = csr_matrix(scaler.transform(features).astype("float32"))
    x = hstack([x_tfidf, x_features], format="csr")

    if x.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: API produced {x.shape[1]}, "
            f"model expects {model.n_features_in_}"
        )

    prediction = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(model.classes_, probabilities)
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
