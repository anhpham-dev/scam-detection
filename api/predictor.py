from pathlib import Path
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
SCR_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SCR_DIR))

from features import extract_feature_dataframe

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "url_hybrid_v4_classifier_v4_tfidf_v3.pkl"
VECTORIZER_PATH = MODEL_DIR / "url_hybrid_v4_tfidf_v4_tfidf_v3.pkl"
SCALER_PATH = MODEL_DIR / "url_hybrid_v4_scaler_v4_tfidf_v3.pkl"

print("Loading ML model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

print("ML model loaded")

def predict_url(url: str) -> dict:
    # TF-IDF
    X_tfidf = vectorizer.transform([url])

    # V3 handcrafted features
    features = extract_feature_dataframe([url])

    feature_columns = [
        "url_length",
        "hostname_length",
        "path_length",
        "query_length",
        "num_dots",
        "num_slashes",
        "num_hyphens",
        "num_underscores",
        "num_digits",
        "num_special_chars",
        "num_at",
        "num_question",
        "num_equal",
        "num_ampersand",
        "num_percent",
        "num_colons",
        "num_semicolons",
        "has_https",
        "has_ip",
        "num_subdomains",
        "num_suspicious_words",
        "has_double_slash",
        "has_port"
    ]

    X_features = features[feature_columns]

    # StandardScaler
    X_features = scaler.transform(X_features)
    X_features = csr_matrix(X_features)

    # Combine
    X = hstack([X_tfidf, X_features], format="csr")

    # Prediction
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    probability_dict = {
        class_name: float(probability) for class_name, probability in zip(model.classes_, probabilities)
    }

    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict
    }