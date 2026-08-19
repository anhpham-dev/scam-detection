from pathlib import Path
import sys

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "url_hybrid_classifier.pkl"
)

VECTORIZER_PATH = (
    PROJECT_DIR
    / "models"
    / "url_hybrid_tfidf.pkl"
)

sys.path.append(str(Path(__file__).resolve().parent))

from features import normalize_url, extract_feature_dataframe

print("Loading TF-IDF vectorizer...")

vectorizer = joblib.load(VECTORIZER_PATH)

print("Loading classifier...")
model = joblib.load(MODEL_PATH)

print("Model loaded.")

def predict_url(url):
    url = normalize_url(url)

    # TF-IDF
    X_tfidf = vectorizer.transform([url])

    # Handcrafted features
    X_features = extract_feature_dataframe([url])
    X_features = csr_matrix(X_features.astype("float32").values)
    X = hstack([X_tfidf, X_features], format="csr")

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    probability_dict = {
        class_name: float(probability) for class_name, probability in zip(classes, probabilities)
    }

    return prediction, probability_dict

if __name__ == "__main__":
    print("\n")

    while True:
        url = input("Enter URL (type 'exit' to quit): ")

        if url.lower == "quit":
            break

        if not url.strip():
            continue

        prediction, probabilities = predict_url(url)

        print(f"\nPrediction:")
        print(prediction)
        print(f"\nProbabilities:")
        for class_name, probability in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name} {probability:.2%}")