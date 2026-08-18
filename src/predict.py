from pathlib import Path
import sys

import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent)) # to import stuff

from features import extract_features

PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "url_classifier.pkl"
)

print("Loading model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded.")

def predict_url(url):
    features = extract_features(url)

    X = pd.DataFrame([features])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    probability_dict = {
        class_name: float(probability) for class_name, probability in zip(classes, probabilities)
    }

    return (prediction, probability_dict)

if __name__ == "__main__":
    print("\n")
    while True:
        url = input("Enter URL (type 'exit' to quit): ")
        if url.lower() == "exit":
            break

        if not url.strip():
            continue

        prediction, probabilities = (predict_url(url))

        print("\nProbabilities:")

        for (
            class_name,
            probability
        ) in sorted (
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {class_name:<12} {probability:.2%}")
        