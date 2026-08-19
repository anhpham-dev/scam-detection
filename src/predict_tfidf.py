from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from features import normalize_url


# Paths
PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "url_tfidf_classifier.pkl"
)

VECTORIZER_PATH = (
    PROJECT_DIR
    / "models"
    / "url_char_tfidf.pkl"
)

# Load
print("Loading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)
print("Loading classifier...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")

# predict
def predict_url(url):

    url = normalize_url(url)

    X = vectorizer.transform([url])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    probability_dict = {
        class_name: float(probability)
        for class_name, probability
        in zip(classes, probabilities)
    }

    return prediction, probability_dict

if __name__ == "__main__":
    print("\n")

    while True:
        url = input("Enter url (type 'exit' to quit): ")

        if url.lower() == "exit":
            break

        if not url.strip():
            continue

        prediction, probabilities = predict_url(url)

        print("\nPrediction:")
        print(f"  {prediction}")
        print("\nProbabilities:")
        for class_name, probability in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name:<12} {probability:.2%}")