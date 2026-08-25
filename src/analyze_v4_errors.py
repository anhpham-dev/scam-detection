import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ERRORS_PATH = ROOT_DIR / "models" / "v4_false_predictions.csv"


def add_url_features(frame):
    frame = frame.copy()
    parsed = frame["url"].map(urlparse)
    frame["path_length"] = parsed.map(lambda value: len(value.path))
    frame["has_query"] = parsed.map(lambda value: int(bool(value.query)))
    frame["has_https"] = frame["url"].str.startswith("https://").astype(int)
    frame["direction"] = frame["actual"] + " -> " + frame["predicted"]
    return frame


def print_direction(frame, actual, predicted, top_n):
    errors = frame[
        (frame["actual"] == actual) & (frame["predicted"] == predicted)
    ].sort_values("confidence", ascending=False)

    print(f"\n{actual} -> {predicted}: {len(errors):,}")
    if errors.empty:
        return

    print("\nHighest-confidence examples:")
    print(
        errors[
            ["experiment", "confidence", "registered_domain", "url"]
        ].head(top_n).to_string(index=False)
    )

    print("\nMost common domains:")
    print(
        errors.groupby(["experiment", "registered_domain"])
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze V4 false positives and false negatives.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ERRORS_PATH)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"experiment", "url", "registered_domain", "actual", "predicted", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = add_url_features(frame)
    print(f"Input: {args.input}")
    print(f"Total false predictions: {len(frame):,}")
    print("\nErrors by experiment:")
    print(frame.groupby("experiment").size().to_string())

    print_direction(frame, "benign", "phishing", args.top)
    print_direction(frame, "phishing", "benign", args.top)

    print("\nPattern summary for the two requested directions:")
    selected = frame[frame["direction"].isin(["benign -> phishing", "phishing -> benign"])]
    print(
        selected.groupby(["experiment", "direction"])
        [["confidence", "path_length", "has_query", "has_https"]]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()