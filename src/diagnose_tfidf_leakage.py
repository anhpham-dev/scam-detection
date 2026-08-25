from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from domain_features import TLD_EXTRACT
from hybrid_model import (
    RANDOM_STATE,
    TEST_SIZE,
    clean_dataset,
    load_dataset,
)
from features import normalize_url


def hostname(url):
    try:
        return (urlparse(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""


def scheme(url):
    try:
        return urlparse(url).scheme
    except ValueError:
        return ""


def registered_domain(url):
    host = hostname(url)
    if not host:
        return ""

    result = TLD_EXTRACT(host)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return result.domain or host


def class_purity(frame):
    counts = frame.groupby(["registered_domain", "type"]).size()
    domain_totals = counts.groupby(level=0).sum()
    dominant = counts.groupby(level=0).max()
    return (dominant / domain_totals).sort_values(ascending=False)


def print_scheme_diagnosis(frame):
    frame = frame.assign(scheme=frame["url"].map(scheme))
    table = pd.crosstab(frame["scheme"], frame["type"], normalize="index")
    print("\nScheme distribution by class:")
    print(table.round(4).to_string())
    print("\nScheme counts:")
    print(pd.crosstab(frame["scheme"], frame["type"]).to_string())


def print_domain_diagnosis(train, test):
    train_domains = set(train["registered_domain"])
    test_domains = set(test["registered_domain"])
    overlap = train_domains & test_domains
    print("\nDomain overlap:")
    print(f"Train domains: {len(train_domains):,}")
    print(f"Test domains:  {len(test_domains):,}")
    print(f"Overlapping domains: {len(overlap):,} ({len(overlap) / max(len(test_domains), 1):.2%} of test domains)")
    print(f"Test rows on seen domains: {test['registered_domain'].isin(overlap).mean():.2%}")

    domain_labels = train.groupby("registered_domain")["type"].agg(
        lambda labels: labels.value_counts().index[0]
    )
    known_test = test[test["registered_domain"].isin(domain_labels.index)]
    if not known_test.empty:
        predictions = known_test["registered_domain"].map(domain_labels)
        print("\nSeen-domain majority baseline:")
        print(f"Rows evaluated: {len(known_test):,}")
        print(f"Accuracy: {accuracy_score(known_test['type'], predictions):.4f}")
        print(f"Macro F1: {f1_score(known_test['type'], predictions, average='macro'):.4f}")

    purity = class_purity(train.reset_index(drop=True))
    print("\nMost class-pure training domains (at least 10 rows):")
    counts = train["registered_domain"].value_counts()
    for domain, value in purity[counts[purity.index].ge(10)].head(15).items():
        print(f"{domain:<35} purity={value:.2%} rows={counts[domain]:,}")


def print_ngram_diagnosis(train, test):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=4,
        max_df=0.95,
        lowercase=True,
    )
    vectorizer.fit(train["url"])
    train_vocabulary = set(vectorizer.vocabulary_)
    test_ngrams = set()
    for url in test["url"]:
        test_ngrams.update(
            url[index:index + size].lower()
            for size in range(3, 6)
            for index in range(len(url) - size + 1)
        )
    overlap = train_vocabulary & test_ngrams
    print("\nCharacter n-gram overlap:")
    print(f"Training vocabulary: {len(train_vocabulary):,}")
    print(f"Test n-grams represented in training vocabulary: {len(overlap):,} ({len(overlap) / max(len(test_ngrams), 1):.2%})")
    print("This overlap is expected for general URL syntax; domain overlap and class-pure n-grams are the leakage signals.")


def main():
    frame = clean_dataset(load_dataset()).copy()
    frame["registered_domain"] = frame["url"].map(registered_domain)

    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["type"],
    )

    print(f"Rows: {len(frame):,}")
    print(f"Exact normalized URL duplicates: {frame['url'].duplicated().sum():,}")
    print_scheme_diagnosis(frame)
    print_domain_diagnosis(train, test)
    print_ngram_diagnosis(train, test)


if __name__ == "__main__":
    main()