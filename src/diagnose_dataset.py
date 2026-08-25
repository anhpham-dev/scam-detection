from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

# Config
DATASET_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "sid321axn"
    / "malicious-urls-dataset"
    / "versions"
    / "1"
)

TOP_N = 30

WATCH_DOMAINS = [
    "google.com",
    "docs.google.com",
    "forms.google.com",
    "microsoft.com",
    "office.com",
    "dropbox.com",
    "github.com",
    "discord.com",
    "telegram.org"
]

# Dataset
def find_csv():
    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in \n{DATASET_DIR}"
        )

    print("\nCSV found:")
    for file in csv_files:
        print(f"  {file}")

    return csv_files[0]

def normalize_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url

def load_and_clean_dataset():
    csv_path = find_csv()

    print("\nLoading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Original rows: {len(df):,}")

    required_columns = {"url", "type"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[["url", "type"]].copy()

    df = df.dropna(subset=["url", "type"])

    df["url"] = (
        df["url"].astype(str).apply(normalize_url)
    )

    df["type"] = (
        df["type"].astype(str).str.lower().str.strip()
    )

    df = df[df["url"].str.len() > 0]

    before = len(df)

    df = df.drop_duplicates(subset=["url"])

    print(f"Duplicates removed: {before - len(df):,}")
    print(f"Final rows: {len(df):,}")

    return df

# Domain extraction
def extract_hostname(url):
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except:
        return ""

def get_registered_domain(hostname):
    """
    Lightweight registered-domain extraction.

    For common domain suchas:
        docs.google.com -> google.com
        forms.google.com -> google.com
    
        This intentionally avoids requiring tldextract
    """

    if not hostname:
        return ""

    parts = hostname.split(".")

    if len(parts) < 2:
        return hostname

    # Common multi-part public suffixes
    common_two_part_suffixes = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.nz",
        "com.br",
        "com.cn",
        "com.sg",
        "co.jp",
        "co.kr",
    }

    suffix = ".".join(parts[-2:])

    if suffix in common_two_part_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])

# Class distribution
def print_class_distribution(df):
    print("\n===================================Class Distribution===================================")

    counts = df["type"].value_counts()

    for label, count in counts.items():
        percentage = count / len(df) * 100

        print(f"{label:<15} {count:>10,} {percentage:6.2f}%")

# Watch domain analysis
def analyze_watch_domains(df):
    print("\n===================================Watched Domain===================================")

    df = df.copy()

    df["hostname"] = df["url"].apply(extract_hostname)
    df["registered_domain"] = df["hostname"].apply(get_registered_domain)

    for domain in WATCH_DOMAINS:

        matches = df[
            (df["hostname"] == domain)
            | df["hostname"].str.endswith("." + domain)
        ]

        print(f"\n=================================={domain}==================================")

        if matches.empty:
            print("  No sample found.")
            continue

        print(f"  Total samples: {len(matches):,}")

        counts = matches["type"].value_counts()

        for label, count in counts.items():
            percentage = count / len(matches) * 100

            print(f"  {label:<15} {count:>8,} ({percentage:6.2f}%)")

            print("\n  Example URLs:")

            for url in matches["url"].head(5):
                print(f"   {url}")

def analyze_top_domains(df):
    print(f"\n==================================Top Registered Domains by Class==================================")

    df = df.copy()

    df["hostname"] = df["url"].apply(extract_hostname)

    df["registered_domain"] = df["hostname"].apply(get_registered_domain)

    for label in sorted(df["type"].unique()):
        class_df = df[df["type"] == label]

        counts = ( class_df["registered_domain"].value_counts().head(TOP_N) )

        print(f"\n[{label.upper()}]")
        print("==================================")

        for domain, count in counts.items():
            if not domain:
                continue

            print(f"{domain:<40} {count:>8,}")

# Domain clas bias
def analyze_domain_bias(df):
    print(f"\n==================================Domain Class Bias==================================")

    df = df.copy()

    df["hostname"] = df["url"].apply(extract_hostname)

    df["registered_domain"] = df["hostname"].apply(get_registered_domain)

    grouped = (df.groupby(["registered_domain", "type"]).size().unstack(fill_value=0))

    if grouped.empty:
        print("No domain data available.")
        return

    class_columns = list(df["type"].unique())

    # Add missing class columns
    for column in class_columns:
        if column not in grouped.columns:
            grouped[column] = 0

    grouped["total"] = grouped[class_columns].sum(axis=0)

    # Only examine domains with at least 10 samples
    grouped = grouped[grouped["total"] >= 10]

    # Calculate dominant class
    grouped["dominant_class"] = grouped[class_columns].idxmax(axis=1)

    grouped["dominant_ratio"] = (grouped[class_columns].max(axis=1) / grouped["total"])

    # Most strongly class-associated domains
    suspicious = (grouped.sort_values(["dominant_ratio", "total"], ascending=[False, False]).head(TOP_N))

    print(
        "\nDomains strongly associated with one class "
        "(minimum 10 samples):"
    )

    print()

    for domain, row in suspicious.iterrows():
        print(
            f"{domain:<40} "
            f"{row['dominant_class']:<12} "
            f"{row['dominant_ratio'] * 100:6.2f}% "
            f"({int(row['total']):,} samples)"
        )

        class_info = []

        for label in class_columns:
            count = int(row[label])

            if count > 0:
                class_info.append(f"{label}={count:,}")

        print(
            " " * 4
            + ", ".join(class_info)
        )

# Google-specific analysis
def analyze_google(df):
    print(f"\n==================================Domain Class Bias==================================")

    df = df.copy()

    df["hostname"] = df["url"].apply(extract_hostname)

    google_mask = (
        df["hostname"] == "google.com"
    ) | (
        df["hostname"].str.endswith(".google.com")
    )

    google_df = df[google_mask]

    print(
        f"\nTotal google.com / *.google.com samples: "
        f"{len(google_df):,}"
    )

    if google_df.empty:
        print("No Google samples found.")
        return

    print("\nClass distribution:")

    counts = google_df["type"].value_counts()

    for label, count in counts.items():
        percentage = count / len(google_df) * 100

        print(
            f"  {label:<15} "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # Look specifically in docs and forms
    for name, pattern in [
        ("docs.google.com", "docs.google.com"),
        ("forms.google.com", "forms.google.com"),
    ]:

        subset = google_df[
            google_df["hostname"] == pattern
        ]

        print(f"\n{name}: {len(subset):,} samples")

        if not subset.empty:

            counts = subset["type"].value_counts()

            for label, count in counts.items():

                percentage = count / len(subset) * 100

                print(
                    f"  {label:<15} "
                    f"{count:>8,} "
                    f"({percentage:6.2f}%)"
                )

    print("\nExample Google URLs:")

    for url in google_df["url"].head(20):
        print(f"  {url}")

def main():
    df = load_and_clean_dataset()
    print_class_distribution(df)
    analyze_google(df)
    analyze_watch_domains(df)
    analyze_top_domains(df)
    analyze_domain_bias(df)

if __name__ == "__main__":
    main()