import math
import re

from urllib.parse import urlparse

import tldextract

TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

HIGH_ENTROPY_THRESHOLD = 4.0
LONG_SUBDOMAIN_THRESHOLD = 3

def shannon_entropy(text):
    """Shannon entropy H(x) = -Sigma p(x) log2 p(x) of a string's characters"""
    if not text:
        return 0.0

    length = len(text)
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy

def normalize_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url

def extract_domain_features(url):
    url = normalize_url(url)
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = urlparse("http://invalid")

    hostname = parsed.hostname or ""

    try:
        result = TLD_EXTRACT(hostname)
    except Exception:
        result = tldextract.ExtractResult(subdomain="", domain="", suffix="")

    subdomain = result.subdomain or ""
    suffix = result.suffix or ""
    domain = result.domain

    # Fallback: IP addresses/ single labels / tld-only hostname
    if not domain:
        domain = hostname if not suffix else ""
        registered_domain = domain or suffix
    else:
        registered_domain = f"{domain}.{suffix}" if suffix else domain

    subdomain_labels = [label for label in subdomain.split(".") if label]
    num_subdomains = len(subdomain_labels)

    domain_num_digits = sum(char.isdigit() for char in domain)
    domain_num_hyphens = domain.count("-")
    domain_num_special_chars = len(re.findall(r"[^a-zA-Z0-9]", domain))
    domain_length = len(domain)

    entropy = shannon_entropy(domain)

    num_domain_labels = len([label for label in hostname.split(".") if label])
    is_punycode = int(any(label.startswith("xn--") for label in hostname.split(".")))
    is_numeric_tld = int(suffix.isdigit()) if suffix else 0
    has_long_subdomain = int(num_subdomains >= LONG_SUBDOMAIN_THRESHOLD)
    high_domain_entropy = int(entropy > HIGH_ENTROPY_THRESHOLD)

    features = {
        # Domain identity and structure
        "registered_domain_length": len(registered_domain),
        "domain_length": domain_length,
        "tld_length": len(suffix),
        "subdomain_length": len(subdomain),
        "num_subdomains": num_subdomains,
        "num_domain_labels": num_domain_labels,

        # Domain compostition
        "domain_num_digits": domain_num_digits,
        "domain_num_hyphens": domain_num_hyphens,
        "domain_num_special_chars": domain_num_special_chars,
        "domain_digit_ratio": domain_num_digits / domain_length if domain_length else 0.0,
        "domain_hyphens_ratio": domain_num_hyphens / domain_length if domain_length else 0.0,
        "domain_entropy": round(entropy, 4),

        #suspicion indicators
        "is_punycode": is_punycode,
        "has_long_subdomain": has_long_subdomain,
        "is_numeric_tld": is_numeric_tld,
        "high_domain_entropy": high_domain_entropy
    }

    return features

DOMAIN_FEATURE_GROUPS = {
    "identity": [
        "registered_domain_length",
        "domain_length",
        "tld_length",
        "subdomain_length",
        "num_subdomains",
        "num_domain_labels",
    ],
    "composition": [
        "domain_num_digits",
        "domain_num_hyphens",
        "domain_num_special_chars",
        "domain_digit_ratio",
        "domain_hyphen_ratio",
        "domain_entropy",
    ],
    "suspicion": [
        "is_punycode",
        "has_long_subdomain",
        "is_numeric_tld",
        "high_domain_entropy",
    ],
}

DOMAIN_FEATURE_GROUPS["all"] = list(extract_domain_features("https://example.com").keys())

def extract_domain_feature_dataframe(urls, columns=None):
    # Convert a list/Series of URLs into a DataFrame of the new domain feature
    import pandas as pd

    feature_rows = [extract_domain_features(url) for url in urls]
    df = pd.DataFrame(feature_rows)

    if columns is not None:
        df = df[list(columns)]

    return df
