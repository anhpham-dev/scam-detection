from urllib.parse import urlparse
import ipaddress
import re

SUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "password",
    "passwd",
    "confirm",
    "confirmation",
    "bank",
    "paypal",
    "wallet",
    "auth",
    "authenticate",
    "credential",
    "recover",
    "reset",
    "unlock"
]

def normalize_url(url) -> str:
    """
    Make sure the URL has a scheme so urlparse() can correctly indentify the hostname
    """

    if not isinstance(url, str):
        return "" # none

    url = url.strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url

def remove_scheme(url) -> str:
    """Remove the URL scheme while preserving host, path, query, and fragment."""
    url = normalize_url(url)
    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    value = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        value += f":{port}"
    value += parsed.path or ""
    if parsed.query:
        value += f"?{parsed.query}"
    if parsed.fragment:
        value += f"#{parsed.fragment}"
    return value

def has_ip_address(hostname):
    "Returns 1 if hostn is an IPv4 or IPv6 address. Otherwise 0"
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0

def count_digits(text):
    return sum(character.isdigit() for character in text)

def count_special_characters(text):
    return len(re.findall(r"[^a-zA-Z0-9]", text))

def extract_features(url):
    "Convert one URL into numerial features"

    url = normalize_url(url)
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = urlparse("http://invalid")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query=parsed.query or ""

    lowercase_url = url.lower()

    sus_word_count = sum(word in lowercase_url for word in SUS_WORDS)

    try:
        has_port = int(parsed.port is not None) if hostname else 0
    except ValueError:
        has_port = 0

    features = {
        # Basic length
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),

        # Char counts
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": count_digits(url),
        "num_special_chars": count_special_characters(url),

        # URL symbols
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equal": url.count("="),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "num_colons": url.count(":"),
        "num_semicolons": url.count(";"),

        # Protocol
        "has_https": int(parsed.scheme == "https"),

        # Hostname
        "has_ip": has_ip_address(hostname),

        # Number of subdomain
        "num_subdomains": max(
            0,
            len(hostname.split(".")) - 2
        ),

        # Sus words
        "has_suspicious_word": int(
            sus_word_count > 0
        ),

        "num_suspicious_words": sus_word_count,

        # Other
        "has_double_slash": int("//" in parsed.path),

        "has_encoded_characters": int("%" in url),

        "has_port": has_port,
    }

    return features

FEATURE_GROUPS = {
    "length": [
        "url_length",
        "hostname_length",
        "path_length",
        "query_length"
    ],
    "structural": [
        "num_dots", "num_slashes", "num_hyphens", "num_underscores",
        "num_digits", "num_special_chars", "num_at", "num_question",
        "num_equal", "num_ampersand", "num_percent", "num_colons",
        "num_semicolons", "has_https", "has_ip", "num_subdomains",
        "has_double_slash", "has_encoded_characters", "has_port",
    ],
    "suspicious": [
        "has_suspicious_word",
        "num_suspicious_words",
    ],
    "redundant": [
        "has_suspicious_word",
        "has_encoded_characters"
    ]
}

FEATURE_GROUPS["all"] = list(extract_features("http://x").keys())
FEATURE_GROUPS["all_minus_redundant"] = [
    col
    for col in FEATURE_GROUPS["all"]
    if col not in set(FEATURE_GROUPS["redundant"]) and col != "has_https"
]

def extract_feature_dataframe(urls, columns=None):
    "Convert a pandas Series/list of URL into a pandas DataFrame of numerical features"
    import pandas as pd

    feature_rows = [
        extract_features(url) for url in urls
    ]
    df = pd.DataFrame(feature_rows)

    if columns is not None:
        df = df[list(columns)]

    return df
    # return pd.DataFrame(feature_rows)