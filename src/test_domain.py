import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from domain_features import extract_domain_features

TEST_URLS = [
    "https://google.com",
    "https://login.accounts.example.co.uk/path",
    "http://192.168.1.1/login",
    "https://xn--80ak6aa92e.com/",
    "https://x7q9k2m8z1.example.net",
    "https://secure-login-account-verification.example.com",
    "http://example.com",
]

for url in TEST_URLS:
    features = extract_domain_features(url)
    print(f"====================\nURL: {url}")
    for key, value in features.items():
        print(f"  {key:<28} {value}")