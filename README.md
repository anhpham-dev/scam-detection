# Scam Buzzer

Scam buzzer is a Chrome extension that checks the pages you visit against a small ML model running on your own machine, then it warns you when a URL looks like a scam or phishing attempt.

The whole thing runs locally! The extension uses a FastAPI server on localhost, which loads a trained classifier and returns a prediction for the URL. **NOTHING** is sent to any external service, and browsing data never leaves your computer.

## How it works

There are three pieces:

1. **Model training** (`src/`) - scripts that download and clean a public dataset of labeled URLs, then train a classifier.
2. **API** (`api/`) - a FastAPI app that loads the trained model artifacts from `models/` and exposes a single `/predict` endpoint.
3. **Extension** (`extension/`) - a Manifest V3 Chrome extension. When you open a page, it sends the URL to the local API and shows the result in the popup. If the phishing probability crosses a threshold (configurable in the options page), it injects a warning banner at the top of the page.

The classifier itself combines two views of each URL:

- Character-level TF-IDF (4-6 grams) over the URL text, with the scheme stripped so `http` vs `https` doesn't leak into the n-grams.
- Handcrafted domain features (URL length, entropy, suspicious tokens, TLD info via tldextract, etc.), standardized and appended to the TF-IDF vector.

Both go into a logistic regression trained on four classes: benign, defacement, phishing, and malware. A short list of well-known domains (see `data/trusted_domains.csv`) skips the model entirely and is always treated as safe.

## Results

I want to be honest about these numbers, because the first versions of this project looked much better than they actually were.

The dataset has many URLs per domain, so a random train/test split puts near-identical domains in both sets and inflates accuracy. My early models scored around 0.97 that way. When I re-ran the evaluation with domains grouped so test domains are never seen during training, the real performance was:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Char TF-IDF only | 0.900 | 0.818 |
| Handcrafted features only | 0.669 | 0.632 |
| TF-IDF + handcrafted (current API setup) | 0.897 | 0.812 |

Full comparison is in `models/experiment_results_v4_robust.csv`, produced by `src/evaluate_v4_robust.py`. If you normalize URLs before vectorizing, scores drop further (0.810 accuracy), which tells you how much the models still lean on memorizing exact character sequences rather than generalizing. That's a known limitation, not something I've fully solved here.

The API maps the phishing probability to risk levels: >= 0.90 is HIGH, >= 0.60 is MEDIUM, otherwise LOW. Those cutoffs came from sweeping thresholds against false positives/negatives (`src/evaluate_phishing_thresholds.py`) - they trade recall for precision deliberately, since a browser extension crying wolf gets ignored.

## Setup

You'll need Python 3.10+.

```powershell
git clone https://github.com/anhpham-dev/scam-detection.git
cd scam-detection
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Get the dataset (only needed for training)

Training uses the [Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) from Kaggle (~650k URLs). The scripts read it from the kagglehub cache, so the easiest way is:

```python
import kagglehub
kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
```

Run that once with any Python that has `kagglehub` installed. If you'd rather not use kagglehub, drop the CSV anywhere under `~/.cache/kagglehub/datasets/sid321axn/malicious-urls-dataset/versions/1/` and the loaders will find it with a glob.

### Train

```powershell
.\.venv\Scripts\python.exe .\src\train_v5.py
```

This writes three files into `models/`: the classifier, the fitted TF-IDF vectorizer, and the feature scaler. Training takes a few minutes; the TF-IDF step is the fast part.

### Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

Quick sanity check:

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"url\": \"http://secure-paypal-verify.example.com/login\"}"
```

### Install the extension

1. Open `chrome://extensions`
2. Enable Developer mode (top right)
3. Click "Load unpacked" and select the `extension/` folder
4. Open a site, click the extension icon, and check that the popup says the API is online

The extension defaults to `http://localhost:8000`. You can change the port in the options page, as long as it stays on localhost or 127.0.0.1 - the manifest only grants those hosts.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| POST | `/predict` | Analyze a URL, body: `{"url": "..."}` |
| GET | `/` | Service info |

A `/predict` response looks like this:

```json
{
    "prediction": "phishing",
    "confidence": 0.94,
    "probabilities": {
        "benign": 0.03,
        "defacement": 0.01,
        "malware": 0.02,
        "phishing": 0.94
    },
    "phishing_probability": 0.94,
    "risk": "HIGH",
    "risk_level": "HIGH",
    "trusted_domain": false
}
```

## Project layout

```
api/          FastAPI app (main.py, predictor.py, schemas.py)
extension/    Chrome extension (popup, content script banner, options page)
src/          Training, evaluation, and diagnostic scripts
models/       Trained artifacts (*.pkl) and experiment results (*.csv)
data/         Trusted domains list used by the API
```

The numbered "versions" in `models/` and the older scripts in `src/` document how the approach evolved - V1 was plain TF-IDF, later versions added handcrafted features, then removed leaky ones like the scheme itself. `train_v5.py` plus `api/predictor.py` are what's actually running; most other files are experiments I kept because they're useful references.

## Caveats

- This is a learning project, not a security product. It misses things (especially brand-new phishing domains, per the grouped evaluation above) and will occasionally flag legit sites.
- The model only sees the URL string. It can't inspect page content, redirects, or anything behind a login.
- Only run the API on localhost. It has no authentication whatsoever.
