# Scam-detection

## Leakage-resistant V4 evaluation

Run the V4.1 diagnosis with the project environment:

```powershell
.\.venv\Scripts\python.exe .\src\diagnose_tfidf_leakage.py
```

Run the grouped, unseen-domain comparison. Omit `--max-rows` for the full
dataset; the smaller sample is useful for a quick smoke test:

```powershell
.\.venv\Scripts\python.exe .\src\evaluate_v4_robust.py --max-rows 20000
```

The evaluator compares raw TF-IDF, handcrafted features, their combination,
and scheme/domain-free TF-IDF plus handcrafted features. It writes metrics to
`models/experiment_results_v4_robust.csv` and false predictions to
`models/v4_false_predictions.csv`.

