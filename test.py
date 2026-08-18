from pathlib import Path
import pandas as pd

dataset_dir = Path(
    r"C:\Users\Quoc Anh\.cache\kagglehub\datasets\sid321axn\malicious-urls-dataset\versions\1"
)

csv_path = dataset_dir / "malicious_phish.csv"

df = pd.read_csv(csv_path)

print(df.shape)
print(df.head())
print(df["type"].value_counts())