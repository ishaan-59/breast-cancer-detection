from pathlib import Path
import pandas as pd


data_root = Path("/Volumes/T7/breast_cancer_ml")

metadata_path = data_root / "metadata" / "metadata.csv"

df = pd.read_csv(metadata_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nCollection counts:")
print(df["Collection"].value_counts())

print("\nModality counts:")
print(df["Modality"].value_counts())

print("\nStudy Description counts:")
print(df["Study Description"].value_counts(dropna=False).head(20))

print("\nSeries Description counts:")
print(df["Series Description"].value_counts(dropna=False).head(50))

print("\nNumber of Images counts:")
print(df["Number of Images"].value_counts().head(20))

print("\nFirst 10 File Locations:")
print(df["File Location"].head(10).to_string(index=False))

print("\nFirst 10 Subject IDs:")
print(df["Subject ID"].head(10).to_string(index=False))