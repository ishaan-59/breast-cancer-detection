from pathlib import Path
import pandas as pd


index_path = Path("data/processed/final_dataset_index.csv")

df = pd.read_csv(index_path)

print("Shape:", df.shape)

print("\nMissing values:")
print(df[["full_mammogram_path", "roi_mask_path", "binary_label"]].isna().sum())

print("\nBinary label counts by split:")
print(pd.crosstab(df["split"], df["binary_label"]))

print("\nAbnormality type by split:")
print(pd.crosstab(df["split"], df["abnormality_type"]))

print("\nView counts:")
print(df["view"].value_counts())

print("\nRows missing ROI mask:")
print(
    df[df["roi_mask_path"].isna()][
        [
            "patient_id",
            "split",
            "abnormality_type",
            "side",
            "view",
            "abnormality_id",
            "pathology",
            "source_csv",
        ]
    ]
)