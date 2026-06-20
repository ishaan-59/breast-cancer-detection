from pathlib import Path
import pandas as pd


label_root = Path("/Volumes/T7/breast_cancer_ml/labels")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

label_files = [
    ("mass_case_description_train_set.csv", "mass", "training"),
    ("mass_case_description_test_set.csv", "mass", "test"),
    ("calc_case_description_train_set.csv", "calcification", "training"),
    ("calc_case_description_test_set.csv", "calcification", "test"),
]

all_dfs = []

for filename, abnormality_type, split in label_files:
    csv_path = label_root / filename

    df = pd.read_csv(csv_path)

    # Normalize inconsistent column name between mass/calc files
    if "breast density" in df.columns:
        df = df.rename(columns={"breast density": "breast_density"})

    df["abnormality_type_clean"] = abnormality_type
    df["split"] = split
    df["source_csv"] = filename

    all_dfs.append(df)

labels = pd.concat(all_dfs, ignore_index=True)

# Clean useful columns
labels = labels.rename(
    columns={
        "left or right breast": "side",
        "image view": "view",
        "abnormality id": "abnormality_id",
        "abnormality type": "abnormality_type_original",
        "image file path": "image_file_path",
        "cropped image file path": "cropped_image_file_path",
        "ROI mask file path": "roi_mask_file_path",
    }
)

# Convert abnormality_id to string so it can match metadata parsed abnormality_id
labels["abnormality_id"] = labels["abnormality_id"].astype(str)

# Make binary classification label
labels["binary_label"] = labels["pathology"].replace(
    {
        "BENIGN": "benign",
        "BENIGN_WITHOUT_CALLBACK": "benign",
        "MALIGNANT": "malignant",
    }
)

print("Combined labels shape:", labels.shape)

print("\nPathology counts:")
print(labels["pathology"].value_counts())

print("\nBinary label counts:")
print(labels["binary_label"].value_counts())

print("\nSplit counts:")
print(labels["split"].value_counts())

print("\nAbnormality type counts:")
print(labels["abnormality_type_clean"].value_counts())

print("\nFirst 5 rows:")
print(
    labels[
        [
            "patient_id",
            "split",
            "abnormality_type_clean",
            "side",
            "view",
            "abnormality_id",
            "pathology",
            "binary_label",
            "image_file_path",
            "roi_mask_file_path",
        ]
    ].head()
)

out_path = out_dir / "labels_combined.csv"
labels.to_csv(out_path, index=False)

print("\nSaved combined labels to:", out_path)