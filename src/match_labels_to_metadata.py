from pathlib import Path
import pandas as pd


def normalize_abnormality_id(value):
    """
    Converts abnormality IDs into clean strings.

    Examples:
    1      -> "1"
    1.0    -> "1"
    "1.0"  -> "1"
    NaN    -> None
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


labels_path = Path("data/processed/labels_combined.csv")
metadata_path = Path("data/processed/metadata_inventory_parsed.csv")
out_path = Path("data/processed/final_dataset_index.csv")

labels = pd.read_csv(labels_path)
metadata = pd.read_csv(metadata_path)

# Rename label abnormality column to match metadata
labels = labels.rename(columns={"abnormality_type_clean": "abnormality_type"})

# Normalize abnormality_id in both tables
labels["abnormality_id"] = labels["abnormality_id"].apply(normalize_abnormality_id)
metadata["abnormality_id"] = metadata["abnormality_id"].apply(normalize_abnormality_id)

full_images = metadata[
    metadata["series_description"] == "full mammogram images"
].copy()

roi_masks = metadata[
    metadata["series_description"] == "ROI mask images"
].copy()

cropped_images = metadata[
    metadata["series_description"] == "cropped images"
].copy()

full_keys = [
    "patient_id",
    "split",
    "abnormality_type",
    "side",
    "view",
]

lesion_keys = [
    "patient_id",
    "split",
    "abnormality_type",
    "side",
    "view",
    "abnormality_id",
]

label_cols = [
    "patient_id",
    "split",
    "abnormality_type",
    "side",
    "view",
    "abnormality_id",
    "pathology",
    "binary_label",
    "assessment",
    "subtlety",
    "source_csv",
]

labels_small = labels[label_cols].copy()

full_small = full_images[
    full_keys + ["series_path", "subject_id", "series_uid"]
].copy()

full_small = full_small.rename(
    columns={
        "series_path": "full_mammogram_path",
        "subject_id": "full_subject_id",
        "series_uid": "full_series_uid",
    }
)

roi_small = roi_masks[
    lesion_keys + ["series_path", "subject_id", "series_uid", "number_of_images"]
].copy()

roi_small = roi_small.rename(
    columns={
        "series_path": "roi_mask_path",
        "subject_id": "roi_subject_id",
        "series_uid": "roi_series_uid",
        "number_of_images": "roi_number_of_images",
    }
)

crop_small = cropped_images[
    lesion_keys + ["series_path", "subject_id", "series_uid"]
].copy()

crop_small = crop_small.rename(
    columns={
        "series_path": "cropped_image_path",
        "subject_id": "crop_subject_id",
        "series_uid": "crop_series_uid",
    }
)

merged = labels_small.merge(
    full_small,
    on=full_keys,
    how="left",
)

merged = merged.merge(
    roi_small,
    on=lesion_keys,
    how="left",
)

merged = merged.merge(
    crop_small,
    on=lesion_keys,
    how="left",
)

print("Final dataset index shape:", merged.shape)

print("\nMissing full mammogram paths:")
print(merged["full_mammogram_path"].isna().sum())

print("\nMissing ROI mask paths:")
print(merged["roi_mask_path"].isna().sum())

print("\nMissing cropped image paths:")
print(merged["cropped_image_path"].isna().sum())

print("\nBinary label counts:")
print(merged["binary_label"].value_counts())

print("\nSplit counts:")
print(merged["split"].value_counts())

print("\nAbnormality type counts:")
print(merged["abnormality_type"].value_counts())

print("\nFirst 5 rows:")
print(
    merged[
        [
            "patient_id",
            "split",
            "abnormality_type",
            "side",
            "view",
            "abnormality_id",
            "pathology",
            "binary_label",
            "full_mammogram_path",
            "roi_mask_path",
            "cropped_image_path",
        ]
    ].head()
)

merged.to_csv(out_path, index=False)

print("\nSaved final dataset index to:", out_path)