from pathlib import Path
import pandas as pd


data_root = Path("/Volumes/T7/breast_cancer_ml")
metadata_path = data_root / "metadata" / "metadata.csv"

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metadata_path)

# Keep only the columns we actually care about right now
inventory = df[
    [
        "Subject ID",
        "Series Description",
        "Number of Images",
        "File Location",
        "Collection",
        "Modality",
        "Series UID",
        "Study UID",
    ]
].copy()

# Rename columns to easier Python-friendly names
inventory = inventory.rename(
    columns={
        "Subject ID": "subject_id",
        "Series Description": "series_description",
        "Number of Images": "number_of_images",
        "File Location": "file_location",
        "Collection": "collection",
        "Modality": "modality",
        "Series UID": "series_uid",
        "Study UID": "study_uid",
    }
)

# Convert relative TCIA path into actual path on your SSD
inventory["series_path"] = inventory["file_location"].apply(
    lambda x: str(data_root / x.replace("./", ""))
)

# Check whether the folder actually exists on your SSD
inventory["series_path_exists"] = inventory["series_path"].apply(
    lambda x: Path(x).exists()
)

print("Inventory shape:", inventory.shape)

print("\nSeries description counts:")
print(inventory["series_description"].value_counts())

print("\nPath exists counts:")
print(inventory["series_path_exists"].value_counts())

print("\nFirst 5 rows:")
print(inventory.head())

out_path = out_dir / "metadata_inventory.csv"
inventory.to_csv(out_path, index=False)

print("\nSaved inventory to:", out_path)