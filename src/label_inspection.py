from pathlib import Path
import pandas as pd


label_root = Path("/Volumes/T7/breast_cancer_ml/labels")

csv_files = [
    path for path in label_root.glob("*.csv")
    if not path.name.startswith("._") and "case_description" in path.name
]

print("Number of label CSV files:", len(csv_files))

for csv_path in csv_files:
    print("\n" + "=" * 80)
    print("CSV:", csv_path.name)

    df = pd.read_csv(csv_path)

    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3))

    if "pathology" in df.columns:
        print("\nPathology counts:")
        print(df["pathology"].value_counts(dropna=False))

    if "patient_id" in df.columns:
        print("\nFirst 5 patient IDs:")
        print(df["patient_id"].head(5).to_string(index=False))