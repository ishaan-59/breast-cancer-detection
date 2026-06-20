import pandas as pd
from pathlib import Path

from pathlib import Path
import pandas as pd


def parse_subject_id(subject_id: str):
    parts = subject_id.split("_")

    first_part = parts[0]
    abnormality_type_raw, split_raw = first_part.split("-")

    if abnormality_type_raw == "Calc":
        abnormality_type = "calcification"
    elif abnormality_type_raw == "Mass":
        abnormality_type = "mass"
    else:
        abnormality_type = "unknown"

    split = split_raw.lower()

    patient_id = parts[1] + "_" + parts[2]
    side = parts[3]
    view = parts[4]

    if len(parts) > 5:
        abnormality_id = parts[5]
    else:
        abnormality_id = None

    return pd.Series(
        {
            "abnormality_type": abnormality_type,
            "split": split,
            "patient_id": patient_id,
            "side": side,
            "view": view,
            "abnormality_id": abnormality_id,
        }
    )


inventory_path = Path("data/processed/metadata_inventory.csv")
out_path = Path("data/processed/metadata_inventory_parsed.csv")

df = pd.read_csv(inventory_path)

parsed_cols = df["subject_id"].apply(parse_subject_id)

df = pd.concat([df, parsed_cols], axis=1)

print("Shape:", df.shape)

print("\nAbnormality type counts:")
print(df["abnormality_type"].value_counts())

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nSide counts:")
print(df["side"].value_counts())

print("\nView counts:")
print(df["view"].value_counts())

print("\nSeries description by abnormality type:")
print(pd.crosstab(df["abnormality_type"], df["series_description"]))

print("\nFirst 10 parsed rows:")
print(
    df[
        [
            "subject_id",
            "series_description",
            "abnormality_type",
            "split",
            "patient_id",
            "side",
            "view",
            "abnormality_id",
        ]
    ].head(10)
)

df.to_csv(out_path, index=False)

print("\nSaved parsed inventory to:", out_path)