import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")
TARGET_FRACTIONS = np.array([0.70, 0.15, 0.15], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic, leakage-free, patient-level "
            "train/validation/test split for the combined CBIS-DDSM dataset."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("processed/combined/image_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/combined/splits"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000,
        help=(
            "Number of deterministic randomized patient assignments to evaluate. "
            "More trials can improve label balance."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing split CSV files.",
    )
    return parser.parse_args()


def require_columns(dataframe: pd.DataFrame) -> None:
    required = {
        "image_key",
        "patient_id",
        "side",
        "view",
        "malignant",
        "mass_present",
        "calc_present",
        "mixed_pathology",
        "npy_path",
        "mask_path",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Combined metadata is missing columns: {sorted(missing)}"
        )


def validate_binary_column(dataframe: pd.DataFrame, column: str) -> None:
    values = set(dataframe[column].dropna().astype(int).unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"Column {column!r} is not binary. Found values: {sorted(values)}"
        )


def prepare_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    dataframe = pd.read_csv(metadata_path)
    require_columns(dataframe)

    if dataframe.empty:
        raise ValueError("Combined metadata is empty")

    if dataframe["image_key"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["image_key"].duplicated(keep=False),
            ["image_key", "patient_id", "side", "view"],
        ]
        raise ValueError(
            "Duplicate image_key values found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    key_duplicates = dataframe.duplicated(
        ["patient_id", "side", "view"],
        keep=False,
    )

    if key_duplicates.any():
        duplicates = dataframe.loc[
            key_duplicates,
            ["image_key", "patient_id", "side", "view"],
        ]
        raise ValueError(
            "Duplicate patient/side/view keys found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    for column in [
        "malignant",
        "mass_present",
        "calc_present",
        "mixed_pathology",
    ]:
        validate_binary_column(dataframe, column)
        dataframe[column] = dataframe[column].astype(int)

    dataframe["is_cc"] = (
        dataframe["view"].astype(str).str.upper() == "CC"
    ).astype(int)

    dataframe["is_mlo"] = (
        dataframe["view"].astype(str).str.upper() == "MLO"
    ).astype(int)

    dataframe["mass_only"] = (
        (dataframe["mass_present"] == 1)
        & (dataframe["calc_present"] == 0)
    ).astype(int)

    dataframe["calc_only"] = (
        (dataframe["mass_present"] == 0)
        & (dataframe["calc_present"] == 1)
    ).astype(int)

    dataframe["mass_and_calc"] = (
        (dataframe["mass_present"] == 1)
        & (dataframe["calc_present"] == 1)
    ).astype(int)

    dataframe["benign"] = (dataframe["malignant"] == 0).astype(int)

    dataframe["malignant_mass"] = (
        (dataframe["malignant"] == 1)
        & (dataframe["mass_present"] == 1)
    ).astype(int)

    dataframe["malignant_calc"] = (
        (dataframe["malignant"] == 1)
        & (dataframe["calc_present"] == 1)
    ).astype(int)

    return dataframe


def build_patient_table(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [
        "n_images",
        "malignant",
        "benign",
        "mass_present",
        "calc_present",
        "mass_only",
        "calc_only",
        "mass_and_calc",
        "mixed_pathology",
        "malignant_mass",
        "malignant_calc",
        "is_cc",
        "is_mlo",
    ]

    patient_table = (
        dataframe.groupby("patient_id", sort=True)
        .agg(
            n_images=("image_key", "size"),
            malignant=("malignant", "sum"),
            benign=("benign", "sum"),
            mass_present=("mass_present", "sum"),
            calc_present=("calc_present", "sum"),
            mass_only=("mass_only", "sum"),
            calc_only=("calc_only", "sum"),
            mass_and_calc=("mass_and_calc", "sum"),
            mixed_pathology=("mixed_pathology", "sum"),
            malignant_mass=("malignant_mass", "sum"),
            malignant_calc=("malignant_calc", "sum"),
            is_cc=("is_cc", "sum"),
            is_mlo=("is_mlo", "sum"),
        )
        .reset_index()
    )

    if patient_table.empty:
        raise ValueError("No patients found in metadata")

    return patient_table, feature_columns


def patient_split_sizes(number_of_patients: int) -> np.ndarray:
    train_count = int(round(number_of_patients * TARGET_FRACTIONS[0]))
    val_count = int(round(number_of_patients * TARGET_FRACTIONS[1]))
    test_count = number_of_patients - train_count - val_count

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Not enough patients for a 70/15/15 split: {number_of_patients}"
        )

    return np.array(
        [train_count, val_count, test_count],
        dtype=int,
    )


def evaluate_assignment(
    patient_features: np.ndarray,
    assignment: np.ndarray,
    split_sizes: np.ndarray,
) -> float:
    total_features = patient_features.sum(axis=0).astype(np.float64)

    if np.any(total_features <= 0):
        raise ValueError(
            "At least one balancing feature has zero total count"
        )

    split_feature_totals = np.vstack(
        [
            patient_features[assignment == split_index].sum(axis=0)
            for split_index in range(3)
        ]
    ).astype(np.float64)

    feature_shares = split_feature_totals / total_features[None, :]
    feature_error = np.mean(
        (feature_shares - TARGET_FRACTIONS[:, None]) ** 2
    )

    patient_shares = split_sizes / split_sizes.sum()
    patient_error = np.mean(
        (patient_shares - TARGET_FRACTIONS) ** 2
    )

    # A small extra penalty keeps total image counts close to 70/15/15.
    image_totals = split_feature_totals[:, 0]
    image_shares = image_totals / image_totals.sum()
    image_error = np.mean(
        (image_shares - TARGET_FRACTIONS) ** 2
    )

    return float(
        feature_error
        + 2.0 * image_error
        + 0.25 * patient_error
    )


def search_best_assignment(
    patient_table: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    trials: int,
) -> tuple[np.ndarray, float]:
    if trials <= 0:
        raise ValueError("--trials must be a positive integer")

    number_of_patients = len(patient_table)
    split_sizes = patient_split_sizes(number_of_patients)

    patient_features = patient_table[
        feature_columns
    ].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)

    best_assignment = None
    best_score = float("inf")

    for trial in range(1, trials + 1):
        permutation = rng.permutation(number_of_patients)

        assignment = np.empty(number_of_patients, dtype=np.int8)

        train_end = split_sizes[0]
        val_end = train_end + split_sizes[1]

        assignment[permutation[:train_end]] = 0
        assignment[permutation[train_end:val_end]] = 1
        assignment[permutation[val_end:]] = 2

        score = evaluate_assignment(
            patient_features=patient_features,
            assignment=assignment,
            split_sizes=split_sizes,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

        if trial % 2000 == 0 or trial == trials:
            print(
                f"[{trial}/{trials}] "
                f"best balance score: {best_score:.8f}"
            )

    if best_assignment is None:
        raise RuntimeError("Could not create a patient split")

    return best_assignment, best_score


def add_split_labels(
    dataframe: pd.DataFrame,
    patient_table: pd.DataFrame,
    assignment: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_split = patient_table.copy()
    patient_split["split"] = [
        SPLIT_NAMES[int(split_index)]
        for split_index in assignment
    ]

    split_map = dict(
        zip(
            patient_split["patient_id"],
            patient_split["split"],
        )
    )

    output = dataframe.copy()
    output["split"] = output["patient_id"].map(split_map)

    if output["split"].isna().any():
        raise RuntimeError("At least one image did not receive a split")

    return output, patient_split


def validate_no_leakage(dataframe: pd.DataFrame) -> None:
    split_patient_sets = {
        split_name: set(
            dataframe.loc[
                dataframe["split"] == split_name,
                "patient_id",
            ]
        )
        for split_name in SPLIT_NAMES
    }

    train_val = split_patient_sets["train"] & split_patient_sets["val"]
    train_test = split_patient_sets["train"] & split_patient_sets["test"]
    val_test = split_patient_sets["val"] & split_patient_sets["test"]

    if train_val or train_test or val_test:
        raise RuntimeError(
            "Patient leakage detected:\n"
            f"train/val: {sorted(train_val)}\n"
            f"train/test: {sorted(train_test)}\n"
            f"val/test: {sorted(val_test)}"
        )

    patient_split_counts = (
        dataframe.groupby("patient_id")["split"].nunique()
    )

    if (patient_split_counts != 1).any():
        bad = patient_split_counts[patient_split_counts != 1]
        raise RuntimeError(
            "Some patients occur in multiple splits:\n"
            f"{bad.to_string()}"
        )


def create_summary(
    dataframe: pd.DataFrame,
    patient_split: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for split_name in SPLIT_NAMES:
        split_df = dataframe[dataframe["split"] == split_name]
        split_patients = patient_split[
            patient_split["split"] == split_name
        ]

        rows.append(
            {
                "split": split_name,
                "patients": len(split_patients),
                "images": len(split_df),
                "malignant_images": int(split_df["malignant"].sum()),
                "benign_images": int((split_df["malignant"] == 0).sum()),
                "malignant_fraction": float(split_df["malignant"].mean()),
                "mass_images": int(split_df["mass_present"].sum()),
                "calc_images": int(split_df["calc_present"].sum()),
                "mass_only_images": int(split_df["mass_only"].sum()),
                "calc_only_images": int(split_df["calc_only"].sum()),
                "mass_and_calc_images": int(
                    split_df["mass_and_calc"].sum()
                ),
                "mixed_pathology_images": int(
                    split_df["mixed_pathology"].sum()
                ),
                "cc_images": int(split_df["is_cc"].sum()),
                "mlo_images": int(split_df["is_mlo"].sum()),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    dataframe: pd.DataFrame,
    patient_split: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        output_dir / "all_with_split.csv",
        output_dir / "patient_split.csv",
        output_dir / "split_summary.csv",
        *[
            output_dir / f"{split_name}.csv"
            for split_name in SPLIT_NAMES
        ],
    ]

    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)

        raise FileExistsError(
            "Split output files already exist. "
            "Use --overwrite only when intentionally rebuilding:\n"
            f"{formatted}"
        )

    helper_columns = [
        "is_cc",
        "is_mlo",
        "mass_only",
        "calc_only",
        "mass_and_calc",
        "benign",
        "malignant_mass",
        "malignant_calc",
    ]

    export_dataframe = dataframe.drop(
        columns=[
            column
            for column in helper_columns
            if column in dataframe.columns
        ]
    )

    export_dataframe.to_csv(
        output_dir / "all_with_split.csv",
        index=False,
    )

    patient_split.to_csv(
        output_dir / "patient_split.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )

    for split_name in SPLIT_NAMES:
        split_df = export_dataframe[
            export_dataframe["split"] == split_name
        ].copy()

        split_df.to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )


def main():
    args = parse_args()

    dataframe = prepare_metadata(args.metadata)
    patient_table, feature_columns = build_patient_table(dataframe)

    assignment, best_score = search_best_assignment(
        patient_table=patient_table,
        feature_columns=feature_columns,
        seed=args.seed,
        trials=args.trials,
    )

    split_dataframe, patient_split = add_split_labels(
        dataframe=dataframe,
        patient_table=patient_table,
        assignment=assignment,
    )

    validate_no_leakage(split_dataframe)

    summary = create_summary(
        dataframe=split_dataframe,
        patient_split=patient_split,
    )

    write_outputs(
        dataframe=split_dataframe,
        patient_split=patient_split,
        summary=summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("\nSplit summary:")
    print(summary.to_string(index=False))

    print("\nPatient overlap:")
    print("train/val: 0")
    print("train/test: 0")
    print("val/test: 0")

    print(f"\nBalance score: {best_score:.8f}")
    print(f"Seed: {args.seed}")
    print(f"Trials: {args.trials}")
    print(f"Output directory: {args.output_dir}")
    print("\n✅ Global patient-level split created and validated")


if __name__ == "__main__":
    main()

