import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


KEY_COLUMNS = ["patient_id", "side", "view"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge corrected CBIS-DDSM mass and calcification datasets "
            "into one unique-mammogram dataset."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="combined",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the combined output directory.",
    )
    return parser.parse_args()


def aggregate_pathology(values: list[str]) -> str:
    pathologies = {
        str(value).upper().strip()
        for value in values
        if pd.notna(value)
    }

    if "MALIGNANT" in pathologies:
        return "MALIGNANT"

    if "BENIGN" in pathologies:
        return "BENIGN"

    if "BENIGN_WITHOUT_CALLBACK" in pathologies:
        return "BENIGN_WITHOUT_CALLBACK"

    raise ValueError(f"Unexpected pathology values: {sorted(pathologies)}")


def split_pipe_values(values) -> list[str]:
    result = []

    for value in values:
        if pd.isna(value):
            continue

        for item in str(value).split("|"):
            item = item.strip()
            if item:
                result.append(item)

    return result


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def load_metadata(root: Path, dataset_name: str) -> pd.DataFrame:
    path = root / dataset_name / "image_metadata.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    dataframe = pd.read_csv(path)

    required = {
        "patient_id",
        "side",
        "view",
        "pathology",
        "malignant",
        "mass_present",
        "calc_present",
        "lesion_count",
        "npy_path",
        "mask_path",
        "height",
        "width",
        "crop_rmin",
        "crop_rmax",
        "crop_cmin",
        "crop_cmax",
        "source_image_dicom",
        "source_mask_dicoms",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} metadata is missing columns: {sorted(missing)}"
        )

    duplicated = dataframe.duplicated(KEY_COLUMNS, keep=False)

    if duplicated.any():
        rows = dataframe.loc[duplicated, KEY_COLUMNS]
        raise ValueError(
            f"Duplicate keys inside {dataset_name} metadata:\n"
            f"{rows.to_string(index=False)}"
        )

    dataframe["dataset_name"] = dataset_name
    return dataframe


def validate_source_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing NPY: {image_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    image = np.load(image_path, mmap_mode="r")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Unreadable mask: {mask_path}")

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image.shape} vs {mask.shape}"
        )

    if np.count_nonzero(mask) == 0:
        raise ValueError(f"Empty mask: {mask_path}")

    return image, mask


def main():
    args = parse_args()

    processed_root = args.processed_root
    output_root = processed_root / args.output_name

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. "
                "Use --overwrite only when you intend to rebuild it."
            )

        shutil.rmtree(output_root)

    output_npy = output_root / "npy"
    output_masks = output_root / "masks"

    output_npy.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    mass = load_metadata(processed_root, "mass")
    calc = load_metadata(processed_root, "calc")

    mass_index = {
        tuple(row[column] for column in KEY_COLUMNS): row
        for _, row in mass.iterrows()
    }
    calc_index = {
        tuple(row[column] for column in KEY_COLUMNS): row
        for _, row in calc.iterrows()
    }

    all_keys = sorted(set(mass_index) | set(calc_index))
    shared_keys = set(mass_index) & set(calc_index)

    expected_count = len(mass) + len(calc) - len(shared_keys)

    if len(all_keys) != expected_count:
        raise RuntimeError(
            f"Unexpected combined count: {len(all_keys)} vs {expected_count}"
        )

    combined_rows = []
    link_counts = {"hardlink": 0, "copy": 0}
    shared_count = 0

    for index, key in enumerate(all_keys, start=1):
        patient_id, side, view = key
        filename = f"{patient_id}_{side}_{view}"

        source_rows = []

        if key in mass_index:
            source_rows.append(mass_index[key])

        if key in calc_index:
            source_rows.append(calc_index[key])

        source_datasets = [str(row["dataset_name"]) for row in source_rows]

        source_images = []
        source_masks = []

        for row in source_rows:
            dataset_root = processed_root / row["dataset_name"]

            source_image = dataset_root / str(row["npy_path"])
            source_mask = dataset_root / str(row["mask_path"])

            image, mask = validate_source_pair(source_image, source_mask)

            source_images.append((source_image, image))
            source_masks.append((source_mask, mask))

        reference_image_path, reference_image = source_images[0]
        reference_shape = reference_image.shape

        for other_image_path, other_image in source_images[1:]:
            if other_image.shape != reference_shape:
                raise ValueError(
                    f"Shared image shape mismatch for {key}: "
                    f"{reference_shape} vs {other_image.shape}"
                )

            if not np.array_equal(reference_image, other_image):
                raise ValueError(
                    f"Shared NPY images are not identical for {key}: "
                    f"{reference_image_path} vs {other_image_path}"
                )

        crop_columns = [
            "crop_rmin",
            "crop_rmax",
            "crop_cmin",
            "crop_cmax",
        ]

        for column in crop_columns:
            values = {int(row[column]) for row in source_rows}

            if len(values) != 1:
                raise ValueError(
                    f"Shared crop mismatch for {key}, column {column}: "
                    f"{sorted(values)}"
                )

        destination_image = output_npy / f"{filename}.npy"
        method = link_or_copy(reference_image_path, destination_image)
        link_counts[method] += 1

        destination_mask = output_masks / f"{filename}.png"

        if len(source_masks) == 1:
            method = link_or_copy(source_masks[0][0], destination_mask)
            link_counts[method] += 1
            combined_mask = source_masks[0][1]
        else:
            shared_count += 1

            combined_mask = np.zeros(reference_shape, dtype=np.uint8)

            for _, mask in source_masks:
                combined_mask = np.maximum(
                    combined_mask,
                    (mask > 0).astype(np.uint8) * 255,
                )

            success = cv2.imwrite(str(destination_mask), combined_mask)

            if not success:
                raise IOError(f"Could not save combined mask: {destination_mask}")

        pathology_values = [str(row["pathology"]) for row in source_rows]
        pathology = aggregate_pathology(pathology_values)

        source_pathology_values = split_pipe_values(
            [row.get("source_pathologies", row["pathology"]) for row in source_rows]
        )
        unique_source_pathologies = sorted(
            {value.upper() for value in source_pathology_values}
        )

        alignment_methods = split_pipe_values(
            [
                row.get("mask_alignment_methods", "")
                for row in source_rows
            ]
        )

        source_partitions = sorted(
            {
                str(row.get("source_partition", "unknown"))
                for row in source_rows
            }
        )

        source_image_dicoms = sorted(
            {
                str(row["source_image_dicom"])
                for row in source_rows
            }
        )

        source_mask_dicoms = sorted(
            set(
                split_pipe_values(
                    [row["source_mask_dicoms"] for row in source_rows]
                )
            )
        )

        abnormality_type = (
            "mass+calcification"
            if len(source_rows) == 2
            else (
                "mass"
                if int(source_rows[0]["mass_present"]) == 1
                else "calcification"
            )
        )

        combined_rows.append(
            {
                "image_key": filename,
                "patient_id": patient_id,
                "side": side,
                "view": view,
                "abnormality_type": abnormality_type,
                "pathology": pathology,
                "malignant": int(
                    any(int(row["malignant"]) == 1 for row in source_rows)
                ),
                "mass_present": int(
                    any(int(row["mass_present"]) == 1 for row in source_rows)
                ),
                "calc_present": int(
                    any(int(row["calc_present"]) == 1 for row in source_rows)
                ),
                "mixed_pathology": int(len(unique_source_pathologies) > 1),
                "lesion_count": int(
                    sum(int(row["lesion_count"]) for row in source_rows)
                ),
                "source_datasets": "|".join(sorted(source_datasets)),
                "source_pathologies": "|".join(unique_source_pathologies),
                "height": int(reference_shape[0]),
                "width": int(reference_shape[1]),
                "mask_pixels": int(np.count_nonzero(combined_mask)),
                "dropped_mask_pixels": int(
                    sum(
                        int(row.get("dropped_mask_pixels", 0))
                        for row in source_rows
                    )
                ),
                "crop_rmin": int(source_rows[0]["crop_rmin"]),
                "crop_rmax": int(source_rows[0]["crop_rmax"]),
                "crop_cmin": int(source_rows[0]["crop_cmin"]),
                "crop_cmax": int(source_rows[0]["crop_cmax"]),
                "npy_path": f"npy/{filename}.npy",
                "mask_path": f"masks/{filename}.png",
                "source_partitions": "|".join(source_partitions),
                "mask_alignment_methods": "|".join(alignment_methods),
                "source_image_dicoms": "|".join(source_image_dicoms),
                "source_mask_dicoms": "|".join(source_mask_dicoms),
            }
        )

        if index % 100 == 0 or index == len(all_keys):
            print(f"[{index}/{len(all_keys)}] combined")

    combined = pd.DataFrame(combined_rows)
    metadata_path = output_root / "image_metadata.csv"
    combined.to_csv(metadata_path, index=False)

    problems = []

    for row_index, row in combined.iterrows():
        image_path = output_root / row["npy_path"]
        mask_path = output_root / row["mask_path"]

        try:
            image, mask = validate_source_pair(image_path, mask_path)

            if image.shape != (int(row["height"]), int(row["width"])):
                problems.append(
                    (
                        row_index,
                        "metadata shape mismatch",
                        image.shape,
                        (row["height"], row["width"]),
                    )
                )

        except Exception as exc:
            problems.append((row_index, str(exc)))

    if problems:
        for problem in problems[:20]:
            print(problem)

        raise RuntimeError(
            f"Combined dataset validation failed with {len(problems)} problems"
        )

    print("\nCombined dataset created successfully")
    print("Mass input records:", len(mass))
    print("Calc input records:", len(calc))
    print("Shared mammograms merged:", len(shared_keys))
    print("Unique combined mammograms:", len(combined))
    print("Shared masks unioned:", shared_count)
    print("Hard links created:", link_counts["hardlink"])
    print("Files copied:", link_counts["copy"])
    print("Metadata:", metadata_path)

    print("\nFinding combinations:")
    print(
        combined.groupby(["mass_present", "calc_present"])
        .size()
        .to_string()
    )

    print("\nMalignancy labels:")
    print(combined["malignant"].value_counts().sort_index().to_string())

    print("\nMixed-pathology images:")
    print(int(combined["mixed_pathology"].sum()))

    print("\n✅ Combined dataset passed validation")


if __name__ == "__main__":
    main()
    