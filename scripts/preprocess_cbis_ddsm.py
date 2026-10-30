import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom


BASE_DIR = Path("CBIS-DDSM")


@dataclass(frozen=True)
class ResolvedMask:
    path: Path
    pixels: np.ndarray
    alignment_method: str
    original_shape: tuple[int, int]
    scale_x: float
    scale_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correctly preprocess CBIS-DDSM mammograms and aligned ROI masks."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["mass", "calc", "all"],
        required=True,
        help="Subset to process.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed"),
        help="Root directory for generated NPY files, masks, overlays, and CSVs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique mammograms after filtering.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Number of contour-overlay previews to save.",
    )
    parser.add_argument(
        "--patient-id",
        nargs="+",
        default=None,
        help=(
            "Optional patient ID filter. Accepts one or more IDs, for example "
            "--patient-id P_00059 P_00108."
        ),
    )
    parser.add_argument(
        "--scale-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum relative difference between horizontal and vertical mask "
            "scale factors when resizing a smaller ROI mask. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--min-mask-scale",
        type=float,
        default=0.50,
        help="Minimum accepted mask-to-image scale factor. Default: 0.50.",
    )
    parser.add_argument(
        "--max-mask-scale",
        type=float,
        default=2.00,
        help="Maximum accepted mask-to-image scale factor. Default: 2.00.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=500,
        help="Minimum accepted cropped image height and width. Default: 500.",
    )
    return parser.parse_args()


def clean_string(value: object) -> str:
    return str(value).replace("\\n", "\n").strip()


def path_lines(value: object) -> list[str]:
    return [
        line.strip()
        for line in clean_string(value).splitlines()
        if line.strip()
    ]


def direct_existing_path(value: object) -> Path | None:
    for candidate_string in path_lines(value):
        candidate = Path(candidate_string)

        if candidate.exists():
            return candidate

        if not candidate_string.startswith("CBIS-DDSM"):
            prefixed = BASE_DIR / candidate
            if prefixed.exists():
                return prefixed

    return None


def top_level_case_folder(value: object) -> Path:
    """Extract the CBIS-DDSM case folder from a current or stale path."""
    for candidate_string in path_lines(value):
        normalized = candidate_string.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]

        for index, part in enumerate(parts):
            if part == "CBIS-DDSM" and index + 1 < len(parts):
                return BASE_DIR / parts[index + 1]

        for part in parts:
            if part.startswith(("Mass-", "Calc-")):
                return BASE_DIR / part

    raise ValueError(f"Could not extract case folder from: {value}")


def dicom_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Case folder does not exist: {folder}")

    return sorted(
        path
        for path in folder.rglob("*.dcm")
        if not path.name.startswith("._")
    )


def header_shape(path: Path) -> tuple[int, int] | None:
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )
        rows = int(getattr(ds, "Rows", 0))
        columns = int(getattr(ds, "Columns", 0))

        if rows > 0 and columns > 0:
            return rows, columns
    except Exception:
        return None

    return None


def series_description(path: Path) -> str:
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )
        return str(getattr(ds, "SeriesDescription", "")).lower()
    except Exception:
        return ""


def read_full_mammogram(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    image = np.asarray(ds.pixel_array).squeeze()

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2-D mammogram, found {image.shape}: {path}"
        )

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image

    return image


def read_raw_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    pixels = np.asarray(ds.pixel_array).squeeze()

    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D DICOM, found {pixels.shape}: {path}")

    return pixels


def validate_source_path(
    path: Path,
    dataset_name: str,
    patient_id: str,
    side: str,
    view: str,
) -> None:
    lower_path = str(path).lower()
    expected_tokens = [
        dataset_name.lower(),
        patient_id.lower(),
        side.lower(),
        view.lower(),
    ]
    missing = [token for token in expected_tokens if token not in lower_path]

    if missing:
        raise ValueError(
            f"Incorrect source mapping. Missing tokens {missing} from path: {path}"
        )


def resolve_full_image(row: pd.Series, dataset_name: str) -> Path:
    image_reference = row["image_reference"]
    direct_path = direct_existing_path(image_reference)

    if direct_path is not None:
        candidates = [direct_path]
    else:
        folder = top_level_case_folder(image_reference)
        candidates = dicom_files(folder)

    if not candidates:
        raise FileNotFoundError(
            f"No mammogram candidates found for {image_reference}"
        )

    scored_candidates: list[tuple[int, Path]] = []

    for path in candidates:
        shape = header_shape(path)
        if shape is None:
            continue

        description = series_description(path)
        score = shape[0] * shape[1]

        if "full mammogram" in description:
            score += 10**12

        scored_candidates.append((score, path))

    if not scored_candidates:
        raise FileNotFoundError(
            f"No readable mammogram found for {image_reference}"
        )

    selected = max(scored_candidates, key=lambda item: item[0])[1]

    validate_source_path(
        selected,
        dataset_name=dataset_name,
        patient_id=str(row["patient_id"]),
        side=str(row["side"]),
        view=str(row["view"]),
    )

    return selected


def is_binary_mask(mask: np.ndarray) -> bool:
    unique_values = np.unique(mask)
    return 1 < len(unique_values) <= 4 and np.count_nonzero(mask) > 0


def resolve_roi_mask(
    row: pd.Series,
    dataset_name: str,
    full_shape: tuple[int, int],
    scale_tolerance: float,
    min_mask_scale: float,
    max_mask_scale: float,
) -> ResolvedMask:
    """
    Resolve a binary ROI-mask DICOM.

    Exact-size masks are used directly. Masks with a uniformly different
    resolution are resized to the full mammogram shape using nearest-neighbor
    interpolation, preserving their binary labels.
    """
    mask_reference = row["mask_reference"]
    direct_path = direct_existing_path(mask_reference)

    candidate_paths: list[Path] = []
    if direct_path is not None:
        candidate_paths.append(direct_path)

    folder = top_level_case_folder(mask_reference)
    if folder.exists():
        candidate_paths.extend(dicom_files(folder))

    candidate_paths = sorted(set(candidate_paths))
    valid_masks: list[
        tuple[int, Path, np.ndarray, str, tuple[int, int], float, float]
    ] = []

    full_height, full_width = full_shape

    for path in candidate_paths:
        try:
            pixels = read_raw_pixels(path)
        except Exception:
            continue

        if not is_binary_mask(pixels):
            continue

        original_shape = (int(pixels.shape[0]), int(pixels.shape[1]))
        mask_height, mask_width = original_shape
        alignment_method = "exact"
        scale_x = 1.0
        scale_y = 1.0

        if original_shape != full_shape:
            # These are the factors needed to enlarge/shrink the mask to the
            # mammogram coordinate system.
            scale_y = full_height / mask_height
            scale_x = full_width / mask_width

            relative_scale_difference = (
                abs(scale_x - scale_y) / max(scale_x, scale_y)
            )

            if relative_scale_difference > scale_tolerance:
                continue

            if not (
                min_mask_scale <= scale_x <= max_mask_scale
                and min_mask_scale <= scale_y <= max_mask_scale
            ):
                continue

            pixels = cv2.resize(
                (pixels > 0).astype(np.uint8),
                (full_width, full_height),
                interpolation=cv2.INTER_NEAREST,
            )
            alignment_method = "nearest_resize"
        else:
            pixels = (pixels > 0).astype(np.uint8)

        if pixels.shape != full_shape or np.count_nonzero(pixels) == 0:
            continue

        description = series_description(path)
        score = 0

        if "roi mask" in description:
            score += 100
        if direct_path is not None and path == direct_path:
            score += 10
        if alignment_method == "exact":
            score += 20
        else:
            score += 10

        valid_masks.append(
            (
                score,
                path,
                pixels,
                alignment_method,
                original_shape,
                scale_x,
                scale_y,
            )
        )

    if not valid_masks:
        raise FileNotFoundError(
            "No valid binary ROI mask found for "
            f"{row['patient_id']} {row['side']} {row['view']} "
            f"using {mask_reference}"
        )

    (
        _,
        selected_path,
        selected_pixels,
        alignment_method,
        original_shape,
        scale_x,
        scale_y,
    ) = max(valid_masks, key=lambda item: item[0])

    validate_source_path(
        selected_path,
        dataset_name=dataset_name,
        patient_id=str(row["patient_id"]),
        side=str(row["side"]),
        view=str(row["view"]),
    )

    return ResolvedMask(
        path=selected_path,
        pixels=(selected_pixels > 0).astype(np.uint8),
        alignment_method=alignment_method,
        original_shape=original_shape,
        scale_x=float(scale_x),
        scale_y=float(scale_y),
    )


def calculate_breast_crop(image: np.ndarray) -> tuple[int, int, int, int]:
    """Reproduce the spatial crop used by the original preprocessing code."""
    image_float = image.astype(np.float32)
    denominator = image_float.max() - image_float.min()

    if denominator < 1e-6:
        raise ValueError("Mammogram has no usable intensity variation")

    normalized = (image_float - image_float.min()) / (denominator + 1e-6)
    binary = (normalized > 0.2).astype(np.uint8)

    binary[:50, :] = 0
    binary[-50:, :] = 0
    binary[:, -50:] = 0

    number_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, 8
    )

    if number_labels <= 1:
        raise ValueError("No breast component found")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast_component = labels == largest
    rows, columns = np.where(breast_component)

    if len(rows) == 0 or len(columns) == 0:
        raise ValueError("Largest breast component is empty")

    rmin = int(rows.min())
    rmax = int(rows.max())
    cmin = int(columns.min())
    cmax = int(columns.max())

    if rmax <= rmin or cmax <= cmin:
        raise ValueError("Invalid breast crop coordinates")

    return rmin, rmax, cmin, cmax


def normalize_for_preview(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    denominator = image_float.max() - image_float.min()

    if denominator < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image_float - image_float.min()) / (denominator + 1e-6)
    return (normalized * 255).astype(np.uint8)


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    grayscale = normalize_for_preview(image)
    overlay = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    binary_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=8,
    )

    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Could not save overlay: {output_path}")


def aggregate_pathology(values: pd.Series) -> str:
    pathologies = set(values.dropna().astype(str).str.upper())

    if "MALIGNANT" in pathologies:
        return "MALIGNANT"
    if "BENIGN" in pathologies:
        return "BENIGN"
    if "BENIGN_WITHOUT_CALLBACK" in pathologies:
        return "BENIGN_WITHOUT_CALLBACK"

    raise ValueError(f"Unexpected pathology values: {pathologies}")


def source_partition(path: Path) -> str:
    lower_path = str(path).lower()

    if "training" in lower_path:
        return "training"
    if "test" in lower_path:
        return "test"
    return "unknown"


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "mass":
        dataframe = pd.read_csv("metadata/master_dataset_index.csv")
        standardized = pd.DataFrame(
            {
                "patient_id": dataframe["patient_id"],
                "pathology": dataframe["pathology"],
                "side": dataframe["left or right breast"]
                .astype(str)
                .str.upper(),
                "view": dataframe["image view"].astype(str).str.upper(),
                "image_reference": dataframe["full_image"],
                "mask_reference": dataframe["roi_mask"],
                "abnormality_type": "mass",
            }
        )
    elif dataset_name == "calc":
        dataframe = pd.read_csv("metadata/calc_dataset_index.csv")
        standardized = pd.DataFrame(
            {
                "patient_id": dataframe["patient_id"],
                "pathology": dataframe["pathology"],
                "side": dataframe["side"].astype(str).str.upper(),
                "view": dataframe["view"].astype(str).str.upper(),
                "image_reference": dataframe["image file path"],
                "mask_reference": dataframe["roi_mask"],
                "abnormality_type": "calcification",
            }
        )
    else:
        raise ValueError(dataset_name)

    standardized["patient_id"] = (
        standardized["patient_id"].astype(str).str.strip().str.upper()
    )
    standardized["side"] = standardized["side"].astype(str).str.strip().str.upper()
    standardized["view"] = standardized["view"].astype(str).str.strip().str.upper()
    return standardized


def process_dataset(
    dataset_name: str,
    output_root: Path,
    limit: int | None,
    preview_count: int,
    patient_ids: list[str] | None,
    scale_tolerance: float,
    min_mask_scale: float,
    max_mask_scale: float,
    min_crop_size: int,
) -> None:
    dataframe = load_dataset(dataset_name)

    if patient_ids:
        wanted = {patient_id.strip().upper() for patient_id in patient_ids}
        dataframe = dataframe[dataframe["patient_id"].isin(wanted)].copy()

        missing = wanted - set(dataframe["patient_id"])
        if missing:
            raise ValueError(
                f"No rows found for patient ID(s): {sorted(missing)}"
            )

    output_directory = output_root / dataset_name
    npy_directory = output_directory / "npy"
    mask_directory = output_directory / "masks"
    preview_directory = output_directory / "overlays"

    npy_directory.mkdir(parents=True, exist_ok=True)
    mask_directory.mkdir(parents=True, exist_ok=True)

    if preview_count > 0:
        preview_directory.mkdir(parents=True, exist_ok=True)

    grouped = list(
        dataframe.groupby(["patient_id", "side", "view"], sort=True)
    )

    if limit is not None:
        grouped = grouped[:limit]

    processed_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    previews_saved = 0

    print("\n" + "=" * 80)
    print(dataset_name.upper())
    print("=" * 80)
    print("Unique mammograms to process:", len(grouped))

    for group_index, (group_key, group) in enumerate(grouped, start=1):
        patient_id, side, view = group_key
        filename = f"{patient_id}_{side}_{view}"

        print(
            f"[{group_index}/{len(grouped)}] {filename} | lesions={len(group)}"
        )

        try:
            first_row = group.iloc[0]
            full_image_path = resolve_full_image(first_row, dataset_name)
            full_image = read_full_mammogram(full_image_path)

            rmin, rmax, cmin, cmax = calculate_breast_crop(full_image)
            cropped_image = full_image[rmin : rmax + 1, cmin : cmax + 1]

            if (
                cropped_image.shape[0] < min_crop_size
                or cropped_image.shape[1] < min_crop_size
            ):
                raise ValueError(f"Suspiciously small crop: {cropped_image.shape}")

            union_mask = np.zeros(full_image.shape, dtype=np.uint8)
            resolved_mask_paths: list[str] = []
            alignment_methods: list[str] = []
            original_shapes: list[str] = []
            scale_factors: list[str] = []

            for _, lesion_row in group.iterrows():
                resolved = resolve_roi_mask(
                    lesion_row,
                    dataset_name=dataset_name,
                    full_shape=full_image.shape,
                    scale_tolerance=scale_tolerance,
                    min_mask_scale=min_mask_scale,
                    max_mask_scale=max_mask_scale,
                )

                union_mask = np.maximum(union_mask, resolved.pixels)
                resolved_mask_paths.append(str(resolved.path))
                alignment_methods.append(resolved.alignment_method)
                original_shapes.append(
                    f"{resolved.original_shape[0]}x{resolved.original_shape[1]}"
                )
                scale_factors.append(
                    f"x={resolved.scale_x:.6f},y={resolved.scale_y:.6f}"
                )

            cropped_mask = union_mask[rmin : rmax + 1, cmin : cmax + 1]

            if cropped_mask.shape != cropped_image.shape:
                raise ValueError(
                    "Cropped image and mask dimensions differ: "
                    f"{cropped_image.shape} vs {cropped_mask.shape}"
                )

            if np.count_nonzero(cropped_mask) == 0:
                raise ValueError("ROI mask became empty after cropping")

            total_mask_pixels = int(np.count_nonzero(union_mask))
            retained_mask_pixels = int(np.count_nonzero(cropped_mask))
            dropped_mask_pixels = total_mask_pixels - retained_mask_pixels

            npy_path = npy_directory / f"{filename}.npy"
            output_mask_path = mask_directory / f"{filename}.png"

            np.save(npy_path, cropped_image)

            if not cv2.imwrite(str(output_mask_path), cropped_mask * 255):
                raise IOError(f"Could not save mask: {output_mask_path}")

            if previews_saved < preview_count:
                preview_path = preview_directory / f"{filename}_overlay.png"
                save_overlay(cropped_image, cropped_mask, preview_path)
                previews_saved += 1

            pathology = aggregate_pathology(group["pathology"])

            processed_rows.append(
                {
                    "patient_id": patient_id,
                    "side": side,
                    "view": view,
                    "abnormality_type": (
                        "mass" if dataset_name == "mass" else "calcification"
                    ),
                    "pathology": pathology,
                    "malignant": int(pathology == "MALIGNANT"),
                    "mass_present": int(dataset_name == "mass"),
                    "calc_present": int(dataset_name == "calc"),
                    "lesion_count": len(group),
                    "source_pathologies": "|".join(
                        sorted(set(group["pathology"].astype(str)))
                    ),
                    "height": int(cropped_image.shape[0]),
                    "width": int(cropped_image.shape[1]),
                    "mask_pixels": retained_mask_pixels,
                    "dropped_mask_pixels": dropped_mask_pixels,
                    "crop_rmin": rmin,
                    "crop_rmax": rmax,
                    "crop_cmin": cmin,
                    "crop_cmax": cmax,
                    "npy_path": f"npy/{filename}.npy",
                    "mask_path": f"masks/{filename}.png",
                    "source_partition": source_partition(full_image_path),
                    "source_image_dicom": str(full_image_path),
                    "source_mask_dicoms": "|".join(
                        sorted(set(resolved_mask_paths))
                    ),
                    "mask_alignment_methods": "|".join(alignment_methods),
                    "mask_original_shapes": "|".join(original_shapes),
                    "mask_scale_factors": "|".join(scale_factors),
                    "resized_mask_count": sum(
                        method == "nearest_resize"
                        for method in alignment_methods
                    ),
                }
            )

            del full_image, cropped_image, union_mask, cropped_mask
            gc.collect()

        except Exception as exception:
            print(f"  ❌ {exception}")
            failed_rows.append(
                {
                    "patient_id": patient_id,
                    "side": side,
                    "view": view,
                    "error": str(exception),
                }
            )

        if group_index % 25 == 0 or group_index == len(grouped):
            pd.DataFrame(processed_rows).to_csv(
                output_directory / "image_metadata.csv",
                index=False,
            )
            pd.DataFrame(failed_rows).to_csv(
                output_directory / "failed_cases.csv",
                index=False,
            )

    print("\nCompleted:", dataset_name)
    print("Successful images:", len(processed_rows))
    print("Failed images:", len(failed_rows))


def main() -> None:
    args = parse_args()

    if args.scale_tolerance < 0:
        raise ValueError("--scale-tolerance must be non-negative")
    if args.min_mask_scale <= 0 or args.max_mask_scale <= 0:
        raise ValueError("Mask scale bounds must be positive")
    if args.min_mask_scale > args.max_mask_scale:
        raise ValueError("--min-mask-scale cannot exceed --max-mask-scale")
    if args.min_crop_size <= 0:
        raise ValueError("--min-crop-size must be positive")

    datasets = ["mass", "calc"] if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets:
        process_dataset(
            dataset_name=dataset_name,
            output_root=args.output_root,
            limit=args.limit,
            preview_count=args.preview_count,
            patient_ids=args.patient_id,
            scale_tolerance=args.scale_tolerance,
            min_mask_scale=args.min_mask_scale,
            max_mask_scale=args.max_mask_scale,
            min_crop_size=args.min_crop_size,
        )


if __name__ == "__main__":
    main()
    