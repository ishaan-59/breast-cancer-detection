"""Build the deterministic 1024 x 640 cache consumed by the notebooks.

This script documents a reconstructed transformation contract. The historical
workflow did not preserve a script under this filename. It keeps the existing
patient-level split labels unchanged while resizing each aligned image/mask
pair, left-aligning it, and vertically centering it on a fixed-size canvas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


TARGET_HEIGHT = 1024
TARGET_WIDTH = 640
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Cropped/aligned dataset root containing splits/all_with_split.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cbis_ddsm_model_ready_1024x640"),
        help="Destination for the model-ready cache.",
    )
    return parser.parse_args()


def resize_and_pad_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("Images and masks must both be two-dimensional")
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")

    source_height, source_width = image.shape
    scale = min(target_height / source_height, target_width / source_width)
    resized_height = max(1, int(round(source_height * scale)))
    resized_width = max(1, int(round(source_width * scale)))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_mask = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    output_image = np.zeros((target_height, target_width), dtype=np.uint16)
    output_mask = np.zeros((target_height, target_width), dtype=np.uint8)

    row_start = (target_height - resized_height) // 2
    row_end = row_start + resized_height
    col_start = 0
    col_end = resized_width

    output_image[row_start:row_end, col_start:col_end] = resized_image.astype(
        np.uint16
    )
    output_mask[row_start:row_end, col_start:col_end] = (
        (resized_mask > 0).astype(np.uint8) * 255
    )
    return output_image, output_mask


def build_cache(source_root: Path, output_root: Path) -> None:
    manifest_path = source_root / "splits" / "all_with_split.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_columns = {"image_key", "npy_path", "mask_path", "split"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    unexpected_splits = set(manifest["split"].dropna()) - set(SPLIT_NAMES)
    if unexpected_splits:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected_splits)}")

    npy_dir = output_root / "npy"
    mask_dir = output_root / "masks"
    split_dir = output_root / "splits"
    for directory in (npy_dir, mask_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cached_rows: list[dict[str, object]] = []
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Building model-ready cache",
    ):
        source_image_path = source_root / row.npy_path
        source_mask_path = source_root / row.mask_path
        image = np.load(source_image_path)
        mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unreadable mask: {source_mask_path}")
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {row.image_key}: "
                f"{image.shape} vs {mask.shape}"
            )

        image = image.astype(np.uint16, copy=False)
        mask = (mask > 0).astype(np.uint8) * 255
        cached_image, cached_mask = resize_and_pad_pair(image, mask)

        image_filename = f"{row.image_key}.npy"
        mask_filename = f"{row.image_key}.png"
        cached_image_path = npy_dir / image_filename
        cached_mask_path = mask_dir / mask_filename
        np.save(cached_image_path, cached_image)
        if not cv2.imwrite(str(cached_mask_path), cached_mask):
            raise IOError(f"Could not write mask: {cached_mask_path}")

        row_dict = row._asdict()
        row_dict.update(
            {
                "npy_path": f"npy/{image_filename}",
                "mask_path": f"masks/{mask_filename}",
                "height": TARGET_HEIGHT,
                "width": TARGET_WIDTH,
                "mask_pixels": int((cached_mask > 0).sum()),
            }
        )
        cached_rows.append(row_dict)

    cached_manifest = pd.DataFrame(cached_rows)
    cached_manifest.to_csv(split_dir / "all_with_split.csv", index=False)
    for split_name in SPLIT_NAMES:
        cached_manifest.loc[cached_manifest["split"] == split_name].to_csv(
            split_dir / f"{split_name}.csv",
            index=False,
        )

    cache_info = {
        "target_height": TARGET_HEIGHT,
        "target_width": TARGET_WIDTH,
        "image_dtype": "uint16",
        "mask_dtype": "uint8",
        "mask_values": [0, 255],
        "image_interpolation": "cv2.INTER_AREA",
        "mask_interpolation": "cv2.INTER_NEAREST",
        "alignment": {"horizontal": "left", "vertical": "center"},
        "num_images": int(len(cached_manifest)),
    }
    with (output_root / "cache_info.json").open("w", encoding="utf-8") as file:
        json.dump(cache_info, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    build_cache(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
