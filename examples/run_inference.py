"""Run the locked end-to-end pipeline on one model-ready NPY mammogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from breast_cancer_detection.config import (
    CLASSIFIER_THRESHOLD,
    SEGMENTATION_THRESHOLD,
)
from breast_cancer_detection.inference import BreastCancerPipeline
from breast_cancer_detection.preprocessing import load_model_ready_image
from breast_cancer_detection.visualization import save_inference_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Prepared uint16 1024x640 NPY image")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("inference.png"))
    parser.add_argument("--device", default=None, help="For example: cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_model_ready_image(args.image)
    pipeline = BreastCancerPipeline(
        args.classifier_checkpoint,
        args.segmentation_checkpoint,
        device=args.device,
    )
    result = pipeline.predict(image)
    save_inference_figure(image, result, args.output)
    print(
        json.dumps(
            {
                "malignancy_probability": result.malignancy_probability,
                "predicted_class": result.predicted_class,
                "predicted_lesion_pixels": int(result.segmentation_mask.sum()),
                "classifier_threshold": CLASSIFIER_THRESHOLD,
                "segmentation_threshold": SEGMENTATION_THRESHOLD,
                "figure": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
