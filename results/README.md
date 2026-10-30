# Results provenance

These compact JSON and CSV files were extracted from the supplied experiment
artifacts. Aggregate claims in the root README were checked against the
prediction-level CSVs where available.

- `classification/training`: checkpoint-selection metadata and histories.
- `classification/evaluation`: locked threshold, predictions, final metrics,
  subgroup results, and confidence intervals.
- `segmentation/sanity`: single-image overfit diagnostic.
- `segmentation/baseline`: vanilla U-Net history and validation sweep.
- `segmentation/transfer_learning`: decoder-only and `layer4` fine-tuning,
  followed by the valid final validation sweep.
- `segmentation/final`: one-time held-out test metrics and intervals.
- `gradcam`: per-image and grouped qualitative ROI-overlap measurements.
- `inference`: compact end-to-end example metadata.

The stale archive named `segmentation_validation_artifact (1).zip` was excluded
as directed. It represented a transitional evaluation state. The valid final
ResNet50-U-Net validation source was `resnet50_unet_validation_artifact.zip`.

Run `python scripts/verify_results.py` from the repository root to recompute the
key thresholded classification metrics, final segmentation aggregates, and the
validation-selected segmentation threshold from the compact files.
