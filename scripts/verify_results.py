"""Recompute key reported metrics from the compact row-level result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_THRESHOLD = 0.29150390625


def verify_classification() -> float:
    directory = ROOT / "results" / "classification" / "evaluation"
    predictions = pd.read_csv(directory / "test_predictions.csv")
    reported = json.loads((directory / "final_test_metrics.json").read_text())

    targets = predictions["target"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    predicted = (probabilities >= CLASSIFIER_THRESHOLD).astype(int)
    if not np.array_equal(predicted, predictions["predicted_label"]):
        raise AssertionError("Stored classifier labels do not match the locked threshold")

    tp = int(((targets == 1) & (predicted == 1)).sum())
    tn = int(((targets == 0) & (predicted == 0)).sum())
    fp = int(((targets == 0) & (predicted == 1)).sum())
    fn = int(((targets == 1) & (predicted == 0)).sum())
    checks = {
        "threshold": CLASSIFIER_THRESHOLD,
        "accuracy": (tp + tn) / len(targets),
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "precision": tp / (tp + fp),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    for name, value in checks.items():
        if not np.isclose(value, reported[name], rtol=0, atol=1e-12):
            raise AssertionError(f"Classifier {name}: {value} != {reported[name]}")

    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    precision_at_rank = np.cumsum(sorted_targets) / (np.arange(len(targets)) + 1)
    csv_average_precision = float(
        (precision_at_rank * sorted_targets).sum() / sorted_targets.sum()
    )
    return csv_average_precision - float(reported["pr_auc"])


def verify_segmentation() -> None:
    directory = ROOT / "results" / "segmentation" / "final"
    per_image = pd.read_csv(directory / "test_segmentation_metrics.csv")
    reported = json.loads(
        (directory / "final_segmentation_evaluation.json").read_text()
    )["test_metrics"]
    checks = {
        "n_images": len(per_image),
        "mean_soft_dice": per_image["soft_dice"].mean(),
        "mean_dice": per_image["dice"].mean(),
        "median_dice": per_image["dice"].median(),
        "mean_iou": per_image["iou"].mean(),
        "median_iou": per_image["iou"].median(),
        "mean_precision": per_image["precision"].mean(),
        "mean_recall": per_image["recall"].mean(),
    }
    for name, value in checks.items():
        if not np.isclose(value, reported[name], rtol=0, atol=1e-12):
            raise AssertionError(f"Segmentation {name}: {value} != {reported[name]}")


def verify_segmentation_threshold() -> None:
    directory = ROOT / "results" / "segmentation" / "transfer_learning"
    sweep = pd.read_csv(directory / "validation_threshold_sweep.csv")
    reported = json.loads((directory / "threshold_selection.json").read_text())
    best = sweep.loc[sweep["mean_dice"].idxmax()]
    if not np.isclose(best["threshold"], reported["selected_threshold"]):
        raise AssertionError("Segmentation threshold is not the validation maximum")


def main() -> None:
    pr_auc_delta = verify_classification()
    verify_segmentation()
    verify_segmentation_threshold()
    print("Result verification passed.")
    print(
        "Classifier CSV average-precision delta from full-precision JSON: "
        f"{pr_auc_delta:+.10f}"
    )


if __name__ == "__main__":
    main()
