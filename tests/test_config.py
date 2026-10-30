from breast_cancer_detection.config import (
    CLASSIFIER_THRESHOLD,
    SEGMENTATION_THRESHOLD,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


def test_locked_thresholds_and_resolution() -> None:
    assert CLASSIFIER_THRESHOLD == 0.29150390625
    assert SEGMENTATION_THRESHOLD == 0.9
    assert (TARGET_HEIGHT, TARGET_WIDTH) == (1024, 640)
