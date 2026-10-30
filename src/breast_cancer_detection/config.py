"""Locked, experiment-backed constants used by production-facing inference."""

TARGET_HEIGHT = 1024
TARGET_WIDTH = 640

# Selected once on the classifier validation set using maximum Youden J.
CLASSIFIER_THRESHOLD = 0.29150390625

# Selected once on the segmentation validation set using mean per-image Dice.
SEGMENTATION_THRESHOLD = 0.9

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
