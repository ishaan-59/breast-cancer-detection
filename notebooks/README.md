# Executed notebooks

These are the supplied executed experiment records, renamed only to expose the
stage order. Their code cells and outputs were not rewritten during repository
restructuring.

| Stage | Notebook | Purpose |
|---:|---|---|
| 01 | [`01_dataset_preprocessing.md`](01_dataset_preprocessing.md) | Provenance map for the script-based preprocessing workflow |
| 02 | [`02_classifier_training.ipynb`](02_classifier_training.ipynb) | ResNet-50 classifier training |
| 03 | [`03_classifier_evaluation.ipynb`](03_classifier_evaluation.ipynb) | Validation threshold selection and held-out classifier test |
| 04 | [`04_segmentation_preparation_audit.ipynb`](04_segmentation_preparation_audit.ipynb) | Segmentation data and mask audit |
| 05 | [`05_segmentation_sanity_checks.ipynb`](05_segmentation_sanity_checks.ipynb) | Single-image overfit and pipeline sanity checks |
| 06 | [`06_vanilla_unet_training.ipynb`](06_vanilla_unet_training.ipynb) | Full-image vanilla U-Net baseline |
| 07 | [`07_resnet50_unet_training.ipynb`](07_resnet50_unet_training.ipynb) | Classifier-pretrained ResNet50-U-Net and `layer4` fine-tuning |
| 08 | [`08_final_segmentation_evaluation.ipynb`](08_final_segmentation_evaluation.ipynb) | Validation threshold lock and one-time segmentation test |
| 09 | [`09_gradcam_analysis.ipynb`](09_gradcam_analysis.ipynb) | Final classifier Grad-CAM/ROI comparison |
| 10 | [`10_end_to_end_inference.ipynb`](10_end_to_end_inference.ipynb) | Combined classification, Grad-CAM, segmentation, and visualization |

Notebook 10 displays a rounded classifier threshold in its historical output.
The locked validation-selected value used by reusable code and documentation is
exactly `0.29150390625`.
