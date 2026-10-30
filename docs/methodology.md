# Methodology and experimental safeguards

## Dataset preparation

The study uses CBIS-DDSM mammograms and ROI masks. Preprocessing resolves full
mammograms and binary masks from metadata, corrects MONOCHROME1 polarity,
aligns resolution-compatible masks, crops the breast region, merges mass and
calcification records by patient/side/view, and unions masks for mammograms
containing both finding types.

The combined data are split at the patient level with seed 42. The deterministic
search balances image counts, malignancy, finding type, and view. The final
image counts are 1,885 train, 408 validation, and 394 test (2,687 total), with
no patient overlap among splits.

The reconstructed model-ready cache stage reproduces the documented
`1024 × 640` image/mask contract using aspect-ratio-preserving resize and
padding. The repository implementation left-aligns horizontally and centers
vertically; the original historical cache-construction script was not
preserved. Images use `cv2.INTER_AREA` and are stored as `uint16` NPY arrays.
Masks use `cv2.INTER_NEAREST` and are stored as binary `uint8` PNGs with values
0/255. The split column is copied from the input manifest without reassignment.

## Classification

The classifier is an ImageNet-initialized ResNet-50 with one output logit.
Prepared grayscale images are divided by 65,535, repeated across three channels,
and normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

Training first optimizes the classification head, then fine-tunes `layer4` and
the head. The selected checkpoint, `resnet50_layer4_best.pth`, is chosen by
validation ROC-AUC (0.8227387422360248). Maximum Youden J on validation data
selects the exact classification threshold `0.29150390625`. Only after this
checkpoint and threshold are locked is the 394-image test split evaluated.

## Segmentation

A single-image U-Net overfit check reaches Dice 0.952583372592926 and IoU
0.9094600677490234, validating the mask/loss pipeline. The vanilla full-image
U-Net baseline reaches best validation soft Dice 0.1531899576560439.

The final model uses the classifier-trained ResNet-50 encoder. Stage A freezes
the encoder and trains only the decoder, reaching validation soft Dice
0.3329135484901581. Stage B unfreezes `layer4` with a smaller learning rate and
keeps BatchNorm frozen, reaching validation soft Dice 0.3570995714772103.
The chosen checkpoint is `resnet50_unet_layer4_best.pth`.

A separate validation pass measures soft Dice 0.3567443701690268 and selects
segmentation threshold `0.9` by mean per-image Dice. At that threshold,
validation hard Dice is 0.3770005702972412 and IoU is 0.2743273079395294. The
checkpoint and threshold are then locked before one test inference pass.

## Held-out evaluation

Classifier test results are stored in
[`results/classification/evaluation`](../results/classification/evaluation).
Final segmentation results are stored in
[`results/segmentation/final`](../results/segmentation/final), including
patient-bootstrap 95% confidence intervals. Prediction-level CSVs permit direct
recomputation of reported aggregate metrics.

The test set is not used to select a model, epoch, classifier threshold, or
segmentation threshold. The executed evaluation notebooks show the lock-before-
test ordering.

## Grad-CAM interpretation

Grad-CAM targets `layer4[-1]` of the locked classifier and uses the predicted
class direction. ROI overlap is a descriptive analysis, not a training target.
Because maps are diffuse and spatially coarse, the analysis is qualitative and
does not support lesion-localization or clinical claims.

## Reproducibility limitations

- Raw CBIS-DDSM data and derived caches are excluded due to size and dataset
  distribution constraints.
- All checkpoints are excluded from Git; expected filenames, sizes, and hashes
  are recorded in [`models/README.md`](../models/README.md).
- The final cache-builder is a reconstruction of the deterministic transform
  contract. The final cache itself was not supplied for byte-level comparison.
- Historical notebooks retain Kaggle paths and their original outputs. Paths
  require adaptation outside that environment.
- No original environment lockfile was available. Dependencies are declared,
  but exact historical transitive versions cannot be guaranteed.
- Classifier probabilities in the row-level CSV were written at limited decimal
  precision. Recomputing average precision from that CSV gives 0.6978343176,
  versus the full-precision reported PR-AUC 0.6977655494 (difference 0.0000688).
  All thresholded confusion-matrix metrics reproduce exactly.
- Notebook 10 retains the rounded threshold printed at experiment time. The
  exact validation-selected classifier threshold is `0.29150390625` everywhere
  in reusable code, tests, examples, and current documentation.
