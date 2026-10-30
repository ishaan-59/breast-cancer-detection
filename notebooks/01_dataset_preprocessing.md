# Stage 01 — dataset preprocessing provenance

The original preprocessing stage was supplied as scripts rather than an
executed notebook. This Markdown stage marker preserves that fact instead of
inventing notebook output.

Run order:

1. [`scripts/preprocess_cbis_ddsm.py`](../scripts/preprocess_cbis_ddsm.py)
   resolves DICOM mammograms and ROI masks, aligns masks, crops the breast, and
   writes mass/calcification image and mask records.
2. [`scripts/build_combined_dataset.py`](../scripts/build_combined_dataset.py)
   merges mass and calcification records into one unique-mammogram dataset and
   unions masks for shared mammograms.
3. [`scripts/create_patient_splits.py`](../scripts/create_patient_splits.py)
   creates deterministic, patient-disjoint train/validation/test assignments.
4. [`scripts/build_model_ready_cache.py`](../scripts/build_model_ready_cache.py)
   transforms the prepared pairs into the `uint16`, `1024 x 640` cache consumed
   by the notebooks while preserving the input split assignments.

The fourth script is a documented reconstruction of the final deterministic
cache-construction contract. This repository does not claim that the exact
filename existed during the historical experiment.
