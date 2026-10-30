# Preprocessing scripts

Run from the repository root after placing CBIS-DDSM and its metadata locally:

```bash
python scripts/preprocess_cbis_ddsm.py --dataset all --output-root processed
python scripts/build_combined_dataset.py --processed-root processed
python scripts/create_patient_splits.py \
  --metadata processed/combined/image_metadata.csv \
  --output-dir processed/combined/splits
python scripts/build_model_ready_cache.py \
  --source-root processed/combined \
  --output-root cbis_ddsm_model_ready_1024x640
```

The first three files are the supplied preprocessing sources. The final file is
the documented reconstruction of the deterministic cache stage used before the
training notebooks. It preserves the already-assigned patient-level `split`
column and does not perform a new split.
