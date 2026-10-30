# Data

The raw and derived datasets are intentionally not versioned.

1. Obtain the [CBIS-DDSM collection](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
   and its associated metadata. Dataset DOI:
   [`10.7937/K9/TCIA.2016.7O02S9CY`](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY).
2. From the repository root, expose the collection as `CBIS-DDSM/` and metadata
   under `metadata/`, or adapt the source paths in the supplied preprocessing
   scripts.
3. Run the four stages listed in
   [`notebooks/01_dataset_preprocessing.md`](../notebooks/01_dataset_preprocessing.md).

The final cache contract is:

```text
cbis_ddsm_model_ready_1024x640/
├── cache_info.json
├── masks/<image_key>.png      # uint8, values 0 and 255
├── npy/<image_key>.npy        # uint16, shape 1024 x 640
└── splits/
    ├── all_with_split.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Expected final image counts are 1,885 train, 408 validation, and 394 test.
Never commit raw DICOM, NPY caches, or local path-bearing manifests.
