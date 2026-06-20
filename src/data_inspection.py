from pathlib import Path

import pydicom
import numpy as np
import matplotlib.pyplot as plt

def inspect_dicom(dicom_path : str):

    dicom_path = Path(dicom_path)

    ds = pydicom.dcmread(dicom_path)

    pixels = ds.pixel_array

    print("|Main DICOM attributes|\n")
    print("Modality:", ds.Modality)
    print("Rows:", ds.Rows)
    print("Columns:", ds.Columns)
    print("BitsStored:", ds.BitsStored)
    print("BitsAllocated:", ds.BitsAllocated)
    print("PhotometricInterpretation:", ds.PhotometricInterpretation)
    print("PixelRepresentation:", ds.PixelRepresentation)

    print("Shape:", pixels.shape)
    print("Dtype:", pixels.dtype)
    print("Min:", pixels.min())
    print("Max:", pixels.max())
    print("Mean:", pixels.mean())
    print("Std:", pixels.std())

    plt.imshow(pixels, cmap="gray")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    inspect_dicom("/Users/ishaansingh/breast-cancer-detection/data/1-1.dcm")
