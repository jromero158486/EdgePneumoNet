"""
CSV with id, target (0/1), split (train/val).

"""
import random, shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pydicom
import cv2
import numpy as np

from config.paths import (
    TRAIN_IMG_DIR, CSV_TRAIN_DET, CSV_CLASS_INFO, CSV_SPLIT_OUT
)

# Original labels

bbox_df = pd.read_csv(CSV_TRAIN_DET) # Target, bbox
class_df = pd.read_csv(CSV_CLASS_INFO) 
labels_df = pd.read_csv(CSV_TRAIN_DET)[["patientId", "Target"]].drop_duplicates()

# Split
seed = 2025
random.seed(seed)
patient_ids = labels_df["patientId"].unique().tolist()
random.shuffle(patient_ids)

val_pct = 0.15
n_val = int(len(patient_ids) * val_pct)
val_ids = set(patient_ids[:n_val])
labels_df["split"] = labels_df["patientId"].apply(
    lambda pid: "val" if pid in val_ids else "train"
)

# Save csv
labels_df.to_csv(CSV_SPLIT_OUT, index = False)
print(f"CSV save {CSV_SPLIT_OUT}")

# DICOM → PNG

CONVERT = True         
OUT_DIR = TRAIN_IMG_DIR.with_name("rsna_png_train")

if CONVERT:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dcm_path in tqdm(list(TRAIN_IMG_DIR.glob("*.dcm"))):
        ds  = pydicom.dcmread(str(dcm_path))
        img = ds.pixel_array.astype(np.float32)

        # normaliza a 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        img = (img * 255).astype(np.uint8)

        png_path = OUT_DIR / (dcm_path.stem + ".png")
        cv2.imwrite(str(png_path), img)

    print(f"DICOMs converted to PNG: {OUT_DIR}")