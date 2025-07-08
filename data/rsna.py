
import torch, cv2, pydicom
from torch.utils.data import Dataset
import pandas as pd
from .transforms import train_tfms, val_tfms
from config.paths import (
    CSV_SPLIT_OUT, TRAIN_IMG_DIR, TEST_IMG_DIR
)

class RSNAPneumonia(Dataset):

    """
    Class definition: Inherits from torch.utils.data.Dataset, which is essential for PyTorch's DataLoader.
    """
    def __init__(self, split="train", use_png=True):

        """
        Constructor for the Dataset class.
        - split (str): Determines the dataset's role ("train", "val", or "test"). Default is "train".
        - use_png (bool): Flag to choose between loading PNG images or DICOM files. Default is False (meaning DICOM).
        """
        # Assertion: Ensures the 'split' argument is one of the valid options.
        assert split in {"train", "val", "test"}
        self.df = pd.read_csv(CSV_SPLIT_OUT if split != "test" else CSV_SPLIT_OUT)

        # Reads the CSV file defined by CSV_SPLIT_OUT into a pandas DataFrame.
        if split != "test":
            self.df = self.df[self.df["split"] == split]
        self.split = split
        self.use_png = use_png
        self.tfms = train_tfms if split == "train" else val_tfms
        # Transformation selection: Assigns `train_tfms` (likely including augmentations) for "train" split
        # and `val_tfms` (likely only normalization/resizing) for "val" or "test" splits.

    def _load_image(self, pid):
        if self.use_png:  # images convertidas
            path = TRAIN_IMG_DIR.with_name("rsna_png_train") / f"{pid}.png"
            img  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:             # leer DICOM directamente
            dcm_path = TRAIN_IMG_DIR / f"{pid}.dcm"
            ds  = pydicom.dcmread(str(dcm_path))
            img = ds.pixel_array.astype("float32")
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            img = (img * 255).astype("uint8")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def __len__(self):
        # Magic method: Returns the total number of items in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # Magic method: Retrieves a single sample from the dataset at a given index.

        row = self.df.iloc[idx]
        # Get row: Retrieves the DataFrame row corresponding to the given index. Looks good.
        img = self._load_image(row.patientId)
        # Load image: Calls the helper method to load the image for the patient ID in the current row. Looks good.
        img = self.tfms(image=img)["image"]
        # Apply transforms: Applies the selected image transformations (train_tfms or val_tfms) to the image.
        # Assumes the transforms return a dictionary with an "image" key, which is common with libraries like Albumentations. Looks good.

        if self.split == "test":
            return img, row.patientId
        # Test split return: If it's the "test" split, it returns the processed image and the patient ID.
        # The patient ID is often needed for generating submission files for competitions. Looks good.

        y = torch.tensor(row.Target, dtype=torch.long)
        # Target creation: For "train" or "val" splits, converts the "Target" column value to a PyTorch long tensor.
        # `dtype=torch.long` is appropriate for classification targets (integers). Looks good.

        return img, y