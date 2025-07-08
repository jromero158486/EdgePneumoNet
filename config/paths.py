
# Import the PAth class from the pathlib module
# pathlib provides an object oriented way to interact with file system paths

from pathlib import Path

# __file__ is a built-in Python variable that holds the path to the current script file.
# Path(__file__) creates a Path object representing the current script's location.
# grandparent directory: parents[1]
# parent directory: parents[0]

"""
my_project/scripts/my_script.py -> Path(__file__).resolve().parents[1] -> myproject
"""
ROOT = Path(__file__).resolve().parents[1]

# ------- RSNA dataset -------
DATA_RSNA = ROOT / "RSNA"
TRAIN_IMG_DIR = ROOT/ "stage_2_train_images"
TEST_IMG_DIR   = DATA_RSNA / "stage_2_test_images"

CSV_TRAIN_DET  = DATA_RSNA / "stage_2_train_labels.csv"
CSV_CLASS_INFO = DATA_RSNA / "stage_2_detailed_class_info.csv"
CSV_SPLIT_OUT  = DATA_RSNA / "labels_split.csv"