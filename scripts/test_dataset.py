import os, sys, glob

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data.dataset import BrainDataset

INPUT_ROOT = os.path.join("data", "raw")

TRAIN_H5_FILES = sorted(
    glob.glob(
        os.path.join(
            INPUT_ROOT,
            "t15_copyTask_neuralData",
            "hdf5_data_final",
            "**",
            "data_train.hdf5"
        ), recursive=True
    )
)

print(f"Found {len(TRAIN_H5_FILES)} training files")

assert len(TRAIN_H5_FILES) > 0, "No train files found — check folder layout!"

ds = BrainDataset(TRAIN_H5_FILES[0])   # test first file

print("Using:", TRAIN_H5_FILES[0])

x, y = ds[0]

print("Input shape :", x.shape)
print("Target shape:", y.shape)
