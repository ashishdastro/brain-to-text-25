import csv
import glob
import h5py
import os
import torch
import pandas as pd

from brain2text.prediction.inference import load_pretrained_model, predict_single_trial

TOKEN_MAP = {i+1: p for i,p in enumerate([
    'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G',
    'HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T',
    'TH','UH','UW','V','W','Y','Z','ZH','|'
])}
TOKEN_MAP[0] = ""

MODEL = load_pretrained_model()

test_files = sorted(glob.glob(
    "data/raw/t15_copyTask_neuralData/hdf5_data_final/**/data_test.hdf5",
    recursive=True
))

rows = []

for path in test_files:
    session = os.path.basename(os.path.dirname(path))   # e.g., t15.2023.08.13
    
    with h5py.File(path, "r") as f:
        for trial_key in sorted(f.keys()):
            full_id = f"{session}_{trial_key}"          # make ID unique
            
            x = torch.tensor(f[trial_key]["input_features"][:], dtype=torch.float32)
            pred = predict_single_trial(MODEL, x, TOKEN_MAP)

            rows.append(pred)

with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","text"])
    for i, text in enumerate(rows):
        if text.strip() == "":
            text = " "
        w.writerow([i, text])

print("Saved submission.csv with", len(rows), "rows")
