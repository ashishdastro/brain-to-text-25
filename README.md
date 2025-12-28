# 🧠 Brain-to-Text ’25 — Kaggle Competition

Decode **speech directly from human brain activity**.

This repository contains a clean, reproducible pipeline for the  
**Brain-to-Text ’25 Kaggle challenge**, including:

- data loading  
- pretrained baseline model  
- CTC decoding (now with **beam search**)  
- submission generation (`submission.csv`)

Leaderboard results: our baseline jumped from **125.5 → 1.00 PER** after better decoding.

---

## 🧩 Problem Overview

The dataset comes from ECoG recordings (electrodes placed on the brain surface).  
Participants silently imagine speech — the model must predict the sequence of **phonemes**.

> **Goal:**  
> Convert neural activity → ordered phoneme sequence.

Phonemes are sound building blocks:

| Word  | Phoneme sequence |
|------|------------------|
| cat  | K AE T           |
| dog  | D AO G           |
| hello | HH AH L OW      |

Evaluation metric:

### 📉 Phoneme Error Rate (PER)

(lower is better — like Word Error Rate in ASR).

---

## 🧪 Data Format

Training and test data are stored in **HDF5** files.

Each trial contains:

```

trial_xxxx/
input_features     # [time_steps, 512]
seq_class_ids      # phoneme labels (train only)

````

Test data does **not** include labels.

---

## 🤖 Model (Baseline)

This repo implements the official baseline model:

1️⃣ 512-dim neural features per timestep  
2️⃣ Adapter projects features → 256  
3️⃣ GRU / RNN encodes the time sequence  
4️⃣ Output layer predicts **41 phoneme classes**  
5️⃣ **CTC (Connectionist Temporal Classification)** handles alignment

CTC lets the network output variable-length phoneme sequences without knowing exact timing.

---

## 🔓 Decoding (Important!)

Originally the baseline used **greedy decoding**.  
We upgraded to:

### ⭐ CTC Prefix Beam Search (beam width = 10)

Benefits:

- considers multiple candidate sequences
- handles repeated symbols + blanks correctly
- dramatically improved leaderboard score

Beam width is configurable.

---

## 🚀 Usage

Install the project:

```bash
pip install -e .
````

Run inference:

```bash
python scripts/run_inference.py
```

This produces:

```
submission.csv
```

### ⚠️ Note on Kaggle IDs

Kaggle expects **numeric IDs (0…N-1)** — the script ensures the correct format.

---

## 📁 Project Structure (Medium-style)

```
ml_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   └── brain2text/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── utils/
│       └── prediction/
├── scripts/
├── tests/
└── README.md
```

---

## 📈 Roadmap

This repository is a starting point — upcoming improvements:

* ✔ beam-search decoding
* 🔜 language-model assisted decoding
* 🔜 model training + tuning
* 🔜 validation and PER tracking
* 🔜 signal preprocessing experiments (power, STFT, etc.)

Contributions welcome — this project is intentionally structured for iteration.

---

## 🙌 Acknowledgements

Kaggle organizers and the research teams making open BCI datasets available —
this competition pushes the boundary of non-invasive communication technologies.