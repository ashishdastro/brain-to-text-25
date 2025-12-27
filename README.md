## 🧠 Project Overview — Brain-to-Text ’25 (Kaggle)

This project is built for the **Brain-to-Text ’25 Kaggle competition**.
The goal is to decode **human speech directly from brain activity**.

Participants are given recordings from electrodes placed on the surface of the brain (ECoG).
While the participant silently *thinks* of words or phonemes, the neural signals are recorded.

The task:

> **Predict the sequence of phonemes (basic sound units of speech) for each brain recording.**

Phonemes are not full words — they are building blocks of spoken language.
Examples:

| Word  | Phoneme sequence |
| ----- | ---------------- |
| cat   | K AE T           |
| dog   | D AO G           |
| hello | HH AH L OW       |

The competition evaluates submissions using **Phoneme Error Rate (PER)** — similar to Word Error Rate used in speech recognition.

---

## 🎯 Why This Problem Matters

This research direction is important for:

* **medical neuro-prosthetics**
  Helping locked-in or paralyzed patients communicate again.

* **speech replacement devices**
  Instead of tracking muscle movement, decode *intended speech*.

* **neuroscience research**
  Understanding how language is represented in the brain.

This project demonstrates a full pipeline from neural data to readable phoneme predictions.

---

## 📂 What Data Looks Like

The data is stored in `HDF5` files. Each recording contains:

* `input_features` — neural signals over time (shape ~ `[time_steps, 512]`)
* labels (for training data only) — phoneme indices

Example structure:

```
trial_0000/
    input_features   →   [T, 512]
    seq_class_ids    →   [L]
```

For test data, labels are not provided.

---

## 🤖 Model Approach (Baseline)

This repo currently implements the **official competition baseline**:

1️⃣ **Neural features (512-dim vectors per timestep)**
2️⃣ Adapter layer reduces dimensionality → 256
3️⃣ **RNN (GRU)** processes temporal sequence
4️⃣ Output layer predicts **41 phoneme classes**
5️⃣ **CTC decoding** converts predictions into readable phoneme strings

CTC (Connectionist Temporal Classification) is commonly used for:

* speech recognition
* handwriting recognition
* any sequence where alignment isn't known

It lets the model output variable-length phoneme sequences from a fixed input length.

---

## 📈 What This Repository Contains

This repository provides:

✔ professional project structure
✔ data loaders for HDF5 files
✔ pretrained baseline model loader
✔ greedy CTC decoder
✔ inference pipeline → generates `submission.csv`

The current workflow:

```bash
pip install -e .
python scripts/run_inference.py
```

This produces a leaderboard-ready file:

```
id,text
t15.2023.08.13_trial_0000,HH AH L OW |
...
```

Upload to Kaggle → get baseline score.

---

## 🚀 Roadmap (What We’ll Improve Next)

This project serves as a foundation to build better systems:

* fine-tune neural model
* add augmentations
* add beam-search / language-model decoding
* evaluate on validation sessions
* visualize activations / interpret brain patterns

Everything starts with a working baseline — which this repo now has.