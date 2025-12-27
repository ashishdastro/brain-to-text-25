import os
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils


def temporal_mask(x, mask_percentage=0.1, mask_value=0.0):
    """
    Randomly masks a percentage of timesteps in a sequence.
    x: tensor [T, F]
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    seq_len = x.shape[0]
    num_mask = int(seq_len * mask_percentage)

    if num_mask > 0:
        idx = torch.randperm(seq_len)[:num_mask]
        x[idx, :] = mask_value

    return x


class BrainDataset(Dataset):
    """
    Dataset for Brain-to-Text HDF5 trials.

    Each trial contains:
        - input_features  [T, 512]
        - seq_class_ids   [L]  (only train/val)
    """

    def __init__(
        self,
        file_path,
        input_key="input_features",
        target_key="seq_class_ids",
        is_test=False,
        use_augmentation=False,
    ):
        self.file_path = file_path
        self.input_key = input_key
        self.target_key = target_key
        self.is_test = is_test
        self.use_augmentation = use_augmentation

        self.file = None

        # list trials
        if os.path.exists(file_path):
            with h5py.File(file_path, "r") as f:
                self.trial_keys = sorted(list(f.keys()))
        else:
            print(f"[WARN] Missing file: {file_path}")
            self.trial_keys = []

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        trial_key = self.trial_keys[idx]
        trial = self.file[trial_key]

        x = torch.tensor(trial[self.input_key][:], dtype=torch.float32)

        if self.use_augmentation and not self.is_test:
            x = temporal_mask(x, mask_percentage=0.1)

        if self.target_key in trial and not self.is_test:
            y = torch.tensor(trial[self.target_key][:], dtype=torch.long)
        else:
            y = torch.tensor([], dtype=torch.long)

        if self.is_test:
            return x, y, trial_key

        return x, y


def ctc_collate(batch):
    """
    Pads variable-length sequences for CTC training/inference.

    Train/Val batch items: (x, y)
    Test batch items:      (x, y, key)
    """

    is_test = len(batch[0]) == 3

    if is_test:
        xs, ys, keys = zip(*batch)
    else:
        xs, ys = zip(*batch)

    x_lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    y_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)

    xs = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)

    if is_test:
        return xs, ys, x_lengths, y_lengths, keys

    return xs, ys, x_lengths, y_lengths
