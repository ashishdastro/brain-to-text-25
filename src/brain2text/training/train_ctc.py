import os
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import glob

from brain2text.data.dataset import BrainDataset, ctc_collate
from brain2text.models.rnn import RecurrentModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_datasets(root_dir: str):
    """
    Scans all hdf5 session folders and builds combined datasets.
    """
    pattern = os.path.join(
        root_dir,
        "t15_copyTask_neuralData",
        "hdf5_data_final",
        "**",
        "data_*.hdf5",
    )

    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    train_sets, val_sets = [], []

    for f in files:
        if f.endswith("data_train.hdf5"):
            train_sets.append(
                BrainDataset(f, is_test=False, use_augmentation=True)
            )
        elif f.endswith("data_val.hdf5"):
            val_sets.append(
                BrainDataset(f, is_test=False, use_augmentation=False)
            )

    train = ConcatDataset(train_sets)
    val = ConcatDataset(val_sets)

    return train, val


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y, x_lens, y_lens in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_lens, y_lens = x_lens.to(DEVICE), y_lens.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)              # [B, T, C]
        logits = logits.permute(1, 0, 2)  # CTC expects [T, B, C]

        loss = criterion(logits, y, x_lens, y_lens)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y, x_lens, y_lens in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_lens, y_lens = x_lens.to(DEVICE), y_lens.to(DEVICE)

            logits = model(x)
            logits = logits.permute(1, 0, 2)

            loss = criterion(logits, y, x_lens, y_lens)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def train_ctc(
    data_root="data/raw",
    epochs=5,
    batch_size=32,
    lr=1e-3,
    save_path="checkpoints/best_model.pth",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds = build_datasets(data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate,
    )

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    model = RecurrentModel(model_type="GRU").to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ Saved new best model → {save_path}")

    return save_path
