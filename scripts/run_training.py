from brain2text.training.train_ctc import train_ctc

train_ctc(
    data_root="data/raw",
    epochs=2,          # start small
    batch_size=16,
    lr=1e-3,
    save_path="checkpoints/ctc_baseline.pth",
)
