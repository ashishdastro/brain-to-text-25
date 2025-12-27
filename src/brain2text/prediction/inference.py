import torch
import yaml
from brain2text.models.rnn import RecurrentModel
from brain2text.utils.decoding import greedy_decoder

CHECKPOINT_DIR = "data/raw/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint"
ARGS_PATH = f"{CHECKPOINT_DIR}/args.yaml"
CKPT_PATH = f"{CHECKPOINT_DIR}/best_checkpoint"

def load_pretrained_model(device="cpu"):

    with open(ARGS_PATH, "r") as f:
        args = yaml.safe_load(f)

    model = RecurrentModel(
        model_type=args.get("model_type", "RNN"),
        data_input_size=args.get("input_size", 256),
        adapter_output_size=args.get("input_size", 256),
        hidden_size=args.get("hidden_size", 512),
        output_size=args.get("output_size", 41),
        num_layers=args.get("num_layers", 1),
        bidirectional=args.get("bidirectional", False),
    )

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model = model.to(device)
    model.eval()

    return model

def predict_single_trial(model, x, token_map, device="cpu"):
    x = x.unsqueeze(0).to(device)   # [1, T, F]
    with torch.no_grad():
        logits = model(x)[0]        # [T, 41]
    return greedy_decoder(logits, token_map)
