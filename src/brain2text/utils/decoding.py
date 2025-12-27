import torch

# CTC blank token
BLANK_ID = 0


def greedy_decoder(logits, token_map):
    """
    Greedy CTC decoding.

    logits: [T, C]  (or [1, T, C])
    token_map: dict {index -> phoneme}

    Returns: string of phonemes separated by spaces
    """

    # If batched, take first element
    if logits.dim() == 3:
        logits = logits[0]

    # Step 1: select highest-probability class at each timestep
    pred = torch.argmax(logits, dim=-1)

    # Step 2: collapse repeats + remove blanks
    result = []
    prev = None

    for idx in pred.tolist():
        if idx != prev and idx != BLANK_ID:
            result.append(idx)
        prev = idx

    # Step 3: map indices to phoneme symbols
    phonemes = [token_map.get(i, "?") for i in result]

    return " ".join(phonemes)
