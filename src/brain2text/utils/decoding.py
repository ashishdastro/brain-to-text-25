import torch
import math
import numpy as np
from collections import defaultdict

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

def beam_search_decode_ctc(log_probs, lengths, token_map, beam_width=10, blank_id=0):
    """
    log_probs: [B, T, V]   (log-softmax from model)
    lengths:   list[int]   valid time-steps per sample
    token_map: list[str]   index -> symbol
    """
    results = []

    for b in range(log_probs.size(0)):
        T = lengths[b]
        probs = log_probs[b, :T]   # [T, V]

        # beam: dict[prefix(str)] -> (prob_blank, prob_non_blank)
        beam = { "": (0.0, -math.inf) }   # log(1)=0 for blank path

        for t in range(T):
            next_beam = defaultdict(lambda: (-math.inf, -math.inf))

            for prefix, (p_b, p_nb) in beam.items():
                for v in range(probs.size(1)):
                    p = probs[t, v].item()

                    if v == blank_id:
                        # extend with blank (stays same prefix)
                        nb_b, nb_nb = next_beam[prefix]
                        nb_b = np.logaddexp(nb_b, p_b + p)
                        nb_b = np.logaddexp(nb_b, p_nb + p)
                        next_beam[prefix] = (nb_b, nb_nb)

                    else:
                        c = token_map[v]
                        new_prefix = prefix + c

                        # case 1: repeated char continues
                        if len(prefix) > 0 and prefix[-1] == c:
                            nb_b, nb_nb = next_beam[prefix]
                            nb_nb = np.logaddexp(nb_nb, p_b + p)
                            next_beam[prefix] = (nb_b, nb_nb)

                        # case 2: add new char
                        nb_b, nb_nb = next_beam[new_prefix]
                        nb_nb = np.logaddexp(nb_nb, p_b + p)
                        nb_nb = np.logaddexp(nb_nb, p_nb + p)
                        next_beam[new_prefix] = (nb_b, nb_nb)

            # prune to beam width
            beam = dict(
                sorted(
                    next_beam.items(),
                    key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                    reverse=True
                )[:beam_width]
            )

        # pick best final sequence
        best = max(
            beam.items(),
            key=lambda kv: np.logaddexp(kv[1][0], kv[1][1])
        )[0]

        results.append(best)

    return results
