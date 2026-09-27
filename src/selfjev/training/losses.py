"""Per-question losses and correctness over concatenated candidate scores."""

import torch
import torch.nn.functional as F


def grouped_loss(s, items):
    """Sum of per-question losses; s holds the items' pair scores concatenated in order."""
    total, k = s.new_zeros(()), 0
    for it in items:
        x = s[k : k + len(it["ids"])]
        k += len(it["ids"])
        if it["type"] == "multiclass":
            total = total + F.cross_entropy(x[None], torch.tensor([it["target"]], device=s.device))
        else:
            y = torch.tensor(it["target"] if it["type"] == "multilabel" else [it["target"]], dtype=s.dtype, device=s.device)
            total = total + F.binary_cross_entropy_with_logits(x, y)
    assert k == len(s), "scores and items are misaligned"
    return total


def question_correct(s, it):
    if it["type"] == "binary":
        return (s[0] >= 0) == it["target"]
    if it["type"] == "multiclass":
        return max(range(len(s)), key=s.__getitem__) == it["target"]
    return all((v >= 0) == y for v, y in zip(s, it["target"]))
