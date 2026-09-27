"""The RLCD objective (what Jev calls "Reinforcement Learning for Calibrated Decisions"): supervised training on proper
scoring rules. A question's probabilities are the model's report; the reward is a weighted sum of strictly proper scores
of that report against the target (log, Brier, spherical), optionally with a cost per confident mistake. Each step
samples reports from a Gaussian around the logits, weights their log-density by the reward minus the question's mean
(a policy gradient with a per-question baseline), and adds beta x KL to the starting model. docs/finetune.md says what
this does and does not buy.
"""

import torch
import torch.nn.functional as F

PROPER = {  # p: [..., labels, outcomes] reported distributions, y: target outcome distributions -> [..., labels]
    "log": lambda p, y: (y * p.clamp_min(1e-6).log()).sum(-1),  # expected under y: right for soft targets too
    "brier": lambda p, y: -((p - y) ** 2).sum(-1),
    "spherical": lambda p, y: (p * y).sum(-1) / p.norm(dim=-1),
    "accuracy": lambda p, y: (p.argmax(-1) == y.argmax(-1)).float(),  # not proper: mix with a proper term
    # a cost, not proper: -1 for a decision made with confidence >= 0.9 that the target says is wrong (expected under a
    # soft target), so a weight of 5 makes a confident mistake cost 5x; non-differentiable, the sampled reports handle it
    "confident_miss": lambda p, y: -(p.max(-1).values >= 0.9).float() * (1 - (y * F.one_hot(p.argmax(-1), p.shape[-1])).sum(-1)),
}


def report(z, typ):
    """Logits [..., n] -> outcome distributions [..., labels, outcomes]: one choice over n options (multiclass), or n
    independent yes/no decisions (binary, multilabel)."""
    if typ == "multiclass":
        return z.softmax(-1).unsqueeze(-2)
    p = z.sigmoid()
    return torch.stack([p, 1 - p], -1)


def onehot(it, n, device=None):
    if "y" in it:  # soft target, set by train()
        return torch.tensor(it["y"], device=device)
    if it["type"] == "multiclass":
        return F.one_hot(torch.tensor(it["target"], device=device), n).float()[None]
    y = torch.tensor(it["target"] if it["type"] == "multilabel" else [it["target"]], dtype=torch.float, device=device)
    return torch.stack([y, 1 - y], -1)


def soft_target(it, soft, w):
    """The label's outcome distributions mixed with a teacher's probabilities (a row's "soft"), weight w on the teacher."""
    if it["type"] == "binary":
        t = torch.tensor([[soft, 1 - soft]])
    elif it["type"] == "multiclass":
        t = torch.tensor([[soft[c] for c in it["candidate_ids"]]])
        t = t / t.sum().clamp_min(1e-9)
    else:
        t = torch.tensor([[soft[c], 1 - soft[c]] for c in it["candidate_ids"]])
    return ((1 - w) * onehot(it, len(it["ids"])) + w * t).tolist()


def reward(z, it, weights):
    p, y = report(z, it["type"]), onehot(it, z.shape[-1], z.device)
    return sum(w * PROPER[k](p, y).mean(-1) for k, w in weights.items())


def kl(s, ref, typ):
    """KL(model || reference) between the two models' reported distributions for one question."""
    if typ == "multiclass":
        lp, lq = s.log_softmax(-1), ref.log_softmax(-1)
        return (lp.exp() * (lp - lq)).sum()
    return sum((F.logsigmoid(a).exp() * (F.logsigmoid(a) - F.logsigmoid(b))).sum() for a, b in ((s, ref), (-s, -ref)))


def rlcd_loss(scores, items, weights, samples=8, sigma=0.3, beta=0.05):
    """Sum over questions of -(policy-gradient surrogate) + beta * KL. scores: the items' candidate logits, in order."""
    total, j = scores.new_zeros(()), 0
    for it in items:
        s = scores[j : j + len(it["ids"])]
        j += len(it["ids"])
        z = s.detach() + sigma * torch.randn(samples, len(s), device=s.device)  # sampled reports
        with torch.no_grad():
            r = reward(z, it, weights)
        logp = -((z - s) ** 2).sum(-1) / (2 * sigma**2)  # log N(z; s, sigma^2) + const
        total = total - ((r - r.mean()) * logp).mean() + beta * kl(s, torch.tensor(it["ref"], device=s.device), it["type"])
    assert j == len(scores), "scores and items are misaligned"
    return total
