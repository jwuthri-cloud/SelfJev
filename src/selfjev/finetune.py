"""Fine-tune the best recipe (Qwen3.5-4B + shared-prefix tree + LoRA) on your own data, then RLCD on top.

  selfjev finetune --data train.jsonl --out runs/mine [--val val.jsonl] [--init weights/selfjev_4b]
  selfjev rlcd     --data train.jsonl --out runs/mine_rlcd --init runs/mine/adapter [--reward log=1,brier=1,spherical=1]

Data: JSONL, one question per line in the eval format: {"state": "...", "question": {"type": "binary" | "multiclass" |
"multilabel", "instruction": "...", "candidates": [{"id": ..., "description": ...}]}, "target": true | "id" | ["id", ...]}
("id" and "family" optional). Every option is listed in the question text (options.py) unless
--no-options-in-question; serve the result the same way (--options-in-question). Needs one CUDA GPU (AGENTS.md: never
the laptop). Writes <out>/adapter (best validation), <out>/adapter_last and <out>/train_meta.json.

finetune: cross-entropy on the targets (the log score), the recipe of weights/qwen35_4b_tree and weights/selfjev_4b.
rlcd: what Jev calls RLCD ("Reinforcement Learning for Calibrated Decisions"). Here it is supervised training on
proper scoring rules, not RL: no environment, and the grade of every answer is known from the label. A question's
probabilities are the model's report; the reward is
a weighted sum of strictly proper scoring rules of that report against the target (log, Brier, spherical), so the
policy that maximizes it reports calibrated probabilities. Each step samples --samples reports per question from a
Gaussian around the model's logits (sd --sigma), weights their log-density by the reward minus the question's mean
reward (a policy gradient with a per-question baseline), and adds --beta x KL to the starting model's probabilities.
"accuracy" (the argmax decision is right) can be mixed in; it is not proper, so keep a proper term next to it.

Soft targets (both modes): a row may carry "soft", a teacher's probabilities (P(yes) for binary, {candidate id: p}
otherwise; scripts/jev_soft_targets.py writes Jev's). Training then scores against (1 - --soft-weight) x the label +
--soft-weight x "soft" (default 0.5: the label still decides, the teacher only says how sure to be); validation stays
on the labels.
"""
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
PROPER = {  # p: [..., labels, outcomes] reported distributions, y: target outcome distributions -> [..., labels]
    "log": lambda p, y: (y * p.clamp_min(1e-6).log()).sum(-1),  # expected under y: right for soft targets too
    "brier": lambda p, y: -((p - y) ** 2).sum(-1),
    "spherical": lambda p, y: (p * y).sum(-1) / p.norm(dim=-1),
    "accuracy": lambda p, y: (p.argmax(-1) == y.argmax(-1)).float(),  # not proper: mix with a proper term
    # a cost, not proper: -1 for a decision made with confidence >= 0.9 that the target says is wrong (expected under a
    # soft target), so a weight of 5 makes a confident mistake cost 5x; non-differentiable, the sampled reports handle it
    "confident_miss": lambda p, y: -(p.max(-1).values >= 0.9).float() * (1 - (y * F.one_hot(p.argmax(-1), p.shape[-1])).sum(-1)),
}


def load_rows(path, options_in_question=True):
    from .options import with_options
    rows = [json.loads(line) for line in (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("family", "data")
        if options_in_question:  # seeded by the row's own id, as scripts/options_in_question.py built data/ova/
            r["question"] = with_options(r["question"], str(r.get("id", i)))
        r["id"] = f"{r.get('id', 'q')}#{i}"  # unique even if the file repeats ids
    return rows


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


def log_loss(scores, items):
    """Cross-entropy of each question's report against its target, soft or hard (= grouped_loss on hard targets)."""
    total, j = scores.new_zeros(()), 0
    for it in items:
        s = scores[j:j + len(it["ids"])]
        j += len(it["ids"])
        lp = s.log_softmax(-1)[None] if it["type"] == "multiclass" else torch.stack([F.logsigmoid(s), F.logsigmoid(-s)], -1)
        total = total - (onehot(it, len(s), s.device) * lp).sum(-1).mean()
    assert j == len(scores), "scores and items are misaligned"
    return total


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
        s = scores[j:j + len(it["ids"])]
        j += len(it["ids"])
        z = s.detach() + sigma * torch.randn(samples, len(s), device=s.device)  # sampled reports
        with torch.no_grad():
            r = reward(z, it, weights)
        logp = -((z - s) ** 2).sum(-1) / (2 * sigma ** 2)  # log N(z; s, sigma^2) + const
        total = total - ((r - r.mean()) * logp).mean() + beta * kl(s, torch.tensor(it["ref"], device=s.device), it["type"])
    assert j == len(scores), "scores and items are misaligned"
    return total


def metrics(items, scores):
    """Accuracy, cross-entropy, Brier, 10-bin ECE and confidently-wrong decisions (confidence >= 0.9)."""
    from .train import grouped_loss, question_correct
    loss = correct = brier = 0.0
    conf, hit = [], []
    for it, s in zip(items, scores):
        t = torch.tensor(s, dtype=torch.float)
        loss += float(grouped_loss(t, [it]))
        correct += question_correct(s, it)
        p, y = report(t, it["type"]), onehot(it, len(s))
        brier += float(((p - y) ** 2).sum(-1).mean())
        c, h = p.max(-1).values, p.argmax(-1) == y.argmax(-1)
        conf += c.tolist()
        hit += h.tolist()
    bins = [min(int(10 * c), 9) for c in conf]
    ece = sum(abs(sum(c - h for c, h, b in zip(conf, hit, bins) if b == k)) for k in range(10)) / len(conf)
    n = len(items)
    return {"loss": loss / n, "accuracy": correct / n, "brier": brier / n, "ece": ece, "n": n,
            "confidently_wrong": sum(c >= 0.9 and not h for c, h in zip(conf, hit))}


def score_items(sc, items, roots, budget):
    from . import qwen35_tree
    from .train_tree import group_by_state, micro_batches, trees_for
    out = {}
    with torch.no_grad():
        for b in micro_batches(items, roots, budget, random.Random(0)):
            chunk = group_by_state([items[i] for i in b])
            s, k = qwen35_tree.score(sc, trees_for(chunk, roots)).float().tolist(), 0
            for it in chunk:
                out[it["id"]], k = s[k:k + len(it["ids"])], k + len(it["ids"])
    return [out[it["id"]] for it in items]


def train(mode, data, out, val=None, init=None, base="qwen35_4b", options_in_question=True, epochs=1, lr=None, lora_r=64,
          max_length=8192, batch_tokens=8192, grad_accum=4, eval_every=150, seed=13, reward_weights=None, samples=8,
          sigma=0.3, beta=0.05, soft_weight=0.5):
    from peft import LoraConfig, PeftModel, get_peft_model

    from . import qwen35_tree
    from .challengers import ChallengerScorer
    from .train import grouped_loss
    from .train_tree import group_by_state, micro_batches, trees_for
    assert mode in ("finetune", "rlcd") and (mode == "finetune" or init), "rlcd starts from a fine-tuned adapter (--init)"
    lr = lr or (2e-4 if mode == "finetune" else 5e-5)
    reward_weights = reward_weights or {"log": 1.0, "brier": 1.0, "spherical": 1.0}
    torch.manual_seed(seed)
    rng, out = random.Random(seed), Path(out)
    rows = load_rows(data, options_in_question)
    if val:
        vrows = load_rows(val, options_in_question)
    else:  # hold out 5% (at most 1,000 questions) for validation
        rng.shuffle(rows)
        k = min(1000, max(1, len(rows) // 20))
        vrows, rows = rows[:k], rows[k:]
    sc = ChallengerScorer(base)
    tr, roots, dropped = qwen35_tree.encode_items(sc, rows, max_length)
    va, vroots, vdropped = qwen35_tree.encode_items(sc, vrows, max_length)
    soft = {r["id"]: r["soft"] for r in rows if "soft" in r}
    for it in tr:  # training items only: validation keeps scoring against the labels
        if it["id"] in soft:
            it["y"] = soft_target(it, soft[it["id"]], soft_weight)
    if init:
        sc.model = PeftModel.from_pretrained(sc.model, init, is_trainable=True)
    else:
        found = {n.rsplit(".", 1)[-1] for n, m in sc.model.named_modules() if isinstance(m, torch.nn.Linear)}
        sc.model = get_peft_model(sc.model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                        target_modules=sorted(set(LINEAR) & found), bias="none"))
    if mode == "rlcd":  # the starting model's logits, for the KL penalty
        sc.model.eval()
        for it, s in zip(tr, score_items(sc, tr, roots, batch_tokens)):
            it["ref"] = s
    params = [p for p in sc.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = math.ceil(len(micro_batches(tr, roots, batch_tokens, random.Random(seed))) / grad_accum) * epochs
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    select = "loss" if mode == "finetune" else "brier"  # the objective's own proper score on validation
    meta = {"mode": mode, "base": sc.meta["model"], "revision": sc.meta["revision"], "prompt": sc.meta["prompt"],
            "init": init, "options_in_question": options_in_question, "data": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [data] + ([val] if val else [])}, "train_questions": len(tr), "val_questions": len(va),
            "dropped_over_max_length": {"train": dropped, "val": vdropped}, "max_length": max_length, "epochs": epochs,
            "lr": lr, "lora_r": lora_r, "batch_tokens": batch_tokens, "grad_accum": grad_accum, "steps": steps, "seed": seed,
            "select_by": f"validation {select}", "soft_targets": sum("y" in it for it in tr), "soft_weight": soft_weight, "log": []}
    if mode == "rlcd":
        meta["rlcd"] = {"reward": reward_weights, "samples": samples, "sigma": sigma, "beta": beta}
    out.mkdir(parents=True, exist_ok=True)
    best, step, start = math.inf, 0, time.perf_counter()

    def checkpoint():
        nonlocal best
        sc.model.eval()
        v = metrics(va, score_items(sc, va, vroots, batch_tokens))
        meta["log"].append({"step": step, "validation": v, "wall_s": time.perf_counter() - start})
        print("VALIDATION", step, json.dumps({k: round(x, 4) for k, x in v.items()}), flush=True)
        sc.model.save_pretrained(out / "adapter_last")
        if v[select] < best:
            best = v[select]
            sc.model.save_pretrained(out / "adapter")
            meta["best"] = {"step": step, **v}
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1))

    print("TRAIN", mode, json.dumps({k: meta[k] for k in ("train_questions", "val_questions", "steps")}), flush=True)
    checkpoint()
    for _ in range(epochs):
        batches = micro_batches(tr, roots, batch_tokens, rng)
        for b0 in range(0, len(batches), grad_accum):
            group = batches[b0:b0 + grad_accum]
            nq = sum(len(b) for b in group)
            sc.model.train()
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for b in group:
                its = group_by_state([tr[i] for i in b])
                s = qwen35_tree.score(sc, trees_for(its, roots)).float()
                fit = log_loss if soft else grouped_loss
                loss = (fit(s, its) if mode == "finetune" else rlcd_loss(s, its, reward_weights, samples, sigma, beta)) / nq
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                print("STEP", step, "/", steps, "loss", round(total, 4), "elapsed_s", round(time.perf_counter() - start), flush=True)
            if step % eval_every == 0 and step < steps:
                checkpoint()
    checkpoint()
    return meta
