"""Fine-tune the best recipe (Qwen3.5-4B + shared-prefix tree + LoRA) on your own data, then RLCD on top.

  selfjev finetune --data train.jsonl --out runs/mine [--val val.jsonl] [--init weights/selfjev_4b_vision]
  selfjev rlcd     --data train.jsonl --out runs/mine_rlcd --init runs/mine/adapter [--reward log=1,brier=1,spherical=1]

Data: JSONL, one question per line in the eval format: {"state": "...", "question": {"type": "binary" | "multiclass" |
"multilabel", "instruction": "...", "candidates": [{"id": ..., "description": ...}]}, "target": true | "id" | ["id", ...]}
("id" and "family" optional). "state" may be an image (a base64 data URL) or a list of text and image parts: the frozen
vision tower encodes it, only the language LoRA trains. Every option is listed in the question text (selfjev.core.options) unless
--no-options-in-question; selfjev serve and selfjev classify list them too (pass --no-options-in-question there for an
adapter trained without). Needs one CUDA GPU (AGENTS.md: never the laptop). Writes <out>/adapter (best validation),
<out>/adapter_last and <out>/train_meta.json.

finetune: cross-entropy on the targets (the log score), the recipe of weights/selfjev_4b.
rlcd: the same loop with the RLCD objective (selfjev.training.rlcd) instead of cross-entropy.

Soft targets (both modes): a row may carry "soft", a teacher's probabilities (P(yes) for binary, {candidate id: p}
otherwise; scripts/train/jev_soft_targets.py writes Jev's). Training then scores against (1 - --soft-weight) x the label +
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
from peft import LoraConfig, PeftModel, get_peft_model

from ..core.options import with_options
from ..engine import tree
from ..engine.qwen35 import Qwen35Scorer
from .batching import group_by_state, micro_batches, trees_for
from .losses import grouped_loss, question_correct
from .rlcd import onehot, report, rlcd_loss, soft_target

LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def load_rows(path, options_in_question=True):

    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("family", "data")
        if options_in_question:  # seeded by the row's own id, as scripts/data/options_in_question.py built data/ova/
            r["question"] = with_options(r["question"], str(r.get("id", i)))
        r["id"] = f"{r.get('id', 'q')}#{i}"  # unique even if the file repeats ids
    return rows


def log_loss(scores, items):
    """Cross-entropy of each question's report against its target, soft or hard (= grouped_loss on hard targets)."""
    total, j = scores.new_zeros(()), 0
    for it in items:
        s = scores[j : j + len(it["ids"])]
        j += len(it["ids"])
        lp = s.log_softmax(-1)[None] if it["type"] == "multiclass" else torch.stack([F.logsigmoid(s), F.logsigmoid(-s)], -1)
        total = total - (onehot(it, len(s), s.device) * lp).sum(-1).mean()
    assert j == len(scores), "scores and items are misaligned"
    return total


def metrics(items, scores):
    """Accuracy, cross-entropy, Brier, 10-bin ECE and confidently-wrong decisions (confidence >= 0.9)."""

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
    return {
        "loss": loss / n,
        "accuracy": correct / n,
        "brier": brier / n,
        "ece": ece,
        "n": n,
        "confidently_wrong": sum(c >= 0.9 and not h for c, h in zip(conf, hit)),
    }


def score_items(sc, items, roots, budget):

    out = {}
    with torch.no_grad():
        for b in micro_batches(items, roots, budget, random.Random(0)):
            chunk = group_by_state([items[i] for i in b])
            s, k = tree.score(sc, trees_for(chunk, roots)).float().tolist(), 0
            for it in chunk:
                out[it["id"]], k = s[k : k + len(it["ids"])], k + len(it["ids"])
    return [out[it["id"]] for it in items]


def train(
    mode,
    data,
    out,
    val=None,
    init=None,
    options_in_question=True,
    epochs=1,
    lr=None,
    lora_r=64,
    max_length=8192,
    batch_tokens=8192,
    grad_accum=4,
    eval_every=150,
    seed=13,
    reward_weights=None,
    samples=8,
    sigma=0.3,
    beta=0.05,
    soft_weight=0.5,
):

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
    sc = Qwen35Scorer()
    tr, roots, dropped = tree.encode_items(sc, rows, max_length)
    va, vroots, vdropped = tree.encode_items(sc, vrows, max_length)
    soft = {r["id"]: r["soft"] for r in rows if "soft" in r}
    for it in tr:  # training items only: validation keeps scoring against the labels
        if it["id"] in soft:
            it["y"] = soft_target(it, soft[it["id"]], soft_weight)
    if init:
        sc.model = PeftModel.from_pretrained(sc.model, init, is_trainable=True)
    else:
        found = {n.rsplit(".", 1)[-1] for n, m in sc.model.named_modules() if isinstance(m, torch.nn.Linear)}
        sc.model = get_peft_model(
            sc.model,
            LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=sorted(set(LINEAR) & found), bias="none"),
        )
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
    meta = {
        "mode": mode,
        "base": sc.meta["model"],
        "revision": sc.meta["revision"],
        "prompt": sc.meta["prompt"],
        "init": init,
        "options_in_question": options_in_question,
        "data": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [data] + ([val] if val else [])},
        "train_questions": len(tr),
        "val_questions": len(va),
        "dropped_over_max_length": {"train": dropped, "val": vdropped},
        "max_length": max_length,
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r,
        "batch_tokens": batch_tokens,
        "grad_accum": grad_accum,
        "steps": steps,
        "seed": seed,
        "select_by": f"validation {select}",
        "soft_targets": sum("y" in it for it in tr),
        "soft_weight": soft_weight,
        "log": [],
    }
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
            group = batches[b0 : b0 + grad_accum]
            nq = sum(len(b) for b in group)
            sc.model.train()
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for b in group:
                its = group_by_state([tr[i] for i in b])
                s = tree.score(sc, trees_for(its, roots)).float()
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
