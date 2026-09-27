"""LoRA adaptation that keeps the reranker's own yes/no scorer: s = z_yes - z_no, no new heads.

Losses use the exact inference formatting and score extraction:
  binary / multilabel: BCEWithLogits(s, y), averaged over the question's own labels
  multiclass:          CrossEntropy over the question's own candidate scores (never across questions)
Each question contributes equally (per-question mean, then mean over questions in the optimizer step), so a
question with many labels does not dominate. Micro-batches hold whole questions, so no candidate padding is
needed; token padding is masked by attention as at inference. The base checkpoint stays untouched on disk.
"""
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import peft
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model

from .data import load, sha256_file
from .formatting import DEFAULT_PROMPT, prompt_sha, question_pairs
from .model import MODEL_ID, MODEL_REVISION, Scorer, sync
from .schemas import parse_question

DEFAULTS = {
    "train_files": ["data/hf.jsonl"], "val_files": ["data/hf.jsonl"], "out_dir": "runs/lora", "seed": 13,
    "model_id": MODEL_ID, "revision": MODEL_REVISION, "prompt": DEFAULT_PROMPT, "device": None, "dtype": "float32", "max_length": 2048, "max_batch_tokens": 8192, "grad_accum": 4, "epochs": 1,
    "max_steps": None, "lr": 2e-4, "warmup_ratio": 0.03, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "gradient_checkpointing": True, "eval_every": 100, "max_val_questions": 800, "max_train_questions": None, "max_train_per_family": None, "cap_exempt_families": [],
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}


def encode_items(scorer, examples, prompt=DEFAULT_PROMPT):
    """-> (items, dropped): one item per question with its pair token ids and a type-specific target.
    Questions with any pair over max_length are dropped and counted (never truncated)."""
    texts, spans = [], []
    for ex in examples:
        q = parse_question({"id": "q", **ex["question"]})
        pairs = question_pairs(q, ex["state"], prompt)
        spans.append((ex, q, len(texts), len(pairs)))
        texts += pairs
    ids = scorer.encode(texts, check=False)
    items, dropped = [], Counter()
    for ex, q, k, n in spans:
        qids = ids[k:k + n]
        if max(map(len, qids)) > scorer.max_length:
            dropped[ex["family"]] += 1
            continue
        cids = [c.id for c in q.candidates]
        target = {"binary": lambda t: t, "multiclass": cids.index, "multilabel": lambda t: [c in t for c in cids]}[q.type](ex["target"])
        items.append({"id": ex["id"], "family": ex["family"], "type": q.type, "ids": qids, "target": target})
    return items, dict(dropped)


def shuffle_candidates(item, rng):
    """Random candidate order with the target remapped. Pairs are scored independently and the losses are
    permutation-invariant, so this cannot change the math; it keeps target-mapping bugs from hiding."""
    if item["type"] == "binary":
        return item
    perm = list(range(len(item["ids"])))
    rng.shuffle(perm)
    t = item["target"]
    return item | {"ids": [item["ids"][i] for i in perm],
                   "target": perm.index(t) if item["type"] == "multiclass" else [t[i] for i in perm]} \
        | {key: [item[key][i] for i in perm] for key in ("content", "teacher_scores", "candidate_ids") if key in item}


def grouped_loss(s, items):
    """Sum of per-question losses; s holds the items' pair scores concatenated in order."""
    total, k = s.new_zeros(()), 0
    for it in items:
        x = s[k:k + len(it["ids"])]
        k += len(it["ids"])
        if it["type"] == "multiclass":
            total = total + F.cross_entropy(x[None], torch.tensor([it["target"]], device=s.device))
        else:
            y = torch.tensor(it["target"] if it["type"] == "multilabel" else [it["target"]], dtype=s.dtype, device=s.device)
            total = total + F.binary_cross_entropy_with_logits(x, y)
    assert k == len(s), "scores and items are misaligned"
    return total


def micro_batches(items, max_batch_tokens, rng, bucket=512):
    """Whole questions per micro-batch (grouped losses need every candidate together), length-bucketed."""
    order = list(range(len(items)))
    rng.shuffle(order)
    out = []
    for c in range(0, len(order), bucket):
        chunk = sorted(order[c:c + bucket], key=lambda i: max(map(len, items[i]["ids"])))
        cur, pairs, width = [], 0, 0
        for i in chunk:
            n, w = len(items[i]["ids"]), max(map(len, items[i]["ids"]))
            if cur and (pairs + n) * max(width, w) > max_batch_tokens:
                out.append(cur)
                cur, pairs, width = [], 0, 0
            cur, pairs, width = cur + [i], pairs + n, max(width, w)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def question_correct(s, it):
    if it["type"] == "binary":
        return (s[0] >= 0) == it["target"]
    if it["type"] == "multiclass":
        return max(range(len(s)), key=s.__getitem__) == it["target"]
    return all((v >= 0) == y for v, y in zip(s, it["target"]))


def validate(scorer, items):
    scorer.model.eval()
    flat, _ = scorer.score_ids([x for it in items for x in it["ids"]])
    with torch.no_grad():
        loss = grouped_loss(torch.tensor(flat), items).item() / len(items)
    k, correct, by_type = 0, [], Counter()
    for it in items:
        s = flat[k:k + len(it["ids"])]
        k += len(it["ids"])
        correct.append(question_correct(s, it))
        by_type[it["type"], correct[-1]] += 1
    acc = {t: by_type[t, True] / (by_type[t, True] + by_type[t, False]) for t in ("binary", "multiclass", "multilabel")
           if by_type[t, True] + by_type[t, False]}
    return {"loss": loss, "question_accuracy": sum(correct) / len(correct), "accuracy_by_type_at_T1_t0.5": acc}, flat


def lora_targets(model, wanted):
    linear = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
    if missing := set(wanted) - set(linear):
        raise ValueError(f"LoRA targets {sorted(missing)} not in checkpoint; linear module names: {linear}")
    return linear


def select_data(cfg, rng):
    """Train and validation examples; same seed and config -> same questions (shared by train_tree and run_qwen35)."""
    train_ex = load(cfg["train_files"], {"train"})
    if cap := cfg["max_train_per_family"]:  # data-mixture control: no single family dominates the pilot
        fams, exempt = {}, set(cfg.get("cap_exempt_families") or [])  # exempt: e.g. an added family for a data curve
        for ex in train_ex:
            fams.setdefault(ex["family"], []).append(ex)
        train_ex = [ex for f in sorted(fams) for ex in (rng.sample(fams[f], cap) if len(fams[f]) > cap and f not in exempt else fams[f])]
    if k := cfg["max_train_questions"]:  # nested subsets from one fixed shuffle (25% of the data is inside the 50%)
        order = list(range(len(train_ex)))
        random.Random(cfg["seed"]).shuffle(order)
        train_ex = [train_ex[i] for i in sorted(order[:k])]
    val_ex = load(cfg["val_files"], {"validation"})
    return train_ex, rng.sample(val_ex, min(cfg["max_val_questions"], len(val_ex)))


def hardware(device):
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if platform.system() == "Darwin" else platform.processor()
    return {"device": device, "chip": chip, "platform": platform.platform()}


def train(config_path=None, **overrides):
    cfg = DEFAULTS | (json.loads(Path(config_path).read_text()) if config_path else {}) | overrides
    cfg["lora"] = DEFAULTS["lora"] | cfg.get("lora", {})
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t_start = time.perf_counter()

    scorer = Scorer(device=cfg["device"], dtype=cfg["dtype"], max_length=cfg["max_length"], max_batch_tokens=cfg["max_batch_tokens"],
                    model_id=cfg["model_id"], revision=cfg["revision"])
    linear = lora_targets(scorer.model, cfg["lora"]["target_modules"])
    lcfg = cfg["lora"]
    scorer.model = model = get_peft_model(scorer.model, LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                                                                   target_modules=lcfg["target_modules"], bias="none"))
    trainable, total = model.get_nb_trainable_parameters()
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"LoRA on {lcfg['target_modules']} (available: {linear}); trainable {trainable:,} / {total:,} params", flush=True)

    train_ex, val_ex = select_data(cfg, rng)
    train_items, dropped_train = encode_items(scorer, train_ex, cfg["prompt"])
    val_items, dropped_val = encode_items(scorer, val_ex, cfg["prompt"])
    steps_per_epoch = math.ceil(len(micro_batches(train_items, cfg["max_batch_tokens"], random.Random(0))) / cfg["grad_accum"])
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    warmup = max(1, round(cfg["warmup_ratio"] * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))
    print(f"train questions {len(train_items)} (dropped overlength {dropped_train}), val {len(val_items)}; "
          f"{total_steps} optimizer steps, warmup {warmup}", flush=True)

    log, step, seen = [], 0, 0
    v0, _ = validate(scorer, val_items)
    best = {"step": 0, **v0}
    log.append({"step": 0, "val": v0})
    print(f"step 0 (unmodified base + zero-init LoRA) val {v0}", flush=True)
    model.save_pretrained(out / "adapter")
    ref_scores = validate(scorer, val_items[:64])[1]
    window, t_last = [], time.perf_counter()
    done = False
    for epoch in range(cfg["epochs"] if not cfg["max_steps"] else 10**9):
        batches = micro_batches(train_items, cfg["max_batch_tokens"], rng)
        for g in range(0, len(batches), cfg["grad_accum"]):
            group = batches[g:g + cfg["grad_accum"]]
            nq = sum(len(b) for b in group)
            model.train()
            step_loss = 0.0
            for b in group:
                items = [shuffle_candidates(train_items[i], rng) for i in b]
                s = scorer.forward([x for it in items for x in it["ids"]])
                loss = grouped_loss(s, items) / nq
                loss.backward()
                step_loss += loss.item()
            grads = [p.grad for p in params if p.grad is not None]
            if not grads or not all(torch.isfinite(gr).all() for gr in grads):
                raise FloatingPointError(f"non-finite or missing LoRA gradients at step {step}")
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step, seen = step + 1, seen + nq
            window.append(step_loss)
            if step % 10 == 0:
                sync(scorer.device)
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                print(f"step {step}/{total_steps} epoch {epoch} loss {sum(window) / len(window):.4f} gnorm {gnorm:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {10 / dt:.2f} steps/s", flush=True)
                log.append({"step": step, "train_loss": sum(window) / len(window), "grad_norm": gnorm, "questions_seen": seen})
                window = []
            if step % cfg["eval_every"] == 0 or step == total_steps:
                v, _ = validate(scorer, val_items)
                log.append({"step": step, "val": v})
                print(f"step {step} val {v}", flush=True)
                if v["loss"] < best["loss"]:
                    best = {"step": step, **v}
                    model.save_pretrained(out / "adapter")
                    ref_scores = validate(scorer, val_items[:64])[1]
            if step == total_steps:
                done = True
                break
        if done:
            break
    if not any(e.get("step") == step and "val" in e for e in log):  # data ran out before total_steps: validate the end
        v, _ = validate(scorer, val_items)
        log.append({"step": step, "val": v})
        print(f"step {step} (end of data) val {v}", flush=True)
        if v["loss"] < best["loss"]:
            best = {"step": step, **v}
            model.save_pretrained(out / "adapter")
            ref_scores = validate(scorer, val_items[:64])[1]

    # Reload the saved best adapter from disk and check it reproduces the in-memory scores.
    # Same dtype AND batch budget as the reference: in bf16, batch composition alone moves scores by ~0.1-0.2.
    reloaded = Scorer(adapter=out / "adapter", device=scorer.device, dtype=cfg["dtype"], max_length=cfg["max_length"],
                      model_id=cfg["model_id"], revision=cfg["revision"],
                      max_batch_tokens=cfg["max_batch_tokens"])
    re_scores, _ = reloaded.score_ids([x for it in val_items[:64] for x in it["ids"]])
    reload_diff = max(abs(a - b) for a, b in zip(ref_scores, re_scores))
    meta = {
        "config": cfg, "base": {"model": scorer.meta["model"], "revision": scorer.meta["revision"]},
        "prompt": {"name": cfg["prompt"], "sha": prompt_sha(cfg["prompt"])},
        "lora": lcfg | {"trainable_params": trainable, "total_params": total, "linear_modules_available": linear},
        "data": {"train_files": [{"path": p, "sha256": sha256_file(p)} for p in cfg["train_files"]],
                 "val_files": [{"path": p, "sha256": sha256_file(p)} for p in cfg["val_files"]],
                 "train_questions": len(train_items), "val_questions": len(val_items),
                 "train_by_family": dict(Counter(it["family"] for it in train_items)),
                 "train_by_type": dict(Counter(it["type"] for it in train_items)),
                 "dropped_overlength": {"train": dropped_train, "val": dropped_val}},
        "best": best, "steps": step, "log": log, "reload_check": {"questions": 64, "max_abs_score_diff": reload_diff},
        "hardware": hardware(scorer.device),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "wall_s": time.perf_counter() - t_start, "adapter_sha256": sha256_file(out / "adapter/adapter_model.safetensors")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"best step {best['step']} val loss {best['loss']:.4f}; reload max |diff| {reload_diff:.2e}; saved {out / 'adapter'}", flush=True)
    return meta
