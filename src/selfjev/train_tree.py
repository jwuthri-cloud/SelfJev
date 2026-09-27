"""LoRA training for the shared-prefix tree scorer (tree.py), with train.py's data selection, losses and schedule.

Each micro-batch is a set of trees, one per state, holding every question of that state in the batch, run in ONE
packed pass with the tree mask. Every leaf equals its standalone "state + question + candidate" sequence, so this
trains exactly the function tree inference computes, while each state is encoded once per micro-batch (and stays in
the autograd graph). Validation and the reload check use the inference path (root KV cache + branches).
"""

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import sha256_file
from .model import default_device, sync
from .schemas import parse_question
from .train import DEFAULTS as STOCK_DEFAULTS
from .train import grouped_loss, hardware, lora_targets, question_correct, select_data, shuffle_candidates
from .tree import RERANKER_4B, Encoder, TreeModel, TreeScorer, build_tree, leaf_answers

DEFAULTS = STOCK_DEFAULTS | {
    "model_id": RERANKER_4B[0],
    "revision": RERANKER_4B[1],
    "out_dir": "runs/tree_4b",
    "dtype": "bfloat16",
    "max_length": 2048,
    "max_batch_tokens": 16384,
    "grad_accum": 2,
    "epochs": 1,
    "lr": 2e-4,
    "warmup_ratio": 0.05,
    "eval_every": 50,
    "max_val_questions": 1200,
    "max_train_per_family": 1600,
}


def encode_items(enc, examples, max_length):
    """-> (items, roots, dropped). Item: one question with its question-segment ids ("q"), leaf ids ("ids") and
    target; "state" indexes roots (identical states share one). Questions whose longest leaf path (root + question
    + leaf) exceeds max_length are dropped and counted, never truncated."""
    keys, per = {}, []
    for ex in examples:
        q = parse_question({"id": "q", **ex["question"]})
        per.append((ex, q, keys.setdefault(ex["state"], len(keys))))
    roots = [enc.root(s) for s in enc.user(list(keys))]
    q_ids = enc.user([q.instruction for _, q, _ in per])
    a_ids = iter(enc.user([a for _, q, _ in per for a in leaf_answers(q)]))
    items, dropped = [], Counter()
    for (ex, q, s), qi in zip(per, q_ids):
        leaves = [enc.leaf(next(a_ids)) for _ in leaf_answers(q)]
        qseg = enc.question(qi)
        if len(roots[s]) + len(qseg) + max(map(len, leaves)) > max_length:
            dropped[ex["family"]] += 1
            continue
        cids = [c.id for c in q.candidates]
        target = {"binary": lambda t: t, "multiclass": cids.index, "multilabel": lambda t: [c in t for c in cids]}[q.type](ex["target"])  # noqa: B023 lambda called right away
        items.append(
            {
                "id": ex["id"],
                "family": ex["family"],
                "type": q.type,
                "state": s,
                "q": qseg,
                "ids": leaves,
                "target": target,
                "candidate_ids": cids,
            }
        )
    return items, roots, dict(dropped)


def group_by_state(items):
    """Stable reorder so each state's questions are contiguous: leaf order then matches item order."""
    order = {}
    for it in items:
        order.setdefault(it["state"], []).append(it)
    return [it for group in order.values() for it in group]


def trees_for(items, roots):
    groups = {}
    for it in items:
        groups.setdefault(it["state"], []).append(it)
    return [build_tree(roots[s], [(it["q"], it["ids"]) for it in g]) for s, g in groups.items()]


def tree_len(items, roots):
    return len(roots[items[0]["state"]]) + sum(len(it["q"]) + sum(map(len, it["ids"])) for it in items)


def micro_batches(items, roots, max_batch_tokens, rng, bucket=256):
    """Whole states (all their questions) per micro-batch, length-bucketed; padded packed tokens (trees x longest
    tree) stay under max_batch_tokens unless one tree alone exceeds it."""
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it["state"], []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    size = {k: tree_len([items[i] for i in groups[k]], roots) for k in keys}
    out = []
    for c in range(0, len(keys), bucket):
        cur, n, w = [], 0, 0
        for k in sorted(keys[c : c + bucket], key=size.__getitem__):
            if cur and (n + 1) * max(w, size[k]) > max_batch_tokens:
                out.append(cur)
                cur, n, w = [], 0, 0
            cur, n, w = cur + groups[k], n + 1, max(w, size[k])
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


@torch.no_grad()
def score_items(model, items, roots, max_batch_tokens, progress=None):
    """Inference path (root KV cache + branches) -> per-item score lists."""
    model.lm.eval()
    out = [None] * len(items)
    batches = micro_batches(items, roots, max_batch_tokens, random.Random(0))
    for batch_index, b in enumerate(batches):
        chunk = group_by_state([dict(items[i], _i=i) for i in b])
        s, k = model.cached(trees_for(chunk, roots)).tolist(), 0
        for it in chunk:
            out[it["_i"]], k = s[k : k + len(it["ids"])], k + len(it["ids"])
        if progress and ((batch_index + 1) % 10 == 0 or batch_index + 1 == len(batches)):
            progress(batch_index + 1, len(batches))
    return out


def validate(model, items, roots, max_batch_tokens):
    scores = score_items(model, items, roots, max_batch_tokens)
    loss = grouped_loss(torch.tensor([v for s in scores for v in s]), items).item() / len(items)
    by = Counter((it["type"], question_correct(s, it)) for s, it in zip(scores, items))
    acc = {t: by[t, True] / (by[t, True] + by[t, False]) for t in ("binary", "multiclass", "multilabel") if by[t, True] + by[t, False]}
    return {
        "loss": loss,
        "question_accuracy": sum(v for (t, ok), v in by.items() if ok) / len(items),
        "accuracy_by_type_at_T1_t0.5": acc,
    }, scores


def train(config_path=None, **overrides):
    cfg = DEFAULTS | (json.loads(Path(config_path).read_text()) if config_path else {}) | overrides
    cfg["lora"] = DEFAULTS["lora"] | cfg.get("lora", {})
    if cfg.get("teacher_weight"):  # 27B-teacher distillation was a dead end (tree_4b_ova_kd)
        raise ValueError("teacher distillation was removed; its code is at tag archive/pre-cleanup-2026-09-27")
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t_start, device = time.perf_counter(), cfg["device"] or default_device()

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    enc = Encoder(tok, cfg["model_id"], cfg.get("format_name", "tree-v1"))
    lm = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]))
    linear, lc = lora_targets(lm, cfg["lora"]["target_modules"]), cfg["lora"]
    if cfg.get("init_adapter"):
        lm = PeftModel.from_pretrained(lm, cfg["init_adapter"], is_trainable=True)
        actual = lm.peft_config["default"]
        if actual.r != lc["r"] or actual.lora_alpha != lc["alpha"] or set(actual.target_modules) != set(lc["target_modules"]):
            raise ValueError("Continuation adapter LoRA configuration differs from training configuration")
    else:
        lm = get_peft_model(
            lm, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias="none")
        )
    if cfg["gradient_checkpointing"]:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = TreeModel(lm.to(device), tok.pad_token_id)
    trainable, total = lm.get_nb_trainable_parameters()
    print(
        f"{cfg['model_id']} tree scorer, LoRA on {lc['target_modules']} (linear: {linear}); trainable {trainable:,} / {total:,}", flush=True
    )

    train_ex, val_ex = select_data(cfg, rng)
    train_items, train_roots, dropped_train = encode_items(enc, train_ex, cfg["max_length"])
    val_items, val_roots, dropped_val = encode_items(enc, val_ex, cfg["max_length"])
    (out / "selected_examples.json").write_text(
        json.dumps({"train": [it["id"] for it in train_items], "validation": [it["id"] for it in val_items]})
    )
    per_epoch = math.ceil(len(micro_batches(train_items, train_roots, cfg["max_batch_tokens"], random.Random(0))) / cfg["grad_accum"])
    total_steps = cfg["max_steps"] or per_epoch * cfg["epochs"]
    warmup = max(1, round(cfg["warmup_ratio"] * total_steps))
    params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    )
    print(
        f"train questions {len(train_items)} over {len(train_roots)} states (dropped {dropped_train}), val {len(val_items)}; "
        f"{total_steps} optimizer steps ({per_epoch}/epoch), warmup {warmup}",
        flush=True,
    )

    log, best, ref, step, seen = [], {"step": -1, "loss": math.inf}, None, 0, 0
    reload_items = val_items[:64]

    def evaluate(tag):
        nonlocal best, ref
        v, _ = validate(model, val_items, val_roots, cfg["max_batch_tokens"])  # noqa: F821 only called before the del
        log.append({"step": step, "stage": tag, "val": v})
        print(f"step {step} ({tag}) val {v}", flush=True)
        if v["loss"] < best["loss"]:
            best = {"step": step, **v}
            lm.save_pretrained(out / "adapter", save_embedding_layers=False)  # noqa: F821 only called before the del
            (out / "adapter/tree_format.json").write_text(json.dumps(enc.format, indent=2))
            ref = [x for s in score_items(model, reload_items, val_roots, cfg["max_batch_tokens"]) for x in s]  # noqa: F821 only called before the del

    evaluate("init (continued adapter)" if cfg.get("init_adapter") else "init (zero-initialized LoRA)")
    window, t_last, done = [], time.perf_counter(), False
    for epoch in range(cfg["epochs"] if not cfg["max_steps"] else 10**9):
        batches = micro_batches(train_items, train_roots, cfg["max_batch_tokens"], rng)
        for g in range(0, len(batches), cfg["grad_accum"]):
            group = batches[g : g + cfg["grad_accum"]]
            nq = sum(len(b) for b in group)
            lm.train()
            step_loss = 0.0
            for b in group:
                items = group_by_state([shuffle_candidates(train_items[i], rng) for i in b])
                scores = model.packed(trees_for(items, train_roots))
                loss = grouped_loss(scores, items) / nq
                loss.backward()
                step_loss += loss.item()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError(f"missing or non-finite LoRA gradients at step {step}")
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step, seen = step + 1, seen + nq
            window.append(step_loss)
            if step % 10 == 0:
                sync(device)
                dt, t_last = time.perf_counter() - t_last, time.perf_counter()
                print(
                    f"step {step}/{total_steps} epoch {epoch} loss {sum(window) / len(window):.4f} gnorm {gnorm:.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {10 / dt:.2f} steps/s",
                    flush=True,
                )
                log.append({"step": step, "train_loss": sum(window) / len(window), "grad_norm": gnorm, "questions_seen": seen})
                window = []
            if step % cfg["eval_every"] == 0 or step == total_steps:
                evaluate("train")
            if step == total_steps:
                done = True
                break
        if done:
            break
    if log[-1].get("step") != step or "val" not in log[-1]:
        evaluate("end of data")

    del lm, model  # reload the saved adapter from disk into a fresh scorer: same scores, same batching
    torch.cuda.empty_cache() if device == "cuda" else None
    re = TreeScorer(cfg["model_id"], cfg["revision"], adapter=out / "adapter", device=device, dtype=cfg["dtype"])
    again = [x for s in score_items(re.model, reload_items, val_roots, cfg["max_batch_tokens"]) for x in s]
    meta = {
        "config": cfg,
        "base": {"model": cfg["model_id"], "revision": cfg["revision"]},
        "format": enc.format,
        "init_adapter_sha256": sha256_file(Path(cfg["init_adapter"]) / "adapter_model.safetensors") if cfg.get("init_adapter") else None,
        "lora": lc | {"trainable_params": trainable, "total_params": total, "linear_modules_available": linear},
        "data": {
            "train_files": [{"path": p, "sha256": sha256_file(p)} for p in cfg["train_files"]],
            "val_files": [{"path": p, "sha256": sha256_file(p)} for p in cfg["val_files"]],
            "train_questions": len(train_items),
            "train_states": len(train_roots),
            "val_questions": len(val_items),
            "train_by_family": dict(Counter(it["family"] for it in train_items)),
            "dropped_overlength": {"train": dropped_train, "val": dropped_val},
        },
        "best": best,
        "steps": step,
        "log": log,
        "reload_check": {"questions": len(reload_items), "max_abs_score_diff": max(abs(a - b) for a, b in zip(ref, again))},
        "hardware": hardware(device),
        "wall_s": time.perf_counter() - t_start,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "adapter_sha256": sha256_file(out / "adapter/adapter_model.safetensors"),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"best step {best['step']} val loss {best['loss']:.4f}; reload max |diff| {meta['reload_check']['max_abs_score_diff']:.2e}; "
        f"saved {out / 'adapter'}",
        flush=True,
    )
    return meta
