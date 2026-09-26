# Fine-tune and RLCD

Two commands train the best recipe (Qwen3.5-4B, shared-prefix tree, LoRA) on your own data: `pjev finetune` for the
supervised step and `pjev rlcd` for reinforcement learning toward calibrated decisions. Both run on one CUDA GPU (an
AWS box, never the laptop) and write a run directory you can serve directly.

![Fine-tune vs RLCD: the learning signal, the data each needs, and how hard a confident mistake is punished](assets/rlcd_vs_finetune.svg)

```bash
# 1. supervised fine-tune, from scratch or on top of the best model
uv run pjev finetune --data my_train.jsonl --out runs/mine --init weights/qwen35_4b_tree

# 2. RLCD on top of it
uv run pjev rlcd --data my_train.jsonl --out runs/mine_rlcd --init runs/mine/adapter

# 3. serve the result (same option-list transform as in training)
uv run python -m personal_jev.qwen35_tree serve --adapter runs/mine_rlcd/adapter --options-in-question
```

The tree server scores exactly like standalone sequences (CPU test on a tiny model) but has not been timed on a GPU yet.
The vLLM path (`python -m personal_jev.vllm_qwen35 merge`, then `serve`) is the one measured on the real model: same
accuracy, fast for one question, slow for many ([speed](speed.md)).

Code: [src/personal_jev/finetune.py](../src/personal_jev/finetune.py). Tests (CPU, a tiny random model):
[tests/test_finetune.py](../tests/test_finetune.py).

## Data

One question per line, the same format as every eval file in `data/`:

```json
{"state": "Ticket 4411: the invoice was charged twice ...",
 "question": {"type": "multiclass", "instruction": "Which team should handle this?",
              "candidates": [{"id": "billing", "description": "billing and refunds"},
                             {"id": "tech", "description": "outages and bugs"}]},
 "target": "billing"}
```

- `type` is `binary` (target `true` / `false`, no candidates), `multiclass` (target: one candidate id) or `multilabel`
  (target: a list of candidate ids, possibly empty).
- `id` and `family` are optional. Several questions about the same `state` share one tree: the text is encoded once.
- `--val` gives a validation file; without it, 5% of `--data` (at most 1,000 questions) is held out.
- Every option is listed in the question text (`options.py`), as for the best model; `--no-options-in-question` turns
  that off. Serve with the same setting.
- Questions longer than `--max-length` (default 8,192 tokens: root + question + longest candidate) are dropped and
  counted in `train_meta.json`, never truncated.

## Output

`<out>/adapter` (best validation checkpoint), `<out>/adapter_last`, `<out>/train_meta.json` (base model and revision,
data sha256, settings, every validation). Validation reports accuracy, cross-entropy, Brier score, expected calibration
error (10 bins) and the number of confidently wrong decisions (confidence ≥ 0.9). `finetune` keeps the checkpoint with
the lowest validation cross-entropy, `rlcd` the one with the lowest Brier score. To keep a result with the repo, copy
the adapter into [weights/](../weights/README.md).

## finetune

Cross-entropy on the targets (softmax over a multiclass question's candidates, a sigmoid per yes/no decision), the
recipe that produced `weights/qwen35_4b_tree`: LoRA r64 on the attention and DeltaNet projections, lr 2e-4 with 5%
warm-up and linear decay, batches of whole states packed to 8,192 tokens × 4 accumulation steps, per-layer activation
checkpointing. Start from `--init weights/qwen35_4b_tree` to adapt the best model to a new domain, or without `--init`
to train a fresh adapter. `--base qwen35` uses Qwen3.5-2B for quick runs.

## rlcd: Reinforcement Learning for Calibrated Decisions

Jev says it is trained with "RLCD" and gives no details. Laya, an open Jev-like engine, uses the name for policy
gradient on proper-scoring-rule rewards ([landscape](landscape.md)). That is what `pjev rlcd` implements, on top of a
fine-tuned adapter:

- **The report.** For each question the model states probabilities: a softmax over the candidates (multiclass) or one
  yes/no probability per decision (binary, multilabel).
- **The reward** is a weighted sum of strictly proper scoring rules of that report against the target: log score,
  Brier score and spherical score (`--reward log=1,brier=1,spherical=1`). A proper scoring rule pays most, on average,
  for the true probabilities, so the policy that maximizes it is calibrated: 70% sure means right 70% of the time.
  `accuracy` (the argmax decision is right) can be mixed in to express that decisions matter; it is not proper on its
  own, so keep a proper term next to it.
- **The policy gradient.** Each step samples `--samples` reports per question from a Gaussian around the model's
  logits (sd `--sigma`), and raises the likelihood of the reports that scored above the question's mean reward.
- **The anchor.** A KL penalty (`--beta`) to the starting model's probabilities keeps it from drifting. The starting
  model's scores are computed once before training.

`tests/test_finetune.py` checks the property that matters: a model trained only with this objective on questions
whose answer is "yes" 70% of the time learns to say 70% (binary), and 60% for a three-way choice with a 60/30/10 split.

How it relates to plain fine-tuning: with a differentiable model and a proper reward, the policy gradient estimates
the gradient of the expected score, so RLCD with only the log score is close to cross-entropy training. The
difference is in the rewards you can use: bounded proper scores (Brier, spherical) punish a confident mistake far less
than cross-entropy, which pushes the model to hedge where the data is ambiguous (Jev's errors are hedged, ours were
confidently wrong four times as often, [memo](../reports/jev_hypothesis_2026-09-25.md)), and any reward computed
after the fact (a business cost, an abstention rule, a judge's verdict) fits the same loop without being
differentiable.

Status: implemented and unit-tested, not yet run on the real model. The first run to do is `pjev rlcd` from
`weights/qwen35_4b_tree` on the training data, compared on eval2 for accuracy, Brier and ECE.
