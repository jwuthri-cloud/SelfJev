# Fine-tune and RLCD

Two commands train the best recipe (Qwen3.5-4B, shared-prefix tree, LoRA) on your own data: `pjev finetune` for the
supervised step and `pjev rlcd` for reinforcement learning toward calibrated decisions. Both run on one CUDA GPU (an
AWS box, never the laptop) and write a run directory you can serve directly.

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

## Part 1: fine-tune vs RLCD in plain words

Picture a kid learning to guess which of two boxes holds the candy, and to say how sure they are.

- **Fine-tuning is the answer key.** After each guess the teacher opens the right box, and the kid adjusts to point
  at it more firmly. A confident wrong guess gets the biggest correction of all, even when the answer key itself has a
  mistake. On messy lessons the kid ends up very sure, even about things nobody can be sure of.
- **RLCD is a points game.** The kid says "I'm 70% sure it's this box". The box is opened and the kid gets points:
  sure and right earns a lot, sure and wrong loses a lot, unsure earns or loses a little. The kid tries a few
  different "how sure" answers, sees which ones earned more points, and leans toward those. The points are designed
  so that over many rounds the best strategy is to say exactly how sure you really are, and you can choose how much
  a confident miss costs. Learning from points instead of from the answer key is reinforcement learning.
- **The points are the reward.** A reward is a number that grades one attempt: higher is better.

What changes inside the model is the same in both: the same small set of adjustable weights (the LoRA adapter, 1.4%
of the model) gets nudged a little after every batch. Fine-tuning nudges it toward "this is the answer"; RLCD nudges
it toward "be this sure". RLCD does not train a different part of the model, and it adds no new layer.

## Part 2: step by step, the data and one update of each

![One training step of fine-tuning and of RLCD on the same question, with the numbers at every stage](assets/rlcd_vs_finetune_step.svg)

### The data

| | fine-tune | RLCD |
|---|---|---|
| each row | text, question, options, target | the same row |
| also needs | nothing | the starting model's own probabilities for the row (computed once before training: the anchor) and a reward, the rule that grades an attempt |
| where the grade comes from | | here from the target (log, Brier and spherical score); it could be a cost table, a judge's verdict or what happened next, with no target at all |

In our runs both commands read the same file, in order: `pjev finetune` first, then `pjev rlcd --init` on its result.

### One update

Both start identically: the model reads the text through the tree and gives each option a score, here billing 0.49
and tech 0.00, which a softmax turns into 62% billing, 38% tech. Then:

**Fine-tune**

1. Loss = −log p(target) = −log 0.62 = 0.48.
2. The push on each score is p − target: billing 0.62 − 1 = −0.38 (raise it), tech 0.38 − 0 = +0.38 (lower it).
   The push is exactly how wrong the model was.
3. Backpropagation carries the push through all 32 layers into the LoRA matrices, and the optimizer moves them a
   small step (learning rate 2e-4). The next text like this one gets a higher p(billing).

**RLCD**

1. Try 8 answers near the model's own: add a little random noise (sd 0.3) to the scores. Five of them, as
   p(billing): 0.48, 0.55, 0.62, 0.70, 0.78.
2. Grade each try with the reward (log + Brier + spherical score against billing): −0.60, −0.23, +0.09, +0.38, +0.62.
   The average is +0.05.
3. Advantage = grade − average: −0.65, −0.28, +0.03, +0.33, +0.57. Tries above average should become more likely and
   tries below less likely; here that means raise billing and lower tech.
4. Add the anchor: a small pull (KL penalty, weight 0.05) back toward the starting model's probabilities, so a few
   batches cannot drag it far.
5. Backpropagation into the same LoRA matrices, with smaller steps (learning rate 5e-5).

In reinforcement-learning terms this is a policy gradient with a baseline: the model's scores define a distribution
over tries, and the update raises the log-probability of each try in proportion to its advantage. Averaged over many
tries it follows the slope of the expected reward.

### How the reward changes the model

In fine-tuning the push on a score is fixed by one formula (p − target). In RLCD the push comes from the grades, so
the reward decides how hard each example pulls. How hard each score pulls the correct option's score up, with two
options, in three situations:

| situation | p(target) | log score (= fine-tuning) | Brier | spherical |
|---|---|---|---|---|
| right but unsure | 0.62 | 0.38 | 0.36 | 0.23 |
| right and sure | 0.95 | 0.05 | 0.01 | 0.003 |
| sure and wrong | 0.05 | **0.95** | 0.18 | 0.05 |

- The log score pushes hardest on confident mistakes. When the label is right, that fixes a real error; when the
  label is noise or the text is genuinely ambiguous, the model bends to fit it and learns to be sure where it
  should not be.
- Brier and spherical pull much less on those cases, so the model can stay unsure there. That is where hedging
  comes from. On examples where the model is unsure but right, all three pull about the same.
- All three are proper scoring rules: over many similar texts that are billing 70% of the time, each one is best at
  saying 70%. They differ in how much a single example can pull.
- The default reward mixes all three (`--reward log=1,brier=1,spherical=1`, so confident mistakes still get pushed
  about as hard as in fine-tuning). `--reward brier=1,spherical=1` drops the log score for the most hedging.

![Fine-tune vs RLCD: the learning signal, the data each needs, and how hard a confident mistake is punished](assets/rlcd_vs_finetune.svg)

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

## rlcd settings

- `--reward`: weights of `log`, `brier`, `spherical` (strictly proper scoring rules) and `accuracy` (the argmax
  decision is right; not proper on its own, keep a proper term next to it).
- `--samples` (8) tries per question, drawn from a Gaussian around the model's scores with sd `--sigma` (0.3).
- `--beta` (0.05): the KL penalty to the starting model's probabilities, computed once before training.
- `--lr` defaults to 5e-5 (fine-tune: 2e-4); the best checkpoint is the one with the lowest validation Brier score.

`tests/test_finetune.py` checks the property that matters: trained only with this objective, on a question whose
answer is "yes" 70% of the time, the model learns to say 70% (binary), and 60% for a three-way choice with a
60/30/10 split.

Jev says it is trained with "RLCD" and gives no details; Laya, an open Jev-like engine, uses the name for policy
gradient on proper-scoring-rule rewards ([landscape](landscape.md)), which is what `pjev rlcd` implements.

## First test on the real model (2026-09-25): no gain

RLCD from `weights/qwen35_4b_tree` on 4,412 labeled questions it never trained on, against a plain fine-tune on the
same questions ([JOURNAL](JOURNAL.md), `reports/rlcd_2026-09-25/`):

| eval2 | accuracy | ECE | confidently wrong (≥ 0.9) |
|---|---|---|---|
| start | 95.58 | 0.005 | 30 |
| + RLCD | 95.63 | 0.010 | 34 |
| + fine-tune (control) | 95.43 | 0.005 | 31 |
| Jev | 97.24 | 0.041 | 7 |

- On questions the model had already trained on, RLCD made it overconfident within 100 steps: keep RLCD data fresh.
- With one hard label per question, a proper-score reward is the same signal as cross-entropy, only noisier: it
  cannot say "be less sure here". The model is already calibrated (ECE 0.005, Jev 0.041); Jev's edge is fewer
  mistakes and hedged ones.
- The sampled reports also bias the optimum toward overconfidence as `--sigma` grows (70% base rate: 0.695 at 0.3,
  0.754 at 1.0); keep `--sigma` small.
- For RLCD to matter, the reward has to carry more than the label: soft targets (judges' agreement, a teacher's
  probabilities) or a cost that punishes confident mistakes more than it rewards confident right answers.
