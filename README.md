# SelfJev (personal-jev)

Fast, instruction-conditioned **binary / multiclass / multilabel** decisions from open Qwen3 models: send a text (the
*state*), natural-language questions and optional candidate labels with descriptions; get typed decisions and
probabilities from batched forward passes, with no text generation.

It copies the *interface* of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), not its
undisclosed model, and measures every step against Jev on the same questions.

**Docs site: <https://jwuthri.github.io/SelfJev/>**, with the [key findings](docs/findings.md),
[leaderboard](docs/leaderboard.md), [speed and cost](docs/speed.md), model write-ups and the lab notebook. The pages are
the Markdown files in [docs/](docs/).

## Status (2026-09-25)

| model | eval2 (target task, 1,991 q) | dev benchmark (3,471 q) |
|---|---|---|
| Jev (`typesafe/jev`, API) | **97.2** | 82.7 |
| **Qwen3.5-4B + LoRA r64, trained with the shared-prefix tree, round-2b + round-3 data, options in the question** (`qwen35_4b_tree`, [weights/](weights/README.md)) | **95.6** | **84.4** |
| Qwen3-4B-Instruct-2507 + LoRA r64, shared-prefix tree, same data (`tree_4b_combo`, the fastest to serve) | 94.5 | 82.7 |
| Qwen3-4B-Instruct-2507 + LoRA r64, shared-prefix tree, round-2b data (`tree_4b_instruct_r2x64`) | 92.7 | 82.7 |
| Qwen3-Reranker-4B + LoRA, shared-prefix tree, round-1 data (`tree_4b`) | 85.1 | 81.6 |
| Qwen3-Reranker-0.6B + LoRA, stock pairs (`lora_pilot`) | 68.8 | 73.5 |
| GPT-6 Astra (reasoning low) | not scored (it judged eval2) | 85.8 |

- **Quality:** 1.6 points behind Jev on eval2, ahead on the dev benchmark (p = 0.004). The levers, in order: verified
  target-task training data, the base model (Instruct, then Qwen3.5), adapter rank, every option in the question, and
  training Qwen3.5 with the tree (long texts fit).
- **Speed:** the shared-prefix tree reads the text once (32–37× faster than scoring each pair). On an L40S with vLLM,
  the Qwen3 tree answers one question in 55–228 ms server side (Jev ~100–130 ms flat) and costs less per request than
  Jev on a busy GPU. Qwen3.5 on vLLM is exact but slow with many questions ([speed](docs/speed.md)).
- **Train your own:** `pjev finetune` and `pjev rlcd` (reinforcement learning for calibrated decisions),
  [docs/finetune.md](docs/finetune.md).
- Every result, dead end and open idea: [docs/experiments.md](docs/experiments.md). What ran when:
  [docs/JOURNAL.md](docs/JOURNAL.md).

## Quick start

```bash
uv sync && git lfs pull                                              # code + the best adapters in weights/
uv run pytest -q                                                     # CPU tests (some download the 0.6B model)
uv run pjev classify examples/request.json                           # untrained Qwen3-Reranker-0.6B, runs anywhere
# on a CUDA GPU: the best model as an HTTP server (POST /classify, POST /api/alpha/decisions)
uv run python -m personal_jev.qwen35_tree serve --adapter weights/qwen35_4b_tree --options-in-question
# fine-tune on your data, then RLCD
uv run pjev finetune --data my_train.jsonl --out runs/mine --init weights/qwen35_4b_tree
uv run pjev rlcd --data my_train.jsonl --out runs/mine_rlcd --init runs/mine/adapter
```

Request format, output rules and the API mapping: [docs/how_it_works.md](docs/how_it_works.md). Serving options,
training, evaluation and data pipelines: [docs/reproduce.md](docs/reproduce.md).

## How it works

1. A Qwen3 or Qwen3.5 model judges "does this text support this answer?" and the score is its `yes` − `no` logit.
   LoRA adapters are the only trained parameters.
2. The **shared-prefix tree** puts the text first and branches every question and candidate off it: the text is read
   once, and each candidate scores exactly like the standalone sequence. For Qwen3.5's recurrent layers the tree runs
   level by level from copied states.
3. Every option is listed in the question text, so each candidate is judged knowing its alternatives.
4. Scores become typed answers: sigmoid for binary, softmax within a question for multiclass, per-candidate sigmoid for
   multilabel.

## Docs site

Built with [Zensical](https://zensical.org) from `docs/` and `zensical.toml`; `.github/workflows/docs.yml` deploys it
to GitHub Pages on every push to `master` that touches the docs.

```bash
uv run --no-project --with zensical==0.0.65 python -m zensical serve    # preview on http://127.0.0.1:8000
```

## Working in this repo

Several AI sessions work here at once: read [AGENTS.md](AGENTS.md) first. In short: claim work in
[docs/experiments.md](docs/experiments.md) and [docs/JOURNAL.md](docs/JOURNAL.md), never train or tune on test labels,
no heavy jobs on the laptop, and every paid resource needs the user's OK with a price.

## Layout

```
src/personal_jev/  scoring backends (model.py stock, tree.py + vllm_tree.py, custom.py, jina.py, t5_shared.py),
                   training (train*.py), classify.py, evaluate.py, calibration.py, server.py, cli.py
scripts/           data builders, GPU pipelines (run_*.sh), Jev comparison, summaries (ledger, eval2, curve)
data/              public sets, authored eval, synthetic, hard cases (rounds 2–3), eval2, with briefs and reviews
configs/           training configs          calib/  held-out calibration files
reports/           every eval report, benchmark, review and generated summary
docs/              the docs site: findings, write-ups, ledger and journal
tests/             unit and real-model tests
```

## Limitations

- eval2 and the authored eval set are LLM-written and LLM-verified, not human-verified.
- The dev benchmark has been reused for many decisions, and eval2 has informed the research direction: a fresh final
  test set is needed before claiming parity.
- Every result is one run with one seed.
- Public datasets may overlap the base models' pretraining data.
