# SelfJev (personal-jev)

Fast, instruction-conditioned **binary / multiclass / multilabel** decisions from open Qwen3 models: send a text (the
*state*), natural-language questions and optional candidate labels with descriptions; get typed decisions and
probabilities from batched forward passes, with no text generation.

It copies the *interface* of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), not its
undisclosed model, and measures every step against Jev on the same questions.

**Docs site: <https://jwuthri.github.io/SelfJev/>**, with the [key findings](docs/findings.md),
[leaderboard](docs/leaderboard.md), [speed and cost](docs/speed.md), model write-ups and the lab notebook. The pages are
the Markdown files in [docs/](docs/).

## Status (2026-09-24)

| model | eval2 (target task, 1,991 q) | dev benchmark (3,471 q) |
|---|---|---|
| Jev (`typesafe/jev`, API) | **97.2** | 82.7 |
| **Qwen3-4B-Instruct-2507 + LoRA r64, shared-prefix tree, round-2b data** (`tree_4b_instruct_r2x64`) | **92.7** | **82.7** |
| Qwen3-Reranker-4B + LoRA, shared-prefix tree, round-1 data (`tree_4b`) | 85.1 | 81.6 |
| Qwen3-Reranker-4B + LoRA, stock pairs (`lora_4b`) | 86.5 | 80.3 |
| Qwen3-Reranker-0.6B + LoRA, stock pairs (`lora_pilot`) | 68.8 | 73.5 |
| GPT-6 Astra (reasoning low) | not scored (it judged eval2) | 85.8 |

- **Quality:** 4.5 points behind Jev on eval2; tied on the dev benchmark. The biggest lever was verified target-task
  training data (+5.5 on eval2), then the Instruct base (+3.0) and adapter capacity (+2.1).
- **Speed:** the shared-prefix tree reads the text once and is 32–37× faster than scoring each (text, candidate) pair
  for 16 questions × 3 candidates on 8K–16K-token texts. Jev still answers in a flat ~150 ms where one A10G needs
  120 ms (8 tokens) to 755 ms (4,096 tokens) for one question.
- Every result, dead end and open idea: [docs/experiments.md](docs/experiments.md). What ran when:
  [docs/JOURNAL.md](docs/JOURNAL.md).

## Quick start

```bash
uv sync
uv run pjev classify examples/request.json                          # untrained Qwen3-Reranker-0.6B
uv run pjev classify examples/request.json --tree --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 --adapter runs/tree_4b/adapter --dtype bfloat16
uv run pjev serve --adapter runs/lora_pilot/adapter --dtype bfloat16  # Decisions-API-shaped endpoint on :8000
uv run pytest -q
```

Trained adapters live in `runs/` (not in git). Request format, output rules and the API mapping:
[docs/how_it_works.md](docs/how_it_works.md). Training, evaluation and data pipelines:
[docs/reproduce.md](docs/reproduce.md).

## How it works

1. A Qwen3 reranker (or instruct model) judges "does this text support this answer?" and the score is its `yes` − `no`
   logit. LoRA adapters on the attention (and optionally MLP) projections are the only trained parameters.
2. The **shared-prefix tree** puts the text first and branches every question and candidate off it with a tree
   attention mask: the text is read once, and each leaf scores exactly like the standalone pair.
3. Scores become typed answers: sigmoid for binary, softmax within a question for multiclass, per-candidate sigmoid for
   multilabel, with optional held-out temperatures and thresholds.

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
