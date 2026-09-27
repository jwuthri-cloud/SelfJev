# selfjev

An open decisions model with [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)'s API. Send a text
(the *state*) and questions: yes/no (`noul`), pick one (`choice`), a position on a scale (`score`) or all that apply
(`multi`). Get typed answers with probabilities from forward passes, with no text generation. The model, `selfjev-4b`,
is Qwen3.5-4B with a LoRA adapter; it runs on one GPU, and code written for Jev works by changing the base URL.

It copies the *interface* of TypeSafe's Jev, not its undisclosed model, and measures every step against Jev on the
same questions.

**Docs site: <https://jwuthri.github.io/SelfJev/>**, with the [key findings](docs/findings.md),
[leaderboard](docs/leaderboard.md), [speed and cost](docs/speed.md), model write-ups and the lab notebook. The pages are
the Markdown files in [docs/](docs/).

## Status (2026-09-26)

| model | eval2 (target task, 1,991 q) | dev benchmark (3,471 q) | eval_llm (LLM evaluation, 946 q) |
|---|---|---|---|
| Jev (`typesafe/jev`, API) | **97.2** | 82.7 | 92.5 |
| **`selfjev-4b`, the default**: Qwen3.5-4B + LoRA r64 trained from scratch with the shared-prefix tree on all 80K non-test questions (texts up to 16K), half-weight Jev probabilities as soft targets, options in the question ([weights/selfjev_4b](weights/selfjev_4b/model.json)) | **95.8** | 83.8 | **93.1** |
| Qwen3.5-4B + LoRA r64, trained with the shared-prefix tree, round-2b + round-3 data, options in the question (`qwen35_4b_tree`, the previous default) | 95.6 | **84.4** | 82.1 |
| Qwen3-4B-Instruct-2507 + LoRA r64, shared-prefix tree, same data (`tree_4b_combo`, archived) | 94.5 | 82.7 | |
| Qwen3-4B-Instruct-2507 + LoRA r64, shared-prefix tree, round-2b data (`tree_4b_instruct_r2x64`) | 92.7 | 82.7 | |
| Qwen3-Reranker-4B + LoRA, shared-prefix tree, round-1 data (`tree_4b`) | 85.1 | 81.6 | |
| Qwen3-Reranker-0.6B + LoRA, stock pairs (`lora_pilot`) | 68.8 | 73.5 | |
| GPT-6 Astra (reasoning low) | not scored (it judged eval2) | 85.8 | |

- **Quality:** 1.4 points behind Jev on eval2, level with Jev on LLM evaluation (eval_llm 93.1 vs 92.5), ahead on the dev
  benchmark. `selfjev-4b` makes 11 confident mistakes on eval2 (≥ 0.9 sure and wrong; `qwen35_4b_tree` 30, Jev 7). The levers, in order: verified
  target-task training data, the base model (Instruct, then Qwen3.5), adapter rank, every option in the question, and
  training Qwen3.5 with the tree (long texts fit).
- **Speed:** the shared-prefix tree reads the text once (32–37× faster than scoring each pair). On an L40S with vLLM,
  the Qwen3 tree answers one question in 55–228 ms server side (Jev ~100–130 ms flat) and costs less per request than
  Jev on a busy GPU. Qwen3.5 on vLLM is exact but slow with many questions ([speed](docs/speed.md)).
- **Train your own:** `selfjev finetune` and `selfjev rlcd` (calibration training with proper scoring rules; Jev calls it RLCD),
  [docs/finetune.md](docs/finetune.md).
- Every result, dead end and open idea: [docs/experiments.md](docs/experiments.md). What ran when:
  [docs/JOURNAL.md](docs/JOURNAL.md).

## Quick start

**Call it.** The SDK needs only httpx and pydantic:

```bash
pip install "selfjev @ git+https://github.com/Jwuthri/SelfJev"
```

```python
from selfjev import Choice, Noul, Score, SelfJev

client = SelfJev(base_url="http://localhost:8000", api_key="...")  # or SELFJEV_BASE_URL / SELFJEV_API_KEY
res = client.system_one(
    state="Ticket 4411: the invoice was charged twice and the customer wants the second charge back.",
    questions={
        "refund": Noul("Does the customer ask for a refund?"),
        "team": Choice("Which team should handle this?", {"billing": "billing and refunds", "tech": "outages and bugs"}),
        "urgency": Score("How urgent is it?", ["not urgent", "this week", "today"]),
    },
)
res.nouls["refund"].noul, res.choices["team"].choice, res.scores["urgency"].score
```

`AsyncSelfJev` is the same with `await`. Requests, answers and errors: [docs/api.md](docs/api.md).

**Serve it** on one NVIDIA GPU with at least 16 GB:

```bash
git lfs pull --include "weights/selfjev_4b/*"
uv sync --extra serve --extra gpu
uv run selfjev serve --host 0.0.0.0 --port 8000
```

Docker, and one command on AWS (`selfjev deploy aws up`, a g6.xlarge at $0.81/h): [docs/deploy.md](docs/deploy.md).

**Fine-tune it** on your data. Over HTTP, `selfjev serve --fine-tuning` takes a JSONL of requests with their expected
answers (`client.upload_file`, then `client.create_fine_tuning_job(..., method="supervised" or "rlcd")`) and serves
the result next to `selfjev-4b`. On any GPU box:

```bash
uv run selfjev finetune --data my_train.jsonl --out runs/mine --init weights/selfjev_4b
uv run selfjev rlcd --data my_train.jsonl --out runs/mine_rlcd --init runs/mine/adapter
```

What each buys: [docs/finetune.md](docs/finetune.md). How the model decides: [docs/how_it_works.md](docs/how_it_works.md).
Training, evaluation and data pipelines: [docs/reproduce.md](docs/reproduce.md).

**Develop:**

```bash
uv sync && uv run pytest -q            # CPU tests, ~20 s
uv run pre-commit install              # ruff check + format on every commit (CI runs the same)
```

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
src/selfjev/
  client.py, types.py   the SDK: SelfJev, AsyncSelfJev, Noul / Choice / Score / Multi (httpx + pydantic only)
  server/               the HTTP API: app (routes, auth, errors), compat (Jev's format <-> internal), batching,
                        metrics, finetuning (files, jobs, extra adapters)
  engine/               scoring: qwen35 (model, prompt, cache), tree (shared-prefix tree, TreeServer), vllm
  core/                 the internal request schema, option lists, typed answers
  training/             selfjev finetune / rlcd: the loop, losses, the RLCD objective, tree batching
  evaluation/           eval reports, calibration, benchmark, significance tests
  data/                 loading and validation, the dataset catalog, paid API clients (OpenRouter, OpenAI, Jev)
  deploy/               selfjev deploy aws
  cli.py                selfjev serve | classify | eval | calibrate | compare | bench | finetune | rlcd | merge | deploy
tests/                  mirrors src/selfjev; CPU only
scripts/                data/ (builders, generators, judges, batches: grow_batch.sh), eval/ (ledger, eval2 summary,
                        Jev comparison, calibration), train/ (the selfjev-4b recipe), aws/ (aws_launch.sh), docs/
deploy/                 Dockerfile, docker-compose.yml
data/                   THE dataset: data/all.jsonl.gz (every question + Jev's prediction; scripts/data/build_all.py),
                        catalog and growth procedure in data/README.md (new data = a batch, scripts/data/grow_batch.sh)
weights/                selfjev_4b (the default) and qwen35_4b_tree, Git LFS, model.json each
reports/                every eval report, benchmark, review and generated summary
docs/                   the docs site: API, deploy, findings, write-ups, ledger and journal
```

Dead-end code (custom cross-attention, jina, T5Gemma, compact tree, option pointers, teacher distillation) and the
scripts of finished experiments are at tag `archive/pre-cleanup-2026-09-27`.

## Limitations

- eval2 and the authored eval set are LLM-written and LLM-verified, not human-verified.
- The dev benchmark has been reused for many decisions, and eval2 has informed the research direction: a fresh final
  test set is needed before claiming parity.
- Every result is one run with one seed.
- Public datasets may overlap the base models' pretraining data.
