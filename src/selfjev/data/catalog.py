"""The catalog of data/all.jsonl.gz: every dataset in it, the frozen test sets, and the derived views left out on purpose.

scripts/data/build_all.py builds data/all.jsonl.gz and its generated catalog data/README.md from these lists: edit them
here, not the README. Paths are relative to the repo root.
"""

import hashlib
import json
from pathlib import Path

# name, file, what it is, role, write-up
DATASETS = [
    (
        "hf",
        "data/hf.jsonl",
        "public human-labeled sets (Banking77, AG News, TweetEval, MNLI, CLINC150, DBpedia, TREC, emotion, BoolQ, SST-2, ...) "
        "in our schema",
        "training + dev-benchmark test",
        "docs/data.md",
    ),
    (
        "synthetic",
        "data/synthetic.jsonl",
        "round 1: trap-heavy authored texts, 6 families (12 Claude Sonnet agents)",
        "training",
        "docs/data.md",
    ),
    (
        "hardcases",
        "data/hardcases.jsonl",
        "round 2: verified hard cases (4 OpenRouter models + 8 Sonnet agents, blind Astra judge)",
        "training",
        "docs/hardcases_round2.md",
    ),
    (
        "hardcases_r3",
        "data/hardcases_r3.jsonl",
        "round 3: verified hard cases (Luna, Gemini 3.8 Flash, Grok 4.7; blind Astra judge)",
        "training",
        "docs/data.md",
    ),
    (
        "hardcases_llm",
        "data/hardcases_llm.jsonl",
        "LLM evaluation: score, judge, verify, guardrail, jailbreak (Luna, Gemini, Grok, DeepSeek; Astra judge; strict + safety-reviewed)",
        "training",
        "docs/llm_eval_data.md",
    ),
    (
        "eval",
        "data/eval.jsonl",
        "authored evaluation set, 7 families (7 Claude Opus agents); its test split is part of the dev benchmark",
        "dev benchmark",
        "docs/data.md",
    ),
    (
        "eval2",
        "data/eval2.jsonl",
        "FROZEN target-task test set (Opus 5.5, Kimi K3, GLM 5.3; two blind judges)",
        "frozen test",
        "data/eval2/REVIEW.md",
    ),
    (
        "eval_llm",
        "data/eval_llm.jsonl",
        "FROZEN LLM-evaluation test set (Opus 5.5, Kimi K3, GLM 5.3; Astra + Sonnet 5)",
        "frozen test",
        "docs/llm_eval_data.md",
    ),
    (
        "compact_challenge_v1",
        "data/compact_challenge_v1.jsonl",
        "FROZEN programmatic challenge: explicit-record reasoning with evidence/absence contrasts (Codex session)",
        "frozen test",
        "scripts/data/build_compact_challenge.py",
    ),
]
TEST_DATASETS = {"eval", "eval2", "eval_llm", "compact_challenge_v1"}  # never train on these (eval: dev-benchmark validation/test)

# left out of all.jsonl.gz on purpose: path, what it is
DERIVED = [
    (
        "data/hardcases_nb.jsonl",
        "round 2 with `none`-correct capped at 10% (round 2b; a subset of hardcases). Not tracked: "
        "`scripts/data/rebalance_nota.py` rebuilds it byte for byte (docs/reproduce.md, Older recipes)",
    ),
    (
        "data/ova/*.jsonl",
        "every option listed in the question (the transform the best models train and serve with); eval2, eval_llm, "
        "hf and eval are tracked, the training copies are rebuilt by `scripts/data/options_in_question.py`",
    ),
    ("data/val_sample_1200.txt", "the 1,200 validation ids of qwen35_4b_tree / selfjev-4b (`scripts/train/jev_soft_targets.py`)"),
    ("data/dev.jsonl", "14-text fixtures for unit tests"),
    ("data/*/raw/, data/*/review/", "per-writer source files, judge answers, agreement reports (inputs of the builds above)"),
]


def datasets(root="."):
    """DATASETS + every grown batch: data/batches/<name>.jsonl under the repo root `root` (training data; description =
    first line of data/batches/<name>/README.md). A batch needs no edit here: build it and rerun scripts/data/build_all.py."""
    out = list(DATASETS)
    for f in sorted((Path(root) / "data/batches").glob("*.jsonl")):
        readme = f.parent / f.stem / "README.md"
        what = next((line.strip("# ").strip() for line in readme.read_text().splitlines() if line.strip()), "") if readme.exists() else ""
        out.append(
            (f.stem, f"data/batches/{f.name}", f"batch: {what or 'no README.md yet'}", "training", f"data/batches/{f.stem}/README.md")
        )
    return out


def request_sha(ex):
    """Identifies the text + question a stored prediction was made for."""
    return hashlib.sha1(json.dumps([ex["state"], ex["question"]], sort_keys=True).encode()).hexdigest()[:16]
