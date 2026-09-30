"""Latency / throughput / memory on synthetic workloads. Numbers describe this machine and settings only.

Workload = state length x questions x candidates per question. The shared-prefix tree reads the state once per request;
only the short question and candidate branches scale with the number of candidates.
"""

import gc
import json
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import torch
import transformers

from ..core.answers import classify
from ..engine.utils import sync

FILLER = (
    "Hi team, following up on ticket 48213. Since the release on Tuesday our checkout page intermittently returns a "
    "500 error when customers apply a discount code, and the order confirmation emails are delayed by up to two hours. "
    "Our finance lead also noticed that last month's invoice lists 14 seats although we downgraded to 10 in March. "
    "We rely on the webhook integration for inventory sync, and the retry queue shows about 300 failed deliveries. "
    "Could you confirm whether these problems are related, what the expected resolution time is, and whether we "
    "should pause our marketing campaign scheduled for Friday? Thanks, Dana (Operations, Northwind Outfitters). "
)


def make_state(tokenizer, n_tokens):
    ids = tokenizer(FILLER * (n_tokens // 60 + 2), add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def make_request(tokenizer, state_tokens, n_questions, n_candidates, kind="multiclass"):
    state = make_state(tokenizer, state_tokens)
    qs = []
    for i in range(n_questions):
        q = {"id": f"q{i}", "type": kind, "instruction": f"Which team should handle issue number {i + 1} in this message?"}
        if kind != "binary":
            q["candidates"] = [
                {
                    "id": f"c{j}",
                    "description": f"Team {j + 1}: handles category {j + 1} problems such as outages, "
                    f"billing errors or integration failures",
                }
                for j in range(n_candidates)
            ]
        qs.append(q)
    return {"state": state, "questions": qs}


def _mem(device):
    if device == "mps":
        return {
            "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / 2**20,
            "note": "MPS driver allocation after the run, including allocator cache (approximates peak)",
        }
    if device == "cuda":
        return {
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
    return {"process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if platform.system() == "Darwin" else 2**10)}


def _reset_mem(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def default_grid(lengths=(512, 2048, 8192), questions=(1, 4, 16)):
    grid = [(n, q, 3, "multiclass") for n in lengths for q in questions]
    grid += [(2048, 4, c, "multiclass") for c in (2, 8)] + [(2048, 16, 1, "binary")]
    return grid


def run(scorer, grid=None, repeats=20, warmup=1, out_dir=None, min_repeats=3, row_budget_s=60.0, load_ms=None):
    """scorer: any selfjev.engine scorer (TreeServer, VllmScorer) with .tokenizer and .device."""
    grid = grid or default_grid()
    sync(scorer.device)
    first = classify(scorer, make_request(scorer.tokenizer, 512, 1, 3))
    rows = []
    for n_tokens, n_q, n_c, kind in grid:
        req = make_request(scorer.tokenizer, n_tokens, n_q, n_c, kind)
        _reset_mem(scorer.device)
        try:
            for _ in range(warmup):
                classify(scorer, req)
            runs, t_row = [], time.perf_counter()  # up to `repeats` samples, stopping after row_budget_s once min_repeats are in
            while len(runs) < repeats and (len(runs) < min_repeats or time.perf_counter() - t_row < row_budget_s):
                runs.append(classify(scorer, req)["meta"])
        except torch.OutOfMemoryError:  # a result on this GPU, not a crash: record it and go on
            rows.append({"state_tokens": n_tokens, "questions": n_q, "candidates": n_c, "type": kind, "oom": True})
            print(json.dumps(rows[-1]), flush=True)
            continue
        e2e, model = [r["total_ms"] for r in runs], [r["model_ms"] for r in runs]
        m = runs[0]
        rows.append(
            {
                "state_tokens": n_tokens,
                "questions": n_q,
                "candidates": n_c,
                "type": kind,
                "pairs": m["pairs"],
                "input_tokens": m["input_tokens"],
                "padded_tokens": m["padded_tokens"],
                "batches": m["batches"],
                "state_encodes": m.get("state_sequences"),  # None when the engine does not report it
                "state_tokens_encoded": m.get("state_tokens"),
                "candidate_tokens_encoded": m.get("candidate_tokens"),
                "memory_rows": m.get("memory_rows"),
                "e2e_ms_p50": statistics.median(e2e),
                "e2e_ms_p95": _pct(e2e, 0.95),
                "model_ms_p50": statistics.median(model),
                "tokenize_ms_p50": statistics.median(r["tokenize_ms"] for r in runs),
                "pairs_per_s": m["pairs"] / (statistics.median(e2e) / 1e3),
                "tokens_per_s": m["input_tokens"] / (statistics.median(e2e) / 1e3),
                "repeats": len(runs),
            }
            | _mem(scorer.device)
        )
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "note"}), flush=True)
    chip = (
        subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor()
    )
    report = {
        "meta": {"prompt": scorer.meta.get("prompt"), "prompt_sha": scorer.meta.get("prompt_sha")}
        | scorer.meta
        | {
            "max_batch_tokens": getattr(scorer, "max_batch_tokens", None),
            "warmup": warmup,
            "chip": chip,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "cold_start": {"load_ms": load_ms, "first_request_ms": first["meta"]["total_ms"], "first_request_pairs": first["meta"]["pairs"]},
        "rows": rows,
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "bench.json").write_text(json.dumps(report, indent=1))
        (Path(out_dir) / "bench.md").write_text(markdown(report))
    return report


def markdown(r):
    m, c = r["meta"], r["cold_start"]
    mem_key = next((k for x in r["rows"] for k in x if k.endswith("_mb")), None)
    lines = [
        "# Serving benchmark",
        "",
        f"- {m['chip']} / {m['device']} / {m['dtype']}; torch {m['torch']}, transformers {m['transformers']}; {m['created']}",
        f"- model `{m['model']}` @ `{m['revision'][:10]}` ({m.get('architecture', '?')}), "
        f"adapter/checkpoint `{m['adapter']}`, prompt `{m['prompt']}`",
        f"- batching: max_batch_tokens={m['max_batch_tokens']} (padded), max_batch_size={m.get('max_batch_size')}; warmup {m['warmup']}; "
        f"samples per row in the `n` column (up to 20, at least 3, ~60 s budget per row)",
        f"- cold start: load {c['load_ms']:.0f} ms + first request ({c['first_request_pairs']} pairs) {c['first_request_ms']:.0f} ms",
        "- e2e = parse + tokenize + forward + result assembly; model = forward passes incl. device sync",
        "",
        "| state tok | questions | cand | type | pairs | state encodes | tokens processed | n | e2e p50 ms | e2e p95 ms | "
        f"model p50 ms | pairs/s | tokens/s | {mem_key or 'mem'} |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in r["rows"]:
        if x.get("oom"):
            lines.append(f"| {x['state_tokens']} | {x['questions']} | {x['candidates']} | {x['type']} | out of GPU memory |")
            continue
        lines.append(
            f"| {x['state_tokens']} | {x['questions']} | {x['candidates']} | {x['type']} | {x['pairs']} | "
            f"{x.get('state_encodes', x['pairs'])} | {x['input_tokens']} | {x['repeats']} | "
            f"{x['e2e_ms_p50']:.0f} | {x['e2e_ms_p95']:.0f} | {x['model_ms_p50']:.0f} | {x['pairs_per_s']:.1f} | {x['tokens_per_s']:.0f} | "
            f"{x.get(mem_key, 0):.0f} |"
        )
    return "\n".join(lines) + "\n"
