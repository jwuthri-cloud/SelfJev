"""selfjev: serve | classify | eval | calibrate | compare | bench | finetune | rlcd | merge | deploy"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_ADAPTER = "weights/selfjev_4b"
HUB_ADAPTER = "Jwuthrich/selfjev-4b"  # what DEFAULT_ADAPTER holds, for a pip install with no checkout
ADAPTER_FILES = ["adapter_model.safetensors", "adapter_config.json", "model.json"]
# The first module a command imports from each extra, and the extra that provides it.
EXTRAS = dict.fromkeys(("fastapi", "uvicorn", "multipart", "torch", "transformers", "peft", "safetensors", "huggingface_hub"), "serve")
EXTRAS |= {"boto3": "deploy"}


def _adapter(path):
    """A local adapter directory, or a Hugging Face repo id downloaded to the HF cache (DEFAULT_ADAPTER → HUB_ADAPTER)."""
    if not path or Path(path).exists():
        return path
    repo = HUB_ADAPTER if path == DEFAULT_ADAPTER else path
    if repo.count("/") != 1 or repo.startswith((".", "/", "~")):
        raise SystemExit(f"adapter not found: {path}")
    from huggingface_hub import snapshot_download

    print(f"adapter {path} not found locally; downloading {repo} from Hugging Face", file=sys.stderr)
    return snapshot_download(repo, allow_patterns=ADAPTER_FILES)


def _scorer(a):
    """The model behind every command: the shared-prefix tree (transformers, exact) or vLLM on merged weights."""
    if a.engine == "vllm":
        from .engine.vllm import VllmScorer

        return VllmScorer(a.model_dir, max_length=a.max_length)
    from .engine.tree import TreeServer

    return TreeServer(a.adapter, max_length=a.max_length, max_batch_tokens=a.max_batch_tokens, merge=not getattr(a, "fine_tuning", False))


def _calibration(a, scorer):
    if not a.calibration:
        return None
    from .evaluation import calibration

    return calibration.load(a.calibration, scorer.meta, scorer.meta.get("prompt_sha"))


def _model_args(p):
    p.add_argument(
        "--adapter", default=DEFAULT_ADAPTER, help=f"LoRA adapter dir or Hugging Face repo (default {DEFAULT_ADAPTER}, else {HUB_ADAPTER})"
    )
    p.add_argument("--engine", default="tree", choices=["tree", "vllm"], help="tree: exact, any number of questions; vllm: merged weights")
    p.add_argument("--model-dir", help="--engine vllm: merged checkpoint from `selfjev merge`")
    p.add_argument("--max-length", type=int, default=32768, help="state + longest question, in tokens; longer input is an error")
    p.add_argument("--max-batch-tokens", type=int, default=16384, help="packed tokens per forward pass")
    p.add_argument("--calibration", help="calibration JSON from `selfjev calibrate` (must match model/adapter/prompt)")


def _finetune_args(p, rlcd=False):
    p.add_argument("--data", required=True, help="training JSONL or .jsonl.gz (state, question, target per line)")
    p.add_argument("--val", help="validation JSONL (default: 5%% of --data, at most 1,000 questions)")
    p.add_argument("--out", required=True, help="run directory: adapter/, adapter_last/, train_meta.json")
    p.add_argument("--init", required=rlcd, help=f"adapter to start from, e.g. {DEFAULT_ADAPTER}" + (" or a finetune run" if rlcd else ""))
    p.add_argument("--no-options-in-question", action="store_true", help="do not list every option in the question text")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, help="default 2e-4 (finetune), 5e-5 (rlcd)")
    p.add_argument("--lora-r", type=int, default=64, help="LoRA rank when starting without --init")
    p.add_argument("--max-length", type=int, default=16384, help="longer questions are dropped and counted, never truncated")
    p.add_argument("--batch-tokens", type=int, default=16384, help="packed tree tokens per micro-batch (at least --max-length)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--soft-weight", type=float, default=0.5, help='rows with "soft" (a teacher\'s probabilities) train on (1 - w) x label + w x soft'
    )
    if rlcd:
        p.add_argument(
            "--reward", default="log=1,brier=1,spherical=1", help="weights of log, brier, spherical (proper), accuracy, confident_miss"
        )
        p.add_argument("--samples", type=int, default=8, help="sampled reports per question")
        p.add_argument("--sigma", type=float, default=0.3, help="sd of the Gaussian around the logits")
        p.add_argument("--beta", type=float, default=0.05, help="KL penalty to the --init model")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="selfjev", description="Serve, evaluate and fine-tune selfjev-4b, an open decisions model with Jev's API."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="HTTP server with Jev's decisions API (POST /api/alpha/decisions) and POST /classify")
    _model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-options-in-question", action="store_true", help="for adapters trained without the option list")
    p.add_argument(
        "--fine-tuning",
        action="store_true",
        help="also serve /v1/files and /v1/fine_tuning/jobs (LoRA unmerged; training needs GPU memory next to serving, e.g. a 48 GB card)",
    )
    p.add_argument("--home", default=os.environ.get("SELFJEV_HOME", str(Path.home() / ".selfjev" / "server")), help="fine-tuning state")
    p = sub.add_parser("classify", help="answer one request in the internal schema (a JSON file, or - for stdin)")
    p.add_argument("request")
    p.add_argument("--no-options-in-question", action="store_true", help="for adapters trained without the option list")
    _model_args(p)
    p = sub.add_parser("eval", help="evaluate JSONL examples; writes report.json and report.md")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", nargs="*", help="only these splits (e.g. test); default all")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    _model_args(p)
    p = sub.add_parser("calibrate", help="fit temperatures on a calibration-split report; thresholds on a validation-split report")
    p.add_argument("--fit", required=True, help="report.json from `selfjev eval --split calibration`")
    p.add_argument("--thresholds", help="report.json from `selfjev eval --split validation`")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare", help="side-by-side markdown of two or more eval reports")
    p.add_argument("reports", nargs="+")
    p.add_argument("--names", nargs="*")
    p.add_argument("--out")
    p = sub.add_parser("bench", help="latency / throughput / memory on synthetic requests")
    _model_args(p)
    p.add_argument("--lengths", default="512,2048,8192")
    p.add_argument("--questions", default="1,4,16")
    p.add_argument("--repeats", type=int, default=20, help="max samples per row (at least 3; ~60 s budget per row)")
    p.add_argument("--out", required=True)
    _finetune_args(sub.add_parser("finetune", help="LoRA fine-tune selfjev's recipe (Qwen3.5 + shared-prefix tree) on your data"))
    _finetune_args(sub.add_parser("rlcd", help="RLCD (proper-scoring-rule training) from a fine-tuned adapter"), rlcd=True)
    p = sub.add_parser("merge", help="merge a LoRA adapter into Qwen3.5-4B for --engine vllm")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--out", required=True)
    p = sub.add_parser("deploy", help="deploy a server: `selfjev deploy aws up|down|status|list|machines`")
    dsub = p.add_subparsers(dest="provider", required=True)
    aws = dsub.add_parser("aws", help="an EC2 GPU box running `selfjev serve` (pip install selfjev[deploy])")
    aws.add_argument("action", choices=["up", "down", "status", "list", "machines"])
    aws.add_argument("--name", default="selfjev", help="deployment name (tags, state file)")
    aws.add_argument("--instance", default="g6.xlarge", help="EC2 type; see `selfjev deploy aws machines`")
    aws.add_argument("--region", default="us-east-2")
    aws.add_argument("--allow-cidr", default="0.0.0.0/0", help="who may reach the API port (the API key still applies)")
    aws.add_argument("--ref", default="master", help="git ref of this repository to deploy")
    aws.add_argument("--api-key", help="default: a new random key, printed once and kept in the state file")
    aws.add_argument("--max-hours", type=float, help="terminate the box after this many hours (a cost cap for trials)")
    aws.add_argument("--ssh", action="store_true", help="also a key pair and port 22 from this machine, for debugging")
    aws.add_argument("--no-wait", action="store_true", help="return once the instance runs, before the server is ready")
    aws.add_argument("--fine-tuning", action="store_true", help="also serve the fine-tuning routes (jobs train on the box: 48 GB GPU)")
    a = ap.parse_args(argv)
    try:
        for k in ("adapter", "init"):
            if getattr(a, k, None) and not (k == "adapter" and getattr(a, "engine", "tree") == "vllm"):  # vllm reads --model-dir
                setattr(a, k, _adapter(getattr(a, k)))
        _run(a)
    except ModuleNotFoundError as e:
        root = (e.name or "").split(".")[0]
        if root == "vllm":
            raise SystemExit("--engine vllm needs vLLM: pip install vllm") from e
        extra = "train" if a.cmd in ("finetune", "rlcd") else EXTRAS.get(root)
        if not extra:
            raise
        raise SystemExit(f"selfjev {a.cmd} needs the {extra} extra ({root} is missing): pip install 'selfjev[{extra}]'") from e


def _run(a):
    if a.cmd == "serve":
        from .server.app import serve

        scorer = _scorer(a)
        home = Path(a.home) if a.fine_tuning else None
        serve(scorer, a.host, a.port, _calibration(a, scorer), not a.no_options_in_question, fine_tuning_home=home, init_adapter=a.adapter)
    elif a.cmd == "classify":
        from .core.answers import classify
        from .core.options import with_option_lists

        scorer = _scorer(a)
        req = json.load(sys.stdin) if a.request == "-" else json.loads(Path(a.request).read_text())
        req = req if a.no_options_in_question else with_option_lists(req)  # as selfjev-4b was trained
        print(json.dumps(classify(scorer, req, _calibration(a, scorer)), indent=2))
    elif a.cmd == "eval":
        from .evaluation.evaluate import run

        scorer = _scorer(a)
        rep = run(scorer, a.data, a.split, _calibration(a, scorer), a.out, a.limit)
        print((Path(a.out) / "report.md").read_text())
        print(f"n={rep['meta']['n']} -> {a.out}/report.json")
    elif a.cmd == "calibrate":
        from .evaluation.calibration import fit

        print(json.dumps(fit(a.fit, a.thresholds, a.out), indent=2))
    elif a.cmd == "compare":
        from .evaluation.evaluate import compare

        md = compare([json.loads(Path(r).read_text()) for r in a.reports], a.names or [Path(r).parent.name for r in a.reports])
        if a.out:
            Path(a.out).write_text(md)
        print(md)
    elif a.cmd == "bench":
        from .evaluation.benchmark import default_grid, run

        t0 = time.perf_counter()
        scorer = _scorer(a)
        grid = default_grid(tuple(map(int, a.lengths.split(","))), tuple(map(int, a.questions.split(","))))
        run(scorer, grid, a.repeats, out_dir=a.out, load_ms=1e3 * (time.perf_counter() - t0))
        print((Path(a.out) / "bench.md").read_text())
    elif a.cmd in ("finetune", "rlcd"):
        from .training.finetune import train

        extra = {}
        if a.cmd == "rlcd":
            extra = {"reward_weights": {k: float(v) for k, v in (x.split("=") for x in a.reward.split(","))}}
            extra |= {"samples": a.samples, "sigma": a.sigma, "beta": a.beta}
        train(a.cmd, a.data, a.out, val=a.val, init=a.init, options_in_question=not a.no_options_in_question, epochs=a.epochs, lr=a.lr,
              lora_r=a.lora_r, max_length=a.max_length, batch_tokens=a.batch_tokens, grad_accum=a.grad_accum, eval_every=a.eval_every,
              seed=a.seed, soft_weight=a.soft_weight, **extra)  # fmt: skip
    elif a.cmd == "deploy":
        from .deploy import aws

        if a.action == "machines":
            for k, (gpu, price, use) in aws.MACHINES.items():
                print(f"{k:13s} {gpu:22s} ${price:.3f}/h  {use}")
        elif a.action == "up":
            rec = aws.up(a.name, a.instance, a.region, a.allow_cidr, a.ref, a.api_key, a.max_hours, a.ssh, not a.no_wait, a.fine_tuning)
            print(json.dumps({k: rec[k] for k in rec if k != "security_group"}, indent=2))
        elif a.action == "list":
            print(json.dumps([{k: r[k] for k in ("name", "region", "instance_type", "endpoint")} for r in aws.listing()], indent=2))
        else:
            print(json.dumps(getattr(aws, a.action)(a.name), indent=2, default=str))
    elif a.cmd == "merge":
        from .engine.vllm import merge

        merge(a.adapter, a.out)
        print(f"merged -> {a.out}")


if __name__ == "__main__":
    main()
