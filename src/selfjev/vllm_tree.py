"""The tree format served by vLLM (CUDA graphs, fused kernels, paged KV cache) instead of plain transformers.

Each leaf becomes one prompt: the exact standalone sequence the tree scores (root + question + leaf tokens).
Prefix reuse depends on the installed engine and scheduling: record actual cache-hit
counts, including first-use requests. Logical tree counts do not prove reuse. Readout: vLLM's Qwen3-reranker conversion, a 1-logit head
lm_head[yes] - lm_head[no], i.e. s = z_yes - z_no, then classify.decide as for every backend.
Needs the LoRA merged into a checkpoint (`merge`), and a separate venv with vllm (its own torch pin).

  python -m selfjev.vllm_tree merge --adapter runs/tree_4b/adapter --out runs/tree_4b/merged   # project venv
  python -m selfjev.vllm_tree serve --model-dir runs/tree_4b/merged --port 8002                 # vllm venv
"""
import argparse
import json
import time
from pathlib import Path

from .data import sha256_file
from .tree import FORMAT, RERANKER_4B, Encoder, TreeScorer, format_config, leaf_paths


def merge(adapter, out, model_id=RERANKER_4B[0], revision=RERANKER_4B[1]):
    """Merged weights + the ORIGINAL config/tokenizer files (vLLM's transformers may not read a re-saved 5.x config)."""
    import shutil
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    PeftModel.from_pretrained(lm, adapter).merge_and_unload().save_pretrained(out)
    src = Path(snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    for f in [f for pat in ("*.json", "*.txt", "*.jinja") for f in src.glob(pat)]:  # the snapshot also holds the original shards
        if f.name not in ("model.safetensors.index.json", "modules.json", "sentence_bert_config.json", "config_sentence_transformers.json"):
            # skip: the original shard index, and sentence-transformers files whose 1_LogitScore module we don't copy
            shutil.copy(f, Path(out) / f.name)
    fmt = Path(adapter) / "tree_format.json"
    (Path(out) / "tree_format.json").write_text(fmt.read_text() if fmt.exists() else json.dumps(format_config(model_id), indent=2))
    (Path(out) / "merge_meta.json").write_text(json.dumps({"model_id":model_id,"revision":revision,
        "adapter_sha256":sha256_file(Path(adapter) / "adapter_model.safetensors"),"dtype":"bfloat16"},indent=2))


class VllmTreeScorer:
    trees, check_lengths = TreeScorer.trees, TreeScorer.check_lengths  # same tree building and length checks

    def __init__(self, model_dir, model_id=RERANKER_4B[0], max_length=32768, gpu_memory_utilization=0.85,
                 enable_prefix_caching=True, max_num_batched_tokens=8192, quantization=None):
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, PoolingParams
        self.max_length, self.tokenizer = max_length, AutoTokenizer.from_pretrained(model_dir)
        saved = Path(model_dir) / "tree_format.json"
        fmt = json.loads(saved.read_text()) if saved.exists() else format_config(model_id)
        self.enc = Encoder(self.tokenizer, model_id, fmt["name"])
        if self.enc.format["sha"] != fmt["sha"]:
            raise ValueError("Merged checkpoint's recorded tree format hash differs")
        merge_meta = Path(model_dir) / "merge_meta.json"
        provenance = json.loads(merge_meta.read_text()) if merge_meta.exists() else {}
        self.llm = LLM(model=model_dir, runner="pooling", dtype="bfloat16", max_model_len=max_length, enable_prefix_caching=enable_prefix_caching,
                       max_num_batched_tokens=max_num_batched_tokens, enable_chunked_prefill=True,
                       gpu_memory_utilization=gpu_memory_utilization, quantization=quantization,
                       hf_overrides={"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"],
                                     "is_original_qwen3_reranker": True})
        fields = getattr(PoolingParams, "__struct_fields__", ())  # raw logit, no sigmoid (the flag was renamed across versions)
        self.params = PoolingParams(**{"use_activation" if "use_activation" in fields else "activation": False})
        self.meta = {"model": model_id, "revision": provenance.get("revision"), "architecture": f"tree paths ({fmt['name']}) on vLLM",
                     "adapter": f"merged into {model_dir}", "adapter_sha256": provenance.get("adapter_sha256"),
                     "vllm_version":vllm.__version__,"enable_prefix_caching":enable_prefix_caching,"quantization":quantization,
                     "prompt": fmt["name"], "prompt_sha": fmt["sha"],
                     "device": "cuda", "dtype": "bfloat16", "max_length": max_length, "truncation": "none (overlength input raises InputTooLong)"}

    def score_requests(self, reqs):
        t0 = time.perf_counter()
        trees, owners = self.trees(reqs)
        self.check_lengths(trees, owners)
        prompts = [{"prompt_token_ids": [t["ids"][i] for i in p]} for t in trees for p in leaf_paths(t)]
        t1 = time.perf_counter()
        outputs = self.llm.classify(prompts, pooling_params=self.params, use_tqdm=False)
        scores = iter([o.outputs.probs[0] for o in outputs])
        self.last_cached_tokens = [getattr(o, "num_cached_tokens", None) for o in outputs]
        t2 = time.perf_counter()
        by_q = {}
        for own in owners:
            for ri, qid, _ in own:
                by_q.setdefault((ri, qid), []).append(next(scores))
        state_tokens = sum(t["root"] for t in trees)
        return [[by_q[ri, q.id] for q in r.questions] for ri, r in enumerate(reqs)], {
            "pairs": len(prompts), "batches": 1, "input_tokens": sum(len(t["ids"]) for t in trees),
            "padded_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),  # tokens submitted before prefix-cache hits
            "logical_state_sequences": len(trees), "logical_state_tokens": state_tokens,
            "cached_tokens_by_leaf":self.last_cached_tokens,
            "cached_tokens":sum(self.last_cached_tokens) if all(n is not None for n in self.last_cached_tokens) else None,
            "tokenize_ms": 1e3 * (t1 - t0), "model_ms": 1e3 * (t2 - t1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=RERANKER_4B[0])
    p.add_argument("--revision", default=RERANKER_4B[1])
    p = sub.add_parser("serve")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-id", default=RERANKER_4B[0], help="base model id (sets the tree format's chat suffix)")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--options-in-question", action="store_true", help="adapters trained on data/ova/")
    p.add_argument("--quantization", help="vLLM quantization for a bf16 checkpoint, e.g. fp8 (dynamic weights + activations on H100-class GPUs)")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.adapter, a.out, a.model_id, a.revision)
    else:
        from .server import serve
        serve(VllmTreeScorer(a.model_dir, model_id=a.model_id, gpu_memory_utilization=a.gpu_memory_utilization, quantization=a.quantization), port=a.port, options_in_question=a.options_in_question)
