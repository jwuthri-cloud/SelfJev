"""Qwen3.5 (hybrid Gated DeltaNet) served by vLLM: the shared-document model of challengers.ChallengerScorer
(challenger-state-first-v1 prompts, e.g. the qwen35_4b_tree adapter) with its LoRA merged.

One prompt per candidate (root + branch token ids, exactly ChallengerScorer.entry's); vLLM's prefix cache shares the
document between them. Readout: one generated token restricted to {yes, no} with processed logprobs, so
logprob(yes) - logprob(no) = z_yes - z_no, the value ChallengerScorer.readout computes. Needs the merged checkpoint
(`merge`, project venv) and a venv with vllm.

  python -m selfjev.vllm_qwen35 merge --adapter runs/qwen35_4b_tree/adapter --out runs/qwen35_4b_tree/merged
  python -m selfjev.vllm_qwen35 serve --model-dir runs/qwen35_4b_tree/merged --port 8766 --options-in-question
"""

import argparse
import json
import time
from pathlib import Path

from .challengers import INSTRUCTION, MODELS, ChallengerScorer
from .data import sha256_file

QWEN35_4B = MODELS["qwen35_4b"]


def merge(adapter, out, model_id=QWEN35_4B[0], revision=QWEN35_4B[1]):
    """The LoRA merged into a copy of the ORIGINAL checkpoint (multimodal Qwen3_5ForConditionalGeneration, the layout
    vLLM loads): every file copied, every language-model tensor the adapter changes replaced by its merged value."""
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForCausalLM

    peft = PeftModel.from_pretrained(Qwen3_5ForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16), adapter)
    targets = {
        n.removeprefix("base_model.model.").removesuffix(".lora_A") + ".weight" for n, _ in peft.named_modules() if n.endswith(".lora_A")
    }
    merged = peft.merge_and_unload().state_dict()  # only the LoRA targets are copied: other tensors may differ by dtype only
    src, out = Path(snapshot_download(model_id, revision=revision)), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    changed = []
    for f in sorted(src.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file() and f.name != "model.safetensors.index.json":
                shutil.copyfile(f, out / f.name)  # content only: HF cache files are read-only
            continue
        tensors = load_file(f)
        for k, t in tensors.items():
            key = k.replace("model.language_model.", "model.", 1)  # checkpoint key -> Qwen3_5ForCausalLM key
            if key in targets:
                assert merged[key].shape == t.shape, (k, merged[key].shape, t.shape)
                tensors[k] = merged[key].to(t.dtype).contiguous()
                changed.append(k)
        save_file(tensors, out / f.name, metadata={"format": "pt"})
    if (src / "model.safetensors.index.json").exists():
        shutil.copyfile(src / "model.safetensors.index.json", out / "model.safetensors.index.json")
    if len(changed) != len(targets):
        raise RuntimeError(f"{len(changed)} checkpoint tensors replaced but the adapter has {len(targets)} LoRA layers: key mismatch?")
    (out / "merge_meta.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "revision": revision,
                "dtype": "bfloat16",
                "adapter_sha256": sha256_file(Path(adapter) / "adapter_model.safetensors"),
                "changed_tensors": len(changed),
            },
            indent=2,
        )
    )
    print(f"{out}: {len(changed)} tensors merged")


class VllmQwen35Scorer:
    def __init__(
        self, model_dir, max_length=32768, gpu_memory_utilization=0.85, enable_prefix_caching=True, max_num_seqs=128, **llm_kwargs
    ):  # e.g. mamba_block_size: where vLLM may resume a hybrid model's cached recurrent state
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.max_length, self.tokenizer = max_length, AutoTokenizer.from_pretrained(model_dir)
        self._enc = object.__new__(ChallengerScorer)  # only its prompt builder: entry() needs these attributes
        self._enc.__dict__.update(name="qwen35_4b", tokenizer=self.tokenizer, max_length=max_length)
        self.yes, self.no = (self._enc.tokens(s) for s in ("yes", "no"))
        assert len(self.yes) == len(self.no) == 1, (self.yes, self.no)
        self.yes, self.no = self.yes[0], self.no[0]
        provenance = json.loads((Path(model_dir) / "merge_meta.json").read_text())
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            max_model_len=max_length,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            logprobs_mode="processed_logprobs",
            max_logprobs=2,
            limit_mm_per_prompt={"image": 0, "video": 0},
            **llm_kwargs,
        )
        self.params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[self.yes, self.no])
        self.meta = {
            "model": provenance["model_id"],
            "revision": provenance["revision"],
            "adapter": f"merged into {model_dir}",
            "adapter_sha256": provenance["adapter_sha256"],
            "architecture": "shared document on vLLM (prefix cache), readout logprob(yes) - logprob(no) over {yes, no}",
            "prompt": "challenger-state-first-v1",
            "prompt_sha": __import__("hashlib").sha256(INSTRUCTION.encode()).hexdigest()[:12],
            "vllm_version": vllm.__version__,
            "enable_prefix_caching": enable_prefix_caching,
            "llm_kwargs": llm_kwargs,
            "device": "cuda",
            "dtype": "bfloat16",
            "max_length": max_length,
            "truncation": "none (overlength input raises InputTooLong)",
        }

    def score_requests(self, reqs):
        t0 = time.perf_counter()
        entries = [[ChallengerScorer.entry(self._enc, r.state, q) for q in r.questions] for r in reqs]
        prompts = [{"prompt_token_ids": e["root"] + b} for es in entries for e in es for b in e["branches"]]
        t1 = time.perf_counter()
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        t2 = time.perf_counter()
        z = iter(lp[self.yes].logprob - lp[self.no].logprob for o in outs for lp in [o.outputs[0].logprobs[0]])
        per = [[[next(z) for _ in e["branches"]] for e in es] for es in entries]
        cached = [getattr(o, "num_cached_tokens", None) for o in outs]
        roots = {tuple(e["root"]) for es in entries for e in es}
        return per, {
            "pairs": len(prompts),
            "batches": 1,
            "input_tokens": sum(map(len, roots)) + sum(len(b) for es in entries for e in es for b in e["branches"]),
            "padded_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),  # submitted before prefix-cache hits
            "cached_tokens": sum(cached) if None not in cached else None,
            "tokenize_ms": 1e3 * (t1 - t0),
            "model_ms": 1e3 * (t2 - t1),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--options-in-question", action="store_true", help="adapters trained on data/ova/")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.adapter, a.out)
    else:
        from .server import serve

        serve(
            VllmQwen35Scorer(a.model_dir, gpu_memory_utilization=a.gpu_memory_utilization),
            port=a.port,
            options_in_question=a.options_in_question,
        )
