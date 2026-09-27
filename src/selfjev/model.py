"""Qwen3-Reranker pair scorer: official template, length-sorted batches, one forward pass, s = z_yes - z_no."""
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import sha256_file
from .formatting import PREFIX, SUFFIX

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"  # HF commit pinned 2026-09-22
YES_ID, NO_ID = 9693, 2152  # repo's 1_LogitScore/config.json; asserted against the tokenizer on load
MAX_CONTEXT = 32768  # model card context length (config allows 40960 positions; we stay at the documented limit)


class InputTooLong(ValueError):
    def __init__(self, over: list[tuple[int, int]], max_length: int, detail: str = ""):
        self.over, self.max_length = over, max_length  # [(pair index, token count)]
        detail = detail or f"pair {over[0][0]}: {over[0][1]} tokens"
        super().__init__(f"{len(over)} input(s) exceed max_length={max_length} tokens (template included; "
                         f"nothing was truncated). {detail}")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def length_batches(lengths: list[int], max_tokens: int, max_size: int = 64):
    """Length-sorted index batches; padded size (rows x longest) stays under max_tokens."""
    batch = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if batch and ((len(batch) + 1) * lengths[i] > max_tokens or len(batch) == max_size):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Scorer:
    def __init__(self, adapter=None, device=None, dtype="float32", max_length=8192, max_batch_tokens=16384,
                 max_batch_size=64, model_id=MODEL_ID, revision=MODEL_REVISION):
        if not 0 < max_length <= MAX_CONTEXT:
            raise ValueError(f"max_length must be in (0, {MAX_CONTEXT}]")
        self.device, self.max_length = device or default_device(), max_length
        self.max_batch_tokens, self.max_batch_size = max_batch_tokens, max_batch_size
        self.tokenizer = tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if (tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")) != (YES_ID, NO_ID):
            raise RuntimeError("yes/no token ids differ from the pinned checkpoint's LogitScore config")
        self.pad_id = tok.pad_token_id
        self.prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = tok.encode(SUFFIX, add_special_tokens=False)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(self.device).eval()
        self.meta = {"model": model_id, "revision": revision, "adapter": str(adapter) if adapter else None,
                     "adapter_sha256": sha256_file(Path(adapter) / "adapter_model.safetensors") if adapter else None,
                     "device": self.device, "dtype": dtype, "max_length": max_length,
                     "truncation": "none (overlength input raises InputTooLong)"}

    def encode(self, texts: list[str], check=True) -> list[list[int]]:
        """Official Transformers-reference tokenization: prefix + tokens(pair text) + suffix."""
        body = self.tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
        ids = [self.prefix_ids + b + self.suffix_ids for b in body]
        over = [(i, len(x)) for i, x in enumerate(ids) if len(x) > self.max_length]
        if over and check:
            raise InputTooLong(over, self.max_length)
        return ids

    def forward(self, ids: list[list[int]]) -> torch.Tensor:
        """s = z_yes - z_no at the final position for one batch (left padding => position -1 is the last
        real token of every row). Differentiable; callers choose no_grad/inference_mode."""
        n, width = len(ids), max(map(len, ids))
        input_ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for r, x in enumerate(ids):
            input_ids[r, width - len(x):] = torch.tensor(x)
            mask[r, width - len(x):] = 1
        out = self.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device),
                         logits_to_keep=1, use_cache=False)
        z = out.logits[:, -1, [YES_ID, NO_ID]].float()
        return z[:, 0] - z[:, 1]

    def batches(self, lengths: list[int]):
        return length_batches(lengths, self.max_batch_tokens, self.max_batch_size)

    def score(self, texts: list[str]) -> tuple[list[float], dict]:
        t0 = time.perf_counter()
        ids = self.encode(texts)
        scores, stats = self.score_ids(ids)
        return scores, stats | {"tokenize_ms": 1e3 * (time.perf_counter() - t0) - stats["model_ms"]}

    @torch.inference_mode()
    def score_ids(self, ids: list[list[int]]) -> tuple[list[float], dict]:
        lengths = [len(x) for x in ids]
        t1 = time.perf_counter()
        scores, padded, n_batches = [0.0] * len(ids), 0, 0
        for b in self.batches(lengths):
            for i, v in zip(b, self.forward([ids[i] for i in b]).tolist()):  # .tolist() waits for the device
                scores[i] = v
            padded += len(b) * max(lengths[i] for i in b)
            n_batches += 1
        t2 = time.perf_counter()
        return scores, {"pairs": len(ids), "batches": n_batches, "input_tokens": sum(lengths), "padded_tokens": padded,
                        "max_pair_tokens": max(lengths, default=0), "model_ms": 1e3 * (t2 - t1)}
