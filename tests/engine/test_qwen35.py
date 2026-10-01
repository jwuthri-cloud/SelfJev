"""Buffer precision of the Qwen3.5 loader."""

import torch


def test_native_rotary_precision_survives_bf16_device_transfer():
    from selfjev.engine.qwen35 import place_model

    model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    frequency = torch.tensor([0.123456789, 0.000123456789], dtype=torch.float32)
    model.register_buffer("inv_freq", frequency.clone())
    place_model(model, "cpu", "bfloat16")
    assert model.weight.dtype == torch.bfloat16
    assert model.inv_freq.dtype == torch.float32
    assert torch.equal(model.inv_freq, frequency)
    place_model(model, "cpu", "float32")
    assert model.weight.dtype == torch.float32
    assert torch.equal(model.inv_freq, frequency)


def test_entry_branches_equal_full_string_tokenization():
    """entry() tokenizes the instruction once; ids must equal tokenizing each full branch string (as training always did).
    Needs the Qwen3.5 tokenizer in the local Hugging Face cache (no download); skipped otherwise."""
    import pytest
    from transformers import AutoTokenizer

    from selfjev.core.schemas import parse_question
    from selfjev.engine.qwen35 import BASE, Qwen35Scorer

    try:
        tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer not cached")
    sc = object.__new__(Qwen35Scorer)
    sc.tokenizer, sc.max_length = tok, 10**9
    odd = [" lead", "  two", "\nnl", "5 kg", "-1", "(A) x", "'s", "é", "́mark", "日本", "😀", ": c", "\n\nafter blank", "x" * 500]
    options = odd + [f"option number {i}: text {i}" for i in range(255 - len(odd))]
    for instruction in ["Pick one. Options: " + "; ".join(options), "?", "ends with space ", "ends with colon:", "ends\n"]:
        q = parse_question({"id": "q", "type": "multiclass", "instruction": instruction,
                            "candidates": [{"id": f"c{i}", "description": o} for i, o in enumerate(options)]})  # fmt: skip
        after = sc.root("state")[1]
        want = [sc.tokens(instruction + "\nProposed answer: " + o + after) for o in options]
        assert sc.entry("state", q)["branches"] == want
