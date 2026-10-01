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


def test_image_is_read_upright_and_bad_images_are_input_errors():
    """A phone photo stored sideways with an EXIF orientation tag reaches the vision processor upright; an image the
    processor refuses is a 422 (ValidationError), not a 500."""
    import base64
    import io

    import pytest
    from PIL import Image

    from selfjev.core.schemas import ValidationError
    from selfjev.engine.qwen35 import Qwen35Scorer

    sc, seen = Qwen35Scorer.__new__(Qwen35Scorer), {}
    sc.dtype = "float32"

    def processor(images, return_tensors):
        seen["size"] = images[0].size
        if images[0].size[0] > 100 * images[0].size[1]:
            raise ValueError("absolute aspect ratio must be smaller than 200")
        return {"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    sc.__dict__["image_processor"] = processor  # the cached property, without a download
    img, buf = Image.new("RGB", (40, 20)), io.BytesIO()
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees to display
    img.save(buf, "JPEG", exif=exif)
    sc.image("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    assert seen["size"] == (20, 40)
    buf = io.BytesIO()
    Image.new("RGB", (4000, 10)).save(buf, "PNG")
    with pytest.raises(ValidationError):
        sc.image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
