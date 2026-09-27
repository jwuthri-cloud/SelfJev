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
