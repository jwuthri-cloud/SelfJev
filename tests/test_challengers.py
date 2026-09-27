"""Small deterministic checks for cache branching, padding and buffer precision (the Qwen3.5 scorer)."""

import torch
from transformers import DynamicCache, Qwen3Config

from selfjev.challengers import fork_cache, padded


def test_fork_preserves_root_and_independent_rows():
    cache = DynamicCache(config=Qwen3Config(num_hidden_layers=1))
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn_like(k)
    cache.update(k, v, 0)
    branched = fork_cache(cache, 3)
    assert branched.layers[0].keys.shape == (3, 2, 4, 8)
    branched.layers[0].keys[0].zero_()
    assert torch.equal(cache.layers[0].keys, k)
    assert torch.equal(branched.layers[0].keys[1], k[0])


def test_fork_linear_state_independence():
    from transformers import Qwen3_5TextConfig

    cache = DynamicCache(config=Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]))
    cache.update_conv_state(torch.randn(1, 12, 4), 0)
    cache.update_recurrent_state(torch.randn(1, 2, 4, 4), 0)
    cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 1)
    branch = fork_cache(cache, 2)
    assert branch.layers[0].conv_states[0].shape[0] == 2
    old = cache.layers[0].recurrent_states[0].clone()
    branch.layers[0].recurrent_states[0][0].zero_()
    assert torch.equal(cache.layers[0].recurrent_states[0], old)
    assert torch.equal(branch.layers[0].recurrent_states[0][1], old[0])


def test_padding_retains_all_tokens():
    ids, mask = padded([[3, 4, 5], [6]], 0, "cpu")
    assert ids.tolist() == [[3, 4, 5], [6, 0, 0]]
    assert mask.sum(1).tolist() == [3, 1]


def test_native_rotary_precision_survives_bf16_device_transfer():
    from selfjev.challengers import place_model

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
