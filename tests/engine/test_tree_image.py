"""An image in the root: the tree equals Hugging Face's own multimodal forward (vision embeddings + 3D M-RoPE) for every
leaf, on a tiny random Qwen3.5 (CPU, fp32), with a text-only tree in the same batch."""

import random

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from selfjev.engine import tree as qwen35_tree
from selfjev.engine.tree import build_tree, leaf_paths, root_rope

IMG, START, END = 60, 61, 62


class Stub:  # what qwen35_tree.score needs from Qwen35Scorer
    pad, device = 0, "cpu"

    def __init__(self, m):
        self.m, self.visual = m, m.model.visual

    def decoder(self):
        return self.m.model.language_model

    def readout(self, h):
        z = h.float() @ self.m.lm_head.weight[[1, 2]].float().T
        return z[..., 0] - z[..., 1]


def tiny_multimodal():
    text = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    text |= {"head_dim": 8, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2}
    text |= {"linear_num_value_heads": 4, "initializer_range": 0.2}
    rope = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]}
    text["rope_parameters"] = rope | {"mrope_interleaved": True}
    vision = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }
    vision |= {"out_hidden_size": 32, "num_position_embeddings": 16}
    tokens = {"image_token_id": IMG, "vision_start_token_id": START, "vision_end_token_id": END, "video_token_id": 63}
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, **tokens)
    return Qwen3_5ForConditionalGeneration(cfg).float()


def test_image_root_matches_full_multimodal_forward():
    torch.manual_seed(0)
    m = tiny_multimodal().eval()
    rnd = random.Random(0)
    tok = lambda n: [rnd.randrange(3, 59) for _ in range(n)]

    grid = torch.tensor([[1, 4, 6]])  # 2 x 3 LM tokens after the 2 x 2 merge
    pixels = torch.randn(24, 3 * 2 * 2 * 2)
    root = tok(4) + [START] + [IMG] * 6 + [END] + tok(2)
    images = ((pixels, grid, 5),)
    trees = [
        build_tree(root, [(tok(3), [tok(2), tok(1)]), (tok(2), [tok(3)])], images),
        build_tree(tok(3), [(tok(2), [tok(1), tok(2)])]),  # text-only neighbour: padding and mixed positions
    ]

    ids = torch.tensor([root])
    hf_pos = m.model.get_rope_index(ids, mm_token_type_ids=(ids == IMG).int(), image_grid_thw=grid)[0][:, 0]
    assert torch.equal(root_rope(len(root), images), hf_pos)

    with torch.no_grad():
        tree_scores = qwen35_tree.score(Stub(m), trees)
        full = []
        for t in trees:
            for path in leaf_paths(t):
                x = torch.tensor([[t["ids"][i] for i in path]])
                kw = {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": (x == IMG).int()} if t["images"] else {}
                h = m.model(input_ids=x, use_cache=False, **kw).last_hidden_state[0, -1]
                full.append(Stub(m).readout(h))
    assert torch.allclose(tree_scores, torch.stack(full), atol=1e-4), (tree_scores, full)


def test_linear_patch_embed_equals_conv3d():
    import copy

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from selfjev.engine.qwen35 import linear_patch_embed

    small = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32}
    vis = Qwen3_5VisionModel._from_config(Qwen3_5Config(vision_config=small).vision_config).eval()  # the tower alone
    grid, x = torch.tensor([[1, 4, 6]]), torch.randn(24, 3 * 2 * 16 * 16)
    with torch.no_grad():
        ref = vis(x, grid_thw=grid).pooler_output
        assert torch.allclose(linear_patch_embed(copy.deepcopy(vis))(x, grid_thw=grid).pooler_output, ref, atol=1e-5)


def test_training_on_images_matches_full_forward_gradients():
    """encode_items keeps an image as a URL in the root; score() makes its pixels per batch; the LoRA-side gradients (here
    every language-model weight) equal Hugging Face's full multimodal forward."""
    torch.manual_seed(1)
    m, rnd = tiny_multimodal(), random.Random(1)
    grid, pixels = torch.tensor([[1, 4, 6]]), torch.randn(24, 3 * 2 * 2 * 2)
    url = "data:image/png;base64,AAAA"
    tok = lambda n: [rnd.randrange(3, 59) for _ in range(n)]

    class Fake(Stub):  # what encode_items and score() need from Qwen35Scorer
        max_length = 1000

        def entry(self, state, q):
            root = tok(4) + [START] + [IMG] * 6 + [END]
            branches = [tok(3), tok(2)]
            return {"root": root, "branches": branches, "n": 2, "images": ((pixels, grid, 5),)}

        def image(self, u):
            assert u == url
            return pixels, grid

    sc = Fake(m)
    rows = [
        {
            "id": "a",
            "family": "f",
            "state": [url, "text part"],
            "question": {
                "type": "multiclass",
                "instruction": "q",
                "candidates": [{"id": "x", "description": "x"}, {"id": "y", "description": "y"}],
            },
            "target": "x",
        }
    ] * 2
    items, roots, dropped = qwen35_tree.encode_items(sc, rows, 100)
    assert len(roots) == 1 and roots[0].images == [(url, grid, 5)] and not dropped  # both rows share the one root
    from selfjev.training.batching import trees_for

    (tree,) = trees_for(items, roots)
    w = torch.randn(4)
    lm = m.model.language_model
    qwen35_tree.score(sc, [tree]).mul(w).sum().backward()
    tree_grads = {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}
    m.zero_grad()
    full = []
    for path in leaf_paths(tree):
        x = torch.tensor([[tree["ids"][i] for i in path]])
        kw = {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": (x == IMG).int()}
        full.append(sc.readout(m.model(input_ids=x, use_cache=False, **kw).last_hidden_state[0, -1]))
    torch.stack(full).mul(w).sum().backward()
    assert tree_grads and all(torch.allclose(g, dict(lm.named_parameters())[n].grad, atol=1e-4) for n, g in tree_grads.items())
