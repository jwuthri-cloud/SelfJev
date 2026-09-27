"""The Qwen3.5 tree equals standalone sequences: scores and gradients, on a tiny random hybrid model (CPU, fp32).
Covers sibling isolation, right padding inside a level, a root shorter than the conv kernel and two states per batch."""

import random

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from selfjev import qwen35_tree
from selfjev.tree import build_tree, leaf_paths


class Stub:  # the three things qwen35_tree.score needs from ChallengerScorer
    pad, device = 0, "cpu"

    def __init__(self, lm):
        self.lm = lm

    def decoder(self):
        return self.lm.model

    def readout(self, h):
        z = h.float() @ self.lm.lm_head.weight[[1, 2]].float().T
        return z[..., 0] - z[..., 1]


def test_tree_matches_full_sequences():
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        initializer_range=0.2,
    )
    lm = Qwen3_5ForCausalLM(cfg).float()
    lm.train()  # dropout is 0 in this config; train mode exercises the checkpointed path
    sc, rnd = Stub(lm), random.Random(0)
    tok = lambda n: [rnd.randrange(3, 64) for _ in range(n)]
    trees = [build_tree(tok(5), [(tok(3), [tok(2), tok(4), tok(1)]), (tok(1), [tok(3), tok(3)])]), build_tree(tok(2), [(tok(6), [tok(1)])])]
    w = torch.randn(6)

    tree_scores = qwen35_tree.score(sc, trees)
    (tree_scores * w).sum().backward()
    tree_grads = {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}
    lm.zero_grad()

    paths = [[t["ids"][i] for i in p] for t in trees for p in leaf_paths(t)]
    full = torch.stack([sc.readout(lm.model(input_ids=torch.tensor([p]), use_cache=False).last_hidden_state[0, -1]) for p in paths])
    (full * w).sum().backward()

    assert torch.allclose(tree_scores, full, atol=1e-4), (tree_scores, full)
    assert tree_grads.keys() == {n for n, p in lm.named_parameters() if p.grad is not None}
    for n, p in lm.named_parameters():
        if p.grad is not None:
            assert (tree_grads[n] - p.grad).abs().max() <= 1e-4 * p.grad.abs().max(), n  # fp32 summation-order noise


def test_split_branches_keeps_a_token_per_leaf():
    assert qwen35_tree.split_branches([[1, 2, 3, 9], [1, 2, 4, 9]]) == ([1, 2], [[3, 9], [4, 9]])
    assert qwen35_tree.split_branches([[1, 2, 3]]) == ([1, 2], [[3]])  # binary: one branch
    assert qwen35_tree.split_branches([[1, 2], [1, 2]]) == ([1], [[2], [2]])


def test_tree_server_scores_match_standalone_sequences():
    """TreeServer.score_requests (the serving path): mixed requests, questions and candidate counts, mapped back in order."""
    from selfjev.qwen35_tree import TreeServer
    from selfjev.schemas import parse_request

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        initializer_range=0.2,
    )
    lm = Qwen3_5ForCausalLM(cfg).float().eval()

    class Tiny(Stub):  # ChallengerScorer.entry's shape with a character-level "tokenizer"
        max_length = 4096

        def entry(self, state, q):
            tok = lambda t: [3 + ord(c) % 290 for c in t]
            answers = ["Yes"] if q.type == "binary" else [c.description for c in q.candidates]
            return {"root": [1, *tok(state)], "branches": [[*tok(q.instruction), 2, *tok(a)] for a in answers], "n": len(answers)}

    ts = object.__new__(TreeServer)
    ts.sc, ts.max_batch_tokens = Tiny(lm), 64  # small budget: several packed batches
    cands = lambda n: [{"id": f"c{i}", "description": f"option number {i}"} for i in range(n)]
    reqs = [
        parse_request(r)
        for r in (
            {
                "state": "short text",
                "questions": [
                    {"id": "a", "type": "binary", "instruction": "ok?"},
                    {"id": "b", "type": "multiclass", "instruction": "which?", "candidates": cands(3)},
                ],
            },
            {
                "state": "a much longer text about refunds and outages",
                "questions": [{"id": "c", "type": "multilabel", "instruction": "which apply?", "candidates": cands(4)}],
            },
            {"state": "third", "questions": [{"id": "d", "type": "multiclass", "instruction": "pick", "candidates": cands(2)}]},
        )
    ]
    per, meta = ts.score_requests(reqs)
    assert [[len(s) for s in r] for r in per] == [[1, 3], [4], [2]] and meta["batches"] > 1
    for r, scores in zip(reqs, per):
        for q, s in zip(r.questions, scores):
            e = ts.sc.entry(r.state, q)
            full = [
                ts.sc.readout(lm.model(input_ids=torch.tensor([e["root"] + b]), use_cache=False).last_hidden_state[0, -1]).item()
                for b in e["branches"]
            ]
            assert torch.allclose(torch.tensor(s), torch.tensor(full), atol=1e-4), (q.id, s, full)
