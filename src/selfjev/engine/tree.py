"""Shared-prefix tree for Qwen3.5 (hybrid Gated DeltaNet + gated full attention): training and scoring.

Token ids are exactly Qwen35Scorer.entry's (challenger-state-first-v1), so the adapters trained here are served by
the existing forked-cache inference. One packed row per state (build_tree): root = system prompt + document, then
per question the longest common prefix of its branches (instruction + "Proposed answer:"), then one leaf per candidate.
- Full-attention layers read the packed row through tree_mask.
- Gated DeltaNet layers cannot take a mask: they run level by level (roots; every question segment from its root's
  final recurrent and conv state; every leaf from its question's), with gradients through those states. Right padding
  inside a level leaves the state untouched (q = k = v = 0, beta = 0, g = 0).
Each leaf therefore equals its standalone sequence (root + question + leaf) up to kernel rounding, and the text is
encoded once per state instead of once per candidate (tests/test_qwen35_tree.py checks scores and gradients).
"""

import time
from collections import Counter
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as chunk_rule  # fla if installed

from ..core.schemas import InputTooLong, parse_question


def build_tree(root, questions):
    """root: token ids; questions: [(question token ids, [leaf token ids, ...]), ...] ->
    {ids, pos, seg, parent, leaves, root}. Segments are laid out depth-first, so ancestors always come first."""
    ids, pos, seg, parent, leaves, r = list(root), list(range(len(root))), [0] * len(root), [-1], [], len(root)
    for q_ids, leaf_list in questions:
        g = len(parent)
        parent.append(0)
        ids += q_ids
        pos += range(r, r + len(q_ids))
        seg += [g] * len(q_ids)
        for l_ids in leaf_list:
            h = len(parent)
            parent.append(g)
            ids += l_ids
            pos += range(r + len(q_ids), r + len(q_ids) + len(l_ids))
            seg += [h] * len(l_ids)
            leaves.append(len(ids) - 1)
    return {"ids": ids, "pos": pos, "seg": seg, "parent": parent, "leaves": leaves, "root": r}


def tree_mask(tree, *, branches_only=False) -> torch.Tensor:
    """[T, T] bool, True = may attend: the key's segment is the query's own segment or an ancestor, and it is not
    later in the sequence (ancestor segments are entirely earlier)."""
    par = tree["parent"]
    anc = torch.eye(len(par), dtype=torch.bool)
    for g in range(1, len(par)):
        anc[g] |= anc[par[g]]
    # Cached inference already knows every branch may see the real root. Build only
    # branch-to-branch visibility: no quadratic allocation in document length.
    seg = torch.tensor(tree["seg"][tree["root"] :] if branches_only else tree["seg"])
    return anc[seg][:, seg] & torch.ones(len(seg), len(seg), dtype=torch.bool).tril()


def leaf_paths(tree) -> list[list[int]]:
    """Token indices of the standalone sequence each leaf stands for (root + question + leaf): used by tests."""
    par, seg = tree["parent"], tree["seg"]
    out = []
    for leaf in tree["leaves"]:
        chain, g = set(), seg[leaf]
        while g >= 0:
            chain.add(g)
            g = par[g]
        out.append([i for i in range(leaf + 1) if seg[i] in chain])
    return out


def split_branches(branches):
    """One question's branches -> (shared segment, leaves): their longest common prefix; every leaf keeps >= 1 token."""
    n, k = min(map(len, branches)) - 1, 0
    while k < n and all(b[k] == branches[0][k] for b in branches):
        k += 1
    return branches[0][:k], [b[k:] for b in branches]


def encode_items(sc, examples, max_length):
    """-> (items, roots, dropped) in the training schema (selfjev.training) (identical states share one root). Questions whose
    longest standalone sequence exceeds max_length are dropped and counted, never truncated."""
    keys, roots, items, dropped, before = {}, [], [], Counter(), sc.max_length
    sc.max_length = max_length
    try:
        for ex in examples:
            q = parse_question({"id": "q", **ex["question"]})
            try:
                e = sc.entry(ex["state"], q)
            except InputTooLong:
                dropped[ex["family"]] += 1
                continue
            if ex["state"] not in keys:
                keys[ex["state"]] = len(roots)
                roots.append(e["root"])
            qseg, leaves = split_branches(e["branches"])
            cids = [c.id for c in q.candidates]
            target = {"binary": lambda t: t, "multiclass": cids.index, "multilabel": lambda t: [c in t for c in cids]}[q.type](ex["target"])  # noqa: B023 lambda called right away
            items.append(
                {
                    "id": ex["id"],
                    "family": ex["family"],
                    "type": q.type,
                    "state": keys[ex["state"]],
                    "q": qseg,
                    "ids": leaves,
                    "target": target,
                    "candidate_ids": cids,
                }
            )
    finally:
        sc.max_length = before
    return items, roots, dict(dropped)


def pack(trees, pad, device):
    """Index tensors for one micro-batch of trees, shared by every layer: ids / pos [B, T], mask [B, 1, T, T]
    (tree_mask; pads see only themselves), leaves (flat indices of the readout tokens) and, per depth (roots,
    question segments, leaves), the DeltaNet levels (idx [N, L] flat token indices, valid [N, L], lens [N],
    parent [N] = index into the previous level, None for roots)."""
    B, T = len(trees), max(len(t["ids"]) for t in trees)
    ids, pos = torch.full((B, T), pad, dtype=torch.long), torch.zeros((B, T), dtype=torch.long)
    mask = torch.eye(T, dtype=torch.bool).repeat(B, 1, 1)
    nodes, where, leaves = [[], [], []], {}, []
    for b, t in enumerate(trees):
        n, seg, par = len(t["ids"]), t["seg"], t["parent"]
        ids[b, :n], pos[b, :n], mask[b, :n, :n] = torch.tensor(t["ids"]), torch.tensor(t["pos"]), tree_mask(t)
        start, length = {}, Counter(seg)
        for i, g in enumerate(seg):
            start.setdefault(g, i)
        for g in range(len(par)):  # build_tree numbers parents before children
            d, p = (0, None) if par[g] < 0 else (where[b, par[g]][0] + 1, where[b, par[g]][1])
            where[b, g] = (d, len(nodes[d]))
            nodes[d].append((b * T + start.get(g, 0), length[g], p))
        leaves += [b * T + i for i in t["leaves"]]
    levels = []
    for d, ns in enumerate(n for n in nodes if n):
        s, ln = torch.tensor([x[0] for x in ns]), torch.tensor([x[1] for x in ns])
        ar = torch.arange(max(1, int(ln.max())))
        valid = ar[None] < ln[:, None]
        parent = None if d == 0 else torch.tensor([x[2] for x in ns], device=device)
        levels.append((torch.where(valid, s[:, None] + ar, s[:, None]).to(device), valid.to(device), ln.to(device), parent))
    return SimpleNamespace(
        ids=ids.to(device), pos=pos.to(device), mask=mask[:, None].to(device), leaves=torch.tensor(leaves, device=device), levels=levels
    )


def deltanet(mod, x, levels):
    """Qwen3_5GatedDeltaNet.forward over a packed tree, level by level. x: [B, T, d] after input_layernorm."""
    B, T, _ = x.shape
    flat, K, C = x.reshape(B * T, -1), mod.conv_kernel_size, mod.conv_dim
    out = flat.new_zeros(B * T, mod.out_proj.out_features)
    state = ctx = None
    for idx, valid, lens, parent in levels:
        N, L = idx.shape
        h, m = flat[idx], valid[..., None]
        qkv = mod.in_proj_qkv(h) * m
        z = mod.in_proj_z(h).reshape(N, L, -1, mod.head_v_dim)
        b, a = mod.in_proj_b(h), mod.in_proj_a(h)
        prior = qkv.new_zeros(N, C, K - 1) if parent is None else ctx[parent]  # the path's last K-1 conv inputs
        ext = torch.cat([prior, qkv.transpose(1, 2)], -1)
        ctx = ext.gather(2, (lens[:, None] + torch.arange(K - 1, device=x.device))[:, None, :].expand(N, C, K - 1))
        conv = (F.silu(F.conv1d(ext.to(mod.conv1d.weight.dtype), mod.conv1d.weight, groups=C)).to(qkv.dtype).transpose(1, 2)) * m
        q, k, v = torch.split(conv, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q, k = q.reshape(N, L, -1, mod.head_k_dim), k.reshape(N, L, -1, mod.head_k_dim)
        v = v.reshape(N, L, -1, mod.head_v_dim)
        beta = b.sigmoid() * m
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias) * m
        if mod.num_v_heads // mod.num_k_heads > 1:
            q = q.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
            k = k.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
        o, state = chunk_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None if parent is None else state[parent],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o = mod.out_proj(mod.norm(o.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(N, L, -1))
        keep = valid.reshape(-1)
        out = out.index_copy(0, idx.reshape(-1)[keep], o.reshape(N * L, -1)[keep])
    return out.view(B, T, -1)


def _layer(layer, h, cos, sin, mask, levels):
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        x = deltanet(layer.linear_attn, x, levels)
    else:
        x = layer.self_attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=mask)[0]
    h = h + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def score(sc, trees, grad_checkpoint=True):
    """Leaf scores (z_yes - z_no, Qwen35Scorer.readout) of a list of build_tree dicts, in leaf order.
    Per-layer activation checkpointing while training (HF's own drops the cache path this replaces)."""
    dec, p = sc.decoder(), pack(trees, sc.pad, sc.device)
    h = dec.embed_tokens(p.ids)
    cos, sin = dec.rotary_emb(h, p.pos[None].expand(3, -1, -1))
    for layer in dec.layers:
        if grad_checkpoint and torch.is_grad_enabled():
            h = checkpoint(_layer, layer, h, cos, sin, p.mask, p.levels, use_reentrant=False)
        else:
            h = _layer(layer, h, cos, sin, p.mask, p.levels)
    h = dec.norm(h)
    return sc.readout(h.reshape(-1, h.shape[-1])[p.leaves])


class TreeServer:
    """Serving with the training tree, forward only: per request one tree (text once, each question once, then each
    candidate's own tokens), LoRA merged into the weights. Requests are packed by length under max_batch_tokens."""

    def __init__(self, adapter, max_length=32768, max_batch_tokens=16384):
        from .qwen35 import Qwen35Scorer

        self.sc = Qwen35Scorer("qwen35_4b", adapter=adapter, max_length=max_length)
        self.sc.model = self.sc.model.merge_and_unload()
        self.tokenizer, self.device, self.max_batch_tokens = self.sc.tokenizer, self.sc.device, max_batch_tokens
        self.meta = self.sc.meta | {
            "adapter_merged": True,
            "architecture": "shared-prefix tree, forward only: text once per request, each question once, then each candidate",
        }

    @torch.inference_mode()
    def score_requests(self, reqs):
        t0 = time.perf_counter()
        trees, sizes = [], []
        for r in reqs:
            es = [self.sc.entry(r.state, q) for q in r.questions]  # InputTooLong beyond max_length
            trees.append(build_tree(es[0]["root"], [split_branches(e["branches"]) for e in es]))
            sizes.append([e["n"] for e in es])
        batches, cur = [], []
        for i in sorted(range(len(trees)), key=lambda i: len(trees[i]["ids"])):
            if cur and (len(cur) + 1) * len(trees[i]["ids"]) > self.max_batch_tokens:
                batches.append(cur)
                cur = []
            cur.append(i)
        batches.append(cur)
        t1 = time.perf_counter()
        flat = {}
        for b in batches:
            s, k = score(self.sc, [trees[j] for j in b], grad_checkpoint=False).float().tolist(), 0
            for j in b:
                flat[j], k = s[k : k + len(trees[j]["leaves"])], k + len(trees[j]["leaves"])
        t2 = time.perf_counter()
        per = []
        for j, ns in enumerate(sizes):
            it = iter(flat[j])
            per.append([[next(it) for _ in range(n)] for n in ns])
        return per, {
            "pairs": sum(map(sum, sizes)),
            "batches": len(batches),
            "input_tokens": sum(len(t["ids"]) for t in trees),
            "padded_tokens": sum(len(b) * max(len(trees[j]["ids"]) for j in b) for b in batches),
            "tokenize_ms": 1e3 * (t1 - t0),
            "model_ms": 1e3 * (t2 - t1),
        }
