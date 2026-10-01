"""Shared-prefix tree for Qwen3.5 (hybrid Gated DeltaNet + gated full attention): training and scoring.

Token ids are exactly Qwen35Scorer.entry's (challenger-state-first-v1), so the adapters trained here are served by
TreeServer below (or merged, by selfjev.engine.vllm). One packed row per state (build_tree): root = system prompt + document, then
per question the longest common prefix of its branches (instruction + "Proposed answer:"), then one leaf per candidate.
- Full-attention layers read the packed row through tree_mask.
- Gated DeltaNet layers cannot take a mask: they run level by level (roots; every question segment from its root's
  final recurrent and conv state; every leaf from its question's), with gradients through those states. Right padding
  inside a level leaves the state untouched (q = k = v = 0, beta = 0, g = 0).
Each leaf therefore equals its standalone sequence (root + question + leaf) up to kernel rounding, and the text is
encoded once per state instead of once per candidate (tests/engine/test_tree.py checks scores and gradients).
Images in the root (Qwen35Scorer.root): the vision tower's embeddings replace their placeholder tokens and the root gets
Qwen3.5's 3D M-RoPE positions (root_rope); the questions continue after the root's highest position.
"""

import os
import time
from collections import Counter
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as chunk_rule  # fla if installed

from ..core.schemas import InputTooLong, is_image, parse_question


class Root(list):
    """A root's token ids plus its images [(data URL, grid_thw, start)]: training keeps the URL, not the pixels (a corpus
    of pixel tensors does not fit in memory); score() makes them per batch."""

    images = ()


MERGE = 2  # Qwen3.5 vision spatial_merge_size: one LM token per 2x2 patches


def root_rope(n, images):
    """[3, n] M-RoPE positions of a root with images [(pixel_values, grid_thw [1, 3], start)], as
    Qwen3_5Model.get_rope_index: text advances all three rows by 1; an image's tokens sit at (p, p + row, p + col), then
    p advances by max(rows, cols)."""
    pos, p, i = torch.zeros(3, n, dtype=torch.long), 0, 0
    for _, grid, start in images:
        t, h, w = (int(x) for x in grid.reshape(-1))
        assert t == 1, "images only, not video"
        h, w = h // MERGE, w // MERGE
        pos[:, i:start], p = torch.arange(p, p + start - i), p + start - i
        k = start + h * w
        pos[0, start:k], pos[1, start:k], pos[2, start:k] = p, p + torch.arange(h).repeat_interleave(w), p + torch.arange(w).repeat(h)
        p, i = p + max(h, w), k
    pos[:, i:] = torch.arange(p, p + n - i)
    return pos


def build_tree(root, questions, images=()):
    """root: token ids; questions: [(question token ids, [leaf token ids, ...]), ...]; images: Qwen35Scorer.root's ->
    {ids, pos, seg, parent, leaves, root, rope, images}. Segments are laid out depth-first, so ancestors always come first.
    rope: the root's 3D positions when it has images (else pos, repeated)."""
    rope = root_rope(len(root), images) if images else None
    r = len(root) if rope is None else int(rope.max()) + 1
    ids, pos, seg, parent, leaves = list(root), list(range(len(root))), [0] * len(root), [-1], []
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
    return {"ids": ids, "pos": pos, "seg": seg, "parent": parent, "leaves": leaves, "root": len(root), "rope": rope, "images": images}


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
    k = min(len(os.path.commonprefix(branches)), min(map(len, branches)) - 1)  # commonprefix: min/max compare in C, then one pair
    return branches[0][:k], [b[k:] for b in branches]


def encode_items(sc, examples, max_length):
    """-> (items, roots, dropped) in the training schema (selfjev.training) (identical states share one root). Questions whose
    longest standalone sequence exceeds max_length are dropped and counted, never truncated."""
    keys, roots, items, dropped, before = {}, [], [], Counter(), sc.max_length
    sc.max_length = max_length
    try:
        for ex in examples:
            q = parse_question({"id": "q", **ex["question"]})
            state = tuple(ex["state"]) if isinstance(ex["state"], list) else ex["state"]  # parts: text and image URLs
            try:
                e = sc.entry(state, q)
            except InputTooLong:
                dropped[ex["family"]] += 1
                continue
            if state not in keys:
                keys[state] = len(roots)
                roots.append(Root(e["root"]))
                urls = [p for p in ((state,) if isinstance(state, str) else state) if is_image(p)]
                roots[-1].images = [(u, g, s) for u, (_, g, s) in zip(urls, e.get("images", ()), strict=True)]
            qseg, leaves = split_branches(e["branches"])
            cids = [c.id for c in q.candidates]
            target = {"binary": lambda t: t, "multiclass": cids.index, "multilabel": lambda t: [c in t for c in cids]}[q.type](ex["target"])  # noqa: B023 lambda called right away
            items.append(
                {
                    "id": ex["id"],
                    "family": ex["family"],
                    "type": q.type,
                    "state": keys[state],
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
    """Index tensors for one micro-batch of trees, shared by every layer: ids [B, T], pos [3, B, T], mask [B, 1, T, T]
    (tree_mask; pads see only themselves), leaves (flat indices of the readout tokens) and, per depth (roots,
    question segments, leaves), the DeltaNet levels (idx [N, L] flat token indices, valid [N, L], lens [N],
    parent [N] = index into the previous level, None for roots)."""
    B, T = len(trees), max(len(t["ids"]) for t in trees)
    ids, pos = torch.full((B, T), pad, dtype=torch.long), torch.zeros((3, B, T), dtype=torch.long)
    mask = torch.eye(T, dtype=torch.bool).repeat(B, 1, 1)
    nodes, where, leaves = [[], [], []], {}, []
    for b, t in enumerate(trees):
        n, seg, par = len(t["ids"]), t["seg"], t["parent"]
        ids[b, :n], pos[:, b, :n], mask[b, :n, :n] = torch.tensor(t["ids"]), torch.tensor(t["pos"]), tree_mask(t)
        if t.get("rope") is not None:
            pos[:, b, : t["root"]] = t["rope"]
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


def l2norm(x):
    """fla's in-kernel q / k normalization (fp32, eps 1e-6, back to x's dtype), in torch. fla's l2norm kernel autotunes per
    input size (constexpr NB = rows / 65,536, rows = sequences x length x heads), and a batch of requests brings new rows
    with every traffic mix: bursts of 32 requests waited 19-29 s instead of 10 (A10G, 2026-10-01, JOURNAL 15:30)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def bucketed_rule(q, k, v, g, beta, initial_state):
    """chunk_rule over N sequences, N padded with empty ones to a power of 2: fla's kernels autotune per batch size (0.4-1.1 s
    on an L40S for each new N) and a tree's N (roots, questions or leaves in a batch) changes with every traffic mix.
    Buckets bound that to ~15 sizes, which TreeServer.warm_kernels tunes at start; q and k are normalized here (l2norm)."""
    n = q.shape[0]
    pad = (1 << (n - 1).bit_length()) - n

    def grow(t):
        return t if t is None or not pad else torch.cat([t, t.new_zeros((pad, *t.shape[1:]))])

    o, state = chunk_rule(grow(l2norm(q)), grow(l2norm(k)), grow(v), g=grow(g), beta=grow(beta), initial_state=grow(initial_state),
                          output_final_state=True, use_qk_l2norm_in_kernel=False)  # fmt: skip
    return o[:n], state[:n]


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
        o, state = bucketed_rule(q, k, v, g, beta, None if parent is None else state[parent])
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
    for b, t in enumerate(trees):
        for pixels, grid, start in t.get("images", ()):  # ponytail: one image per vision call; batch them if images dominate
            if isinstance(pixels, str):
                pixels = sc.image(pixels)[0]
            with torch.no_grad():  # frozen tower
                e = sc.visual(pixels.to(sc.device, sc.visual.dtype), grid_thw=grid.to(sc.device)).pooler_output
            h[b, start : start + len(e)] = e.to(h.dtype)
    cos, sin = dec.rotary_emb(h, p.pos)
    for layer in dec.layers:
        if grad_checkpoint and torch.is_grad_enabled():
            h = checkpoint(_layer, layer, h, cos, sin, p.mask, p.levels, use_reentrant=False)
        else:
            h = _layer(layer, h, cos, sin, p.mask, p.levels)
    h = dec.norm(h)
    return sc.readout(h.reshape(-1, h.shape[-1])[p.leaves])


class TreeServer:
    """Serving with the training tree, forward only: per request one tree (text once, each question once, then each
    candidate's own tokens). Requests are packed by length under max_batch_tokens.

    merge=True folds the LoRA into the weights (fastest, one model). merge=False keeps it as a PEFT adapter so more can
    be loaded next to it (load_adapter) and chosen per batch (set_adapter): the server does this for fine-tuned models.
    """

    def __init__(
        self,
        adapter,
        max_length=32768,
        max_batch_tokens=16384,
        merge=True,
        device="cuda",
        dtype="bfloat16",
        quantize=None,
        quantized_model=None,
    ):
        from .qwen35 import Qwen35Scorer

        # torch picks cuDNN attention on H100, and it fails to load there (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED, 2026-09-30):
        # use the memory-efficient kernel that A10G / L40S already run
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.sc = Qwen35Scorer(
            adapter=adapter, device=device, dtype=dtype, max_length=max_length, quantize=quantize, quantized_model=quantized_model
        )
        if merge and adapter and not (quantize or quantized_model):  # "" = bare base (zero-shot control); quantized models are merged
            self.sc.model = self.sc.model.merge_and_unload()
        self.merged, self.adapters = merge, {"default": str(adapter)}
        self.tokenizer, self.device, self.max_batch_tokens = self.sc.tokenizer, self.sc.device, max_batch_tokens
        self.meta = self.sc.meta | {
            "adapter_merged": merge,
            "architecture": "shared-prefix tree, forward only: text once per request, each question once, then each candidate",
        }

    def load_adapter(self, name: str, path: str):
        """Another LoRA on the same base weights (needs merge=False)."""
        if self.merged:
            raise RuntimeError("TreeServer(merge=True) serves one adapter; use merge=False to load more")
        if name not in self.adapters:
            self.sc.model.load_adapter(path, adapter_name=name)
            self.adapters[name] = str(path)

    def set_adapter(self, name: str = "default"):
        if not self.merged:
            self.sc.model.set_adapter(name)

    @torch.inference_mode()
    def warm_kernels(self):
        """Autotune fla's DeltaNet kernels for every batch-size bucket of bucketed_rule now, not on live requests: 42 s on an
        A10G the first time, then from Triton's disk cache (0.5 s). Nothing to do on the torch fallback (no fla)."""
        import importlib.util

        if importlib.util.find_spec("fla") is None:
            return
        mod = next(layer.linear_attn for layer in self.sc.decoder().layers if layer.block_type == "linear_attention")
        dt, n = getattr(torch, self.sc.dtype), 1
        while n <= 512:  # ponytail: 512 sequences' states are 1 GB; a larger batch (rare) tunes its bucket on first use
            z = {"device": self.device, "dtype": dt}
            L, H = max(1, min(64, 2**14 // n)), mod.num_v_heads  # T is not in the kernels' autotune keys: keep tensors small
            q, v = torch.zeros(n, L, H, mod.head_k_dim, **z), torch.zeros(n, L, H, mod.head_v_dim, **z)
            g = torch.zeros(n, L, H, device=self.device)
            for init in (None, torch.zeros(n, H, mod.head_k_dim, mod.head_v_dim, device=self.device)):  # roots, then deeper levels
                bucketed_rule(q, q, v, g, g.to(dt), init)
            n *= 2

    @torch.inference_mode()
    def score_requests(self, reqs):
        t0 = time.perf_counter()
        trees, sizes = [], []
        for r in reqs:
            es = [self.sc.entry(r.state, q) for q in r.questions]  # InputTooLong beyond max_length
            trees.append(build_tree(es[0]["root"], [split_branches(e["branches"]) for e in es], es[0].get("images", ())))
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
            "tokens_per_request": [len(t["ids"]) for t in trees],  # text once + each question + each candidate
            "padded_tokens": sum(len(b) * max(len(trees[j]["ids"]) for j in b) for b in batches),
            "tokenize_ms": 1e3 * (t1 - t0),
            "model_ms": 1e3 * (t2 - t1),
        }
