"""Shared-prefix tree for Qwen3.5 (hybrid Gated DeltaNet + gated full attention): training and scoring.

Token ids are exactly Qwen35Scorer.entry's (challenger-state-first-v1), so the adapters trained here are served by
TreeServer below (or merged, by selfjev.engine.vllm). One packed row per state (build_tree): root = system prompt + document, then
per question the longest common prefix of its branches (instruction + "Proposed answer:"), then one leaf per candidate
(a single candidate stays in its question's segment). A batch of trees is one flat row of real tokens (pack): norms,
MLPs and projections run once per token and never on padding.
- Full-attention layers (attention): the text attends causally, as in any sequence, on the fast kernel and with no mask
  tensor; only the branch tokens need a mask (their text and their ancestors, tree_mask), so nothing is quadratic in the
  text on the CPU and the text's attention costs what it costs every engine.
- Gated DeltaNet layers (deltanet) cannot take a mask: the recurrence runs level by level (roots; every question segment
  from its root's final state; every leaf from its question's), with gradients through those states. Right padding
  inside a level leaves the state untouched (q = k = v = 0, beta = 0, g = 0). Their convolution reads each token's
  predecessors on its own path.
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
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
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
    A question with one leaf (yes/no) is one segment, read at its end: a batch of such questions has no leaf level, a third
    less DeltaNet work per layer for a short request. rope: the root's 3D positions when it has images (else pos, repeated)."""
    rope = root_rope(len(root), images) if images else None
    r = len(root) if rope is None else int(rope.max()) + 1
    ids, pos, seg, parent, leaves = list(root), list(range(len(root))), [0] * len(root), [-1], []
    for q_ids, leaf_list in questions:
        g = len(parent)
        parent.append(0)
        if len(leaf_list) == 1:
            q_ids, leaf_list = q_ids + leaf_list[0], []
            leaves.append(len(ids) + len(q_ids) - 1)
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


def pack(trees, kernel, device):
    """Index tensors for one micro-batch of trees, shared by every layer. The hidden states are one flat row of real tokens
    [N, d], tree after tree: norms, MLPs, projections and the DeltaNet convolution never run on padding, so trees of
    unequal length share a forward at almost no cost. Only attention pads, to rows [B, T], and each DeltaNet level to
    its longest segment, by pointing at row N: the zero row deltanet() appends.
    -> ids [N]; for attention(): rows [B, T] (a row's tokens in the flat row; padding -> token 0), back [N] (each
    token's place in the padded rows), pos [3, B, T] and attn (root = the longest text; q [B, Nb] = each branch token's
    position in its row; mask [B, 1, Nb, T] = what it may attend to, its whole text and its ancestors (tree_mask); flat /
    keep put the branch outputs back); leaves (indices of the readout tokens); for deltanet(), per depth (roots, question
    segments, leaves), levels (idx [n, L] token indices, parent [n] = index into the previous level, None for roots) and
    fix / pred: the tokens whose `kernel` - 1 predecessors on their path are not the ones before them in the flat row
    (the first of every segment, and row N), with those predecessors (-> N where the path is shorter).
    A level's n is rounded up to a power of 2 with empty segments: fla's kernels autotune per batch size (0.4-1.1 s on an
    L40S for each new n) and n changes with every traffic mix; ~15 sizes remain, which TreeServer.warm_kernels tunes."""
    B, T, total = len(trees), max(len(t["ids"]) for t in trees), sum(len(t["ids"]) for t in trees)
    n_tok = torch.tensor([len(t["ids"]) for t in trees])
    nb = n_tok - torch.tensor([t["root"] for t in trees])
    rows, pos = torch.zeros((B, T), dtype=torch.long), torch.zeros((3, B, T), dtype=torch.long)
    bq, bmask = torch.zeros((B, int(nb.max())), dtype=torch.long), torch.zeros((B, int(nb.max()), T), dtype=torch.bool)
    bmask[:, :, 0] = True  # padding attends to one token: a fully masked row is NaN, and so is its gradient
    nodes, where, leaves, off = [[], [], []], {}, [], 0
    fix, pred = [total], [[total] * kernel]
    for b, t in enumerate(trees):
        n, r, seg, par = len(t["ids"]), t["root"], t["seg"], t["parent"]
        rows[b, :n], pos[:, b, :n] = torch.arange(off, off + n), torch.tensor(t["pos"])
        bq[b, : n - r], bmask[b, : n - r, :r], bmask[b, : n - r, r:n] = torch.arange(r, n), True, tree_mask(t, branches_only=True)
        if t.get("rope") is not None:
            pos[:, b, : t["root"]] = t["rope"]
        start, length, tail = {}, Counter(seg), {-1: []}
        for i, g in enumerate(seg):
            start.setdefault(g, i)
        for g in range(len(par)):  # build_tree numbers parents before children
            d, p = (0, None) if par[g] < 0 else (where[b, par[g]][0] + 1, where[b, par[g]][1])
            where[b, g] = (d, len(nodes[d]))
            nodes[d].append((off + start.get(g, 0), length[g], p))
            own = range(off + start.get(g, 0), off + start.get(g, 0) + length[g])
            for j, tok in enumerate(own[: kernel - 1]):
                before = (tail[par[g]] + list(own[:j]))[-(kernel - 1) :]
                fix.append(tok)
                pred.append([total] * (kernel - 1 - len(before)) + before + [tok])
            tail[g] = (tail[par[g]] + list(own[-(kernel - 1) :]))[-(kernel - 1) :]
        leaves += [off + i for i in t["leaves"]]
        off += n
    levels = []
    for d, ns in enumerate(n for n in nodes if n):
        ns = ns + [(0, 0, 0)] * ((1 << (len(ns) - 1).bit_length()) - len(ns))
        s, ln = torch.tensor([x[0] for x in ns]), torch.tensor([x[1] for x in ns])
        ar = torch.arange(max(1, int(ln.max())))
        parent = None if d == 0 else torch.tensor([x[2] for x in ns], device=device)
        levels.append((torch.where(ar[None] < ln[:, None], s[:, None] + ar, total).to(device), parent))
    keep = torch.arange(bq.shape[1])[None] < nb[:, None]
    attn = SimpleNamespace(
        root=max(t["root"] for t in trees),
        q=bq.to(device),
        mask=bmask[:, None].to(device),
        flat=(torch.arange(B)[:, None] * T + bq)[keep].to(device),
        keep=keep.reshape(-1).to(device),
    )
    return SimpleNamespace(
        ids=torch.tensor([i for t in trees for i in t["ids"]], device=device),
        rows=rows.to(device),
        back=(torch.arange(B)[:, None] * T + torch.arange(T))[torch.arange(T)[None] < n_tok[:, None]].to(device),
        pos=pos.to(device),
        attn=attn,
        leaves=torch.tensor(leaves, device=device),
        levels=levels,
        fix=torch.tensor(fix, device=device),
        pred=torch.tensor(pred, device=device),
    )


def l2norm(x):
    """fla's in-kernel q / k normalization (fp32, eps 1e-6, back to x's dtype), in torch. fla's l2norm kernel autotunes per
    input size (constexpr NB = rows / 65,536, rows = sequences x length x heads), and a batch of requests brings new rows
    with every traffic mix: bursts of 32 requests waited 19-29 s instead of 10 (A10G, 2026-10-01, JOURNAL 15:30)."""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)  # no fp32 copy of x
    return (x * torch.rsqrt(n.square() + 1e-6)).to(x.dtype)


def rms_norm(mod, x):
    """Qwen3_5RMSNorm.forward (fp32 arithmetic on x's values, then x's dtype), without the fp32 copies of x and of its
    square: the mean square is a reduction. Worth 5-6 % of a request with gated_norm (L40S, JOURNAL 2026-10-05); in
    bf16 it differs from the module on 0.01 % of the values, by one rounding step."""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    return (x * torch.rsqrt(n.square() / x.shape[-1] + mod.eps) * (1.0 + mod.weight.float())).type_as(x)


def gated_norm(mod, x, gate):
    """Qwen3_5RMSNormGated.forward, rounded to x's dtype where it rounds, without the same two fp32 copies."""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    h = mod.weight * (x * torch.rsqrt(n.square() / x.shape[-1] + mod.variance_epsilon)).to(x.dtype)
    return (h * F.silu(gate.float())).to(x.dtype)


def deltanet(mod, x, p):
    """Qwen3_5GatedDeltaNet.forward over packed trees (p: pack's). x: [N, d] after input_layernorm, the flat row.
    Everything per token runs once on the flat row plus one zero row, which is what padding points at: projections, the
    causal convolution (one pass over the row as if it were one sequence, then p.fix: the tokens that start a segment,
    whose predecessors are their path's), q / k normalization (before the key heads are repeated), the output norm.
    Per level only the recurrence is left: gather q, k, v, g, beta, run the kernel from the parents' final states and
    put the outputs back. Before this layout the work around the kernel, repeated per level on padded segments, was 40 %
    of a request (L40S, eval2's requests, JOURNAL 2026-10-05)."""
    N, Hk, Hv = len(x), mod.num_k_heads, mod.num_v_heads
    x0, w = F.pad(x, (0, 0, 0, 1)), mod.conv1d.weight
    real = (torch.arange(N + 1, device=x.device) < N)[:, None]
    qkv = mod.in_proj_qkv(x0)
    conv = flat_conv(qkv.to(w.dtype), w)
    conv = conv.index_copy(0, p.fix, (qkv[p.pred].float() * w[:, 0].T.float()).sum(1).to(w.dtype))
    conv = F.silu(conv).to(x.dtype)
    qk = l2norm(conv[:, : 2 * mod.key_dim].reshape(N + 1, 2 * Hk, mod.head_k_dim))
    q, k = qk[:, :Hk].repeat_interleave(Hv // Hk, dim=1), qk[:, Hk:].repeat_interleave(Hv // Hk, dim=1)
    v = conv[:, 2 * mod.key_dim :].reshape(N + 1, Hv, mod.head_v_dim)
    beta = mod.in_proj_b(x0).sigmoid() * real
    g = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(x0).float() + mod.dt_bias) * real
    out, state = x.new_zeros(N + 1, Hv, mod.head_v_dim), None
    for idx, parent in p.levels:
        o, state = chunk_rule(q[idx], k[idx], v[idx], g=g[idx], beta=beta[idx], initial_state=None if parent is None else state[parent],
                              output_final_state=True, use_qk_l2norm_in_kernel=False)  # fmt: skip
        out.index_copy_(0, idx.reshape(-1), o.reshape(-1, Hv, mod.head_v_dim))  # padding lands on row N
    z = mod.in_proj_z(x)
    return mod.out_proj(gated_norm(mod.norm, out[:N], z.reshape(N, Hv, -1)).reshape(N, -1))


def flat_conv(x, w):
    """The depthwise causal convolution of Qwen3_5GatedDeltaNet over one sequence x [T, C] (w: [C, 1, K])."""
    return F.conv1d(x.T[None], w, groups=x.shape[1], padding=w.shape[2] - 1)[0, :, : len(x)].T


def attention(mod, x, cos, sin, a):
    """Qwen3_5Attention.forward over packed trees (a: pack's attn). Text tokens attend causally, in one call over the first
    a.root positions of every row (causal, so a shorter text never sees what follows it; what that call returns beyond a
    row's text is replaced or is padding). Branch tokens attend to all of their text and to their ancestors: a second
    call with the [Nb, T] mask. The [T, T] mask this replaces cost 14 % of eval2's requests on an L40S, most of it building
    the mask on the CPU (JOURNAL 2026-10-05)."""
    B, T, _ = x.shape
    q, gate = torch.chunk(mod.q_proj(x).view(B, T, -1, 2 * mod.head_dim), 2, dim=-1)
    q = rms_norm(mod.q_norm, q).transpose(1, 2)
    k = rms_norm(mod.k_norm, mod.k_proj(x).view(B, T, -1, mod.head_dim)).transpose(1, 2)
    v = mod.v_proj(x).view(B, T, -1, mod.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    r = a.root
    text = sdpa_attention_forward(mod, q[:, :, :r], k[:, :, :r], v[:, :, :r], None, scaling=mod.scaling, is_causal=True)[0]
    qb = q.gather(2, a.q[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[3]))
    branch = sdpa_attention_forward(mod, qb, k, v, a.mask, scaling=mod.scaling, is_causal=False)[0]
    out = torch.cat([text.reshape(B, r, -1), text.new_zeros(B, T - r, gate.shape[2] * gate.shape[3])], 1).reshape(B * T, -1)
    out = out.index_copy(0, a.flat, branch.reshape(-1, out.shape[1])[a.keep]).view(B, T, -1)
    return mod.o_proj(out * torch.sigmoid(gate.reshape(B, T, -1)))


def _layer(layer, h, cos, sin, p):
    x = rms_norm(layer.input_layernorm, h)
    if layer.block_type == "linear_attention":
        x = deltanet(layer.linear_attn, x, p)
    else:
        x = attention(layer.self_attn, x[p.rows], cos, sin, p.attn).flatten(0, 1)[p.back]
    h = h + x
    return h + layer.mlp(rms_norm(layer.post_attention_layernorm, h))


def score(sc, trees, grad_checkpoint=True):
    """Leaf scores (z_yes - z_no, Qwen35Scorer.readout) of a list of build_tree dicts, in leaf order.
    Per-layer activation checkpointing while training (HF's own drops the cache path this replaces)."""
    dec = sc.decoder()
    kernel = next(layer.linear_attn.conv_kernel_size for layer in dec.layers if layer.block_type == "linear_attention")
    p = pack(trees, kernel, sc.device)
    h, off = dec.embed_tokens(p.ids), 0
    for t in trees:
        for pixels, grid, start in t.get("images", ()):  # ponytail: one image per vision call; batch them if images dominate
            if isinstance(pixels, str):
                pixels = sc.image(pixels)[0]
            with torch.no_grad():  # frozen tower
                e = sc.visual(pixels.to(sc.device, sc.visual.dtype), grid_thw=grid.to(sc.device)).pooler_output
            h[off + start : off + start + len(e)] = e.to(h.dtype)
        off += len(t["ids"])
    cos, sin = dec.rotary_emb(h, p.pos)
    for layer in dec.layers:
        if grad_checkpoint and torch.is_grad_enabled():
            h = checkpoint(_layer, layer, h, cos, sin, p, use_reentrant=False)
        else:
            h = _layer(layer, h, cos, sin, p)
    h = rms_norm(dec.norm, h)
    return sc.readout(h[p.leaves])


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
        """Autotune fla's DeltaNet kernels for every batch-size bucket of pack's levels now, not on live requests: 42 s on an
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
                chunk_rule(q, q, v, g=g, beta=g.to(dt), initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=False)
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
