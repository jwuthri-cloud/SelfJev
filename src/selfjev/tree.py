"""Shared-prefix tree scorer: the state is read ONCE, and every question/candidate still reads it through ALL layers.

One token tree per distinct state:

    [instructions + state] ─┬─ [question 1] ─┬─ [" " + candidate A + suffix] → z_yes - z_no
                            │                └─ [" " + candidate B + suffix] → z_yes - z_no
                            └─ [question 2]  ── [" Yes" + suffix]            → z_yes - z_no   (binary)

A tree attention mask lets every token see its own segment (causally) and its ancestors only, and position ids
continue from the parent, so each leaf's score is exactly the score of the standalone causal sequence
"instructions + state + question + candidate": pairwise joint reading, as in the stock reranker, but with the
state first so that it can be shared. Branches never see each other, so extra questions cannot change answers.

Two exact execution paths (tests check they agree with the standalone sequences):
  packed  one pass over [root | branches] with a [T, T] tree mask            (training: short, with autograd)
  cached  the root alone with the plain causal kernel -> KV cache, then all branches against that cache in a
          second pass (inference: the state costs one pass however many questions/candidates there are)
Scores go through classify.decide as for every backend (sigmoid / grouped softmax / multilabel sigmoid).
Works with any Qwen3-architecture causal LM (Qwen3-Reranker-*, Qwen3-*-Instruct): only the chat suffix differs.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import sha256_file
from .formatting import BINARY_ANSWER, EVIDENCE, PREFIX, SUFFIX
from .model import MAX_CONTEXT, NO_ID, YES_ID, InputTooLong, default_device
from .schemas import Question

RERANKER_4B = ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5")
INSTRUCT_4B = ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")
FORMAT = "tree-v1"  # the compact variant (tree-compact-v2) was a dead end: tag archive/pre-cleanup-2026-09-27
# Template text around the user's text. The state comes FIRST so it can be shared; the question and proposed answer
# follow (the stock "answer-v1" mapping, which validation selected for the 4B reranker).
STATE_TEXT = ("<Document>: ", "\n")
QUESTION_TEXT = (f"<Instruct>: {EVIDENCE}\n<Query>: Question: ", "\nProposed answer:")
LEAF_TEXT = " "
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"  # instruct models: no <think> block


def suffix_for(model_id: str) -> str:
    return SUFFIX if "Reranker" in model_id else CHAT_SUFFIX


def format_config(model_id: str, format_name: str = FORMAT) -> dict:
    cfg = {"name": FORMAT, "prefix": PREFIX, "state": STATE_TEXT, "question": QUESTION_TEXT, "leaf": LEAF_TEXT,
           "suffix": suffix_for(model_id), "binary_answer": BINARY_ANSWER, "readout": ["yes", "no"],
           "user_text": "split_special_tokens=True"}
    if format_name != FORMAT:
        raise ValueError(f"Unknown tree format: {format_name}")
    return cfg | {"sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]}


def leaf_answers(q: Question) -> list[str]:
    return [BINARY_ANSWER] if q.type == "binary" else [c.description for c in q.candidates]


class Encoder:
    """Tokenizes template text normally (its <|im_start|> etc. are real control tokens) and user text with
    split_special_tokens, so a state containing "<|im_end|>" stays plain text. Pieces are tokenized separately and
    concatenated, which fixes every segment boundary."""

    def __init__(self, tokenizer, model_id, format_name=FORMAT):
        self.tok = tokenizer
        self.format = cfg = format_config(model_id, format_name)
        t = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
        self.prefix = t(cfg["prefix"])
        self.state_pre, self.state_post = t(cfg["state"][0]), t(cfg["state"][1])
        self.q_pre, self.q_post = t(cfg["question"][0]), t(cfg["question"][1])
        self.leaf_pre, self.suffix = t(cfg["leaf"]), t(cfg["suffix"])

    def user(self, texts):
        return self.tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"] if texts else []

    def root(self, state_ids):
        return self.prefix + self.state_pre + state_ids + self.state_post

    def question(self, q_ids):
        return self.q_pre + q_ids + self.q_post

    def leaf(self, answer_ids):
        return self.leaf_pre + answer_ids + self.suffix


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
    seg = torch.tensor(tree["seg"][tree["root"]:] if branches_only else tree["seg"])
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


class TreeModel:
    """A causal LM read out at leaf positions: s = z_yes - z_no, from the hidden state times the LM head rows."""

    def __init__(self, lm, pad_id):
        self.lm, self.pad_id = lm, pad_id

    def parts(self):
        base = self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm  # PeftModel -> LoRA'd LM
        return base.model, base.get_output_embeddings().weight

    def _additive(self, allowed, dtype):
        return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)

    def _readout(self, hidden, head):
        z = hidden.float() @ head[[YES_ID, NO_ID]].float().T
        return z[:, 0] - z[:, 1]

    def packed(self, trees):
        """One pass over each whole tree; batch rows are right-padded, pads attend only to themselves."""
        decoder, head = self.parts()
        dev, dtype = head.device, head.dtype
        B, T = len(trees), max(len(t["ids"]) for t in trees)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, T), dtype=torch.long)
        allowed = torch.eye(T, dtype=torch.bool).repeat(B, 1, 1)
        for b, t in enumerate(trees):
            n = len(t["ids"])
            ids[b, :n], pos[b, :n] = torch.tensor(t["ids"]), torch.tensor(t["pos"])
            allowed[b, :n, :n] = tree_mask(t)
        hidden = decoder(input_ids=ids.to(dev), position_ids=pos.to(dev),
                         attention_mask=self._additive(allowed[:, None].to(dev), dtype), use_cache=False).last_hidden_state
        rows = torch.tensor([b for b, t in enumerate(trees) for _ in t["leaves"]], device=dev)
        cols = torch.tensor([i for t in trees for i in t["leaves"]], device=dev)
        return self._readout(hidden[rows, cols], head)

    def cached(self, trees):
        """Roots in one right-padded causal pass (no mask needed: real tokens never see the padding after them),
        then every branch token in a second pass against that KV cache with the tree mask."""
        decoder, head = self.parts()
        dev, dtype = head.device, head.dtype
        B, R = len(trees), max(t["root"] for t in trees)
        roots = torch.full((B, R), self.pad_id, dtype=torch.long)
        for b, t in enumerate(trees):
            roots[b, :t["root"]] = torch.tensor(t["ids"][:t["root"]])
        with torch.profiler.record_function("tree.root_forward"):
            cache = decoder(input_ids=roots.to(dev), use_cache=True).past_key_values
        Tb = max(len(t["ids"]) - t["root"] for t in trees)
        ids = torch.full((B, Tb), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, Tb), dtype=torch.long)
        allowed = torch.zeros((B, Tb, R + Tb), dtype=torch.bool)
        allowed[:, torch.arange(Tb), R + torch.arange(Tb)] = True  # pad rows: never fully masked
        for b, t in enumerate(trees):
            r, n = t["root"], len(t["ids"]) - t["root"]
            ids[b, :n], pos[b, :n] = torch.tensor(t["ids"][r:]), torch.tensor(t["pos"][r:])
            with torch.profiler.record_function("tree.branch_mask"):
                allowed[b, :n, :r] = True  # real root visible; root padding stays masked
                allowed[b, :n, R:R + n] = tree_mask(t, branches_only=True)
        with torch.profiler.record_function("tree.branch_forward"):
            hidden = decoder(input_ids=ids.to(dev), position_ids=pos.to(dev), past_key_values=cache, use_cache=True,
                             attention_mask=self._additive(allowed[:, None].to(dev), dtype)).last_hidden_state
        rows = torch.tensor([b for b, t in enumerate(trees) for _ in t["leaves"]], device=dev)
        cols = torch.tensor([i - t["root"] for t in trees for i in t["leaves"]], device=dev)
        return self._readout(hidden[rows, cols], head)


class TreeScorer:
    """Scorer for classify / evaluate / calibrate / bench / serve (via score_requests)."""

    def __init__(self, model_id=RERANKER_4B[0], revision=RERANKER_4B[1], adapter=None, device=None, dtype="bfloat16",
                 max_length=MAX_CONTEXT, max_batch_tokens=32768, max_batch_size=64, lm=None, merge=False, format_name=None):
        if not 0 < max_length <= MAX_CONTEXT:
            raise ValueError(f"max_length must be in (0, {MAX_CONTEXT}]")
        self.device, self.max_length = device or default_device(), max_length
        self.max_batch_tokens, self.max_batch_size = max_batch_tokens, max_batch_size
        self.tokenizer = tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if (tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")) != (YES_ID, NO_ID):
            raise RuntimeError(f"{model_id}: yes/no token ids differ from {YES_ID}/{NO_ID}")
        saved_format = Path(adapter) / "tree_format.json" if adapter else None
        recorded = json.loads(saved_format.read_text()) if saved_format and saved_format.exists() else None
        if recorded and format_name and format_name != recorded["name"]:
            raise ValueError("Explicit tree format differs from the adapter's recorded training format")
        format_name = format_name or (recorded["name"] if recorded else FORMAT)
        self.enc = Encoder(tok, model_id, format_name)
        if recorded and recorded["sha"] != self.enc.format["sha"]:
            raise ValueError("Tree format hash differs from the adapter's recorded training format")
        if lm is None:
            lm = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
        if adapter:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, adapter)
            if merge:  # inference only: fewer kernels per layer; weights re-rounded to dtype
                lm = lm.merge_and_unload()
        self.model = TreeModel(lm.to(self.device).eval(), tok.pad_token_id)
        fmt = self.enc.format
        self.meta = {"model": model_id, "revision": revision, "architecture": f"shared-prefix tree ({format_name})",
                     "adapter": str(adapter) if adapter else None, "merged": bool(adapter and merge),
                     "adapter_sha256": sha256_file(Path(adapter) / "adapter_model.safetensors") if adapter else None,
                     "prompt": format_name, "prompt_sha": fmt["sha"], "device": self.device, "dtype": dtype, "max_length": max_length,
                     "truncation": "none (overlength input raises InputTooLong)"}

    def trees(self, reqs):
        """-> (trees, per-leaf (request, question id, candidate id)), one tree per distinct state, leaves in
        request / question / candidate order within each tree."""
        states, groups = {}, []
        for ri, r in enumerate(reqs):
            k = states.setdefault(r.state, len(states))
            if k == len(groups):
                groups.append([])
            groups[k] += [(ri, q) for q in r.questions]
        state_ids = self.enc.user(list(states))
        qs = [q for g in groups for _, q in g]
        q_ids = self.enc.user([q.instruction for q in qs])
        a_ids = iter(self.enc.user([a for q in qs for a in leaf_answers(q)]))
        trees, owners, qi = [], [], 0
        for s_ids, g in zip(state_ids, groups):
            branches, own = [], []
            for ri, q in g:
                branches.append((self.enc.question(q_ids[qi]), [self.enc.leaf(next(a_ids)) for _ in leaf_answers(q)]))
                own += [(ri, q.id, c.id) for c in q.candidates] or [(ri, q.id, None)]
                qi += 1
            trees.append(build_tree(self.enc.root(s_ids), branches))
            owners.append(own)
        return trees, owners

    def check_lengths(self, trees, owners):
        """Every leaf's standalone sequence (root + question + leaf) must fit max_length; nothing is truncated."""
        over, where = [], []
        for t, own in zip(trees, owners):
            for n, (leaf, o) in enumerate(zip(t["leaves"], own)):
                length = t["pos"][leaf] + 1
                if length > self.max_length:
                    over.append((len(over), length))
                    where.append(f"request {o[0]} question '{o[1]}'" + (f" candidate '{o[2]}'" if o[2] else "") + f": {length} tokens")
        if over:
            raise InputTooLong(over, self.max_length, "; ".join(where[:5]))

    def batches(self, trees):
        """Trees sorted by root length; a batch's padded tokens (rows x (longest root + longest branch part)) stay
        under max_batch_tokens unless one tree alone exceeds it."""
        order = sorted(range(len(trees)), key=lambda i: trees[i]["root"])
        batch, R, Tb = [], 0, 0
        for i in order:
            r, tb = trees[i]["root"], len(trees[i]["ids"]) - trees[i]["root"]
            if batch and ((len(batch) + 1) * (max(R, r) + max(Tb, tb)) > self.max_batch_tokens or len(batch) == self.max_batch_size):
                yield batch
                batch, R, Tb = [], 0, 0
            batch, R, Tb = batch + [i], max(R, r), max(Tb, tb)
        if batch:
            yield batch

    @torch.inference_mode()
    def score_requests(self, reqs):
        t0 = time.perf_counter()
        trees, owners = self.trees(reqs)
        self.check_lengths(trees, owners)
        t1 = time.perf_counter()
        leaf_scores, padded, n_batches = [None] * len(trees), 0, 0
        for b in self.batches(trees):
            s = self.model.cached([trees[i] for i in b]).tolist()  # .tolist() waits for the device
            k = 0
            for i in b:
                leaf_scores[i], k = s[k:k + len(trees[i]["leaves"])], k + len(trees[i]["leaves"])
            padded += len(b) * (max(trees[i]["root"] for i in b) + max(len(trees[i]["ids"]) - trees[i]["root"] for i in b))
            n_batches += 1
        t2 = time.perf_counter()
        by_q = {}
        for own, s in zip(owners, leaf_scores):
            for (ri, qid, _), v in zip(own, s):
                by_q.setdefault((ri, qid), []).append(v)
        out = [[by_q[ri, q.id] for q in r.questions] for ri, r in enumerate(reqs)]
        state_tokens = sum(t["root"] for t in trees)
        branch_tokens = sum(len(t["ids"]) for t in trees) - state_tokens
        stats = {"pairs": sum(len(t["leaves"]) for t in trees), "batches": n_batches, "input_tokens": state_tokens + branch_tokens,
                 "padded_tokens": padded, "max_pair_tokens": max(t["pos"][l] + 1 for t in trees for l in t["leaves"]),
                 "state_sequences": len(trees), "state_tokens": state_tokens, "candidate_tokens": branch_tokens,
                 "tokenize_ms": 1e3 * (t1 - t0), "model_ms": 1e3 * (t2 - t1)}
        return out, stats
