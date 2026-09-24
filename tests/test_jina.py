"""jina backend checks on the real jinaai/jina-reranker-v3.5 (downloads ~1.2 GB once)."""
import torch

from personal_jev.jina import JinaScorer
from personal_jev.schemas import parse_request


def test_padding_invariance_and_lengths():
    sc = JinaScorer(dtype="float32", max_batch_tokens=8192)
    short = parse_request({"state": "Order 4471 arrived two days late and one mug was chipped.", "questions": [
        {"id": "late", "type": "binary", "instruction": "Did the delivery arrive late?"},
        {"id": "issue", "type": "multiclass", "instruction": "What is the main problem?", "candidates": [
            {"id": "damage", "description": "an item was damaged"}, {"id": "billing", "description": "a billing error"}]}]})
    long = parse_request({"state": " ".join(f"Line {i}: routine status update with no incident." for i in range(120)) + " Then the site went down at 14:02.",
                          "questions": [{"id": "down", "type": "binary", "instruction": "Was there an outage?"}]})
    ctxs, _ = sc.contexts([short, long])
    assert len(ctxs) == 2 and len(ctxs[0]["doc_pos"]) == 3 and len(ctxs[1]["doc_pos"]) == 1
    n = len(ctxs[0]["doc_pos"])
    state_ids = sc.enc.user([short.state])[0]
    # overhead(): template tokens = context length minus state, tail and passage texts
    ids = ctxs[0]["ids"]
    expected = len(state_ids) + min(len(state_ids), sc.enc.query_tail) + sc.enc.overhead(n)
    texts = sum(len(sc.enc.passage(qi, ai)) for qi, ai in zip(
        sc.enc.user([short.questions[0].instruction, short.questions[1].instruction, short.questions[1].instruction]),
        sc.enc.user(["Yes", "an item was damaged", "a billing error"])))
    assert len(ids) == expected + texts
    alone = sc.model.logits([ctxs[0]]).tolist()
    batched = sc.model.logits([ctxs[0], ctxs[1]]).tolist()[:n]
    assert max(abs(a - b) for a, b in zip(alone, batched)) < 1e-3, (alone, batched)
    (res,), _ = sc.score_requests([short])
    assert len(res) == 2 and len(res[0]) == 1 and len(res[1]) == 2
