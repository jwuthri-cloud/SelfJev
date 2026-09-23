"""Real-model tests with the pinned Qwen3-Reranker-0.6B (fp32). Slower: they load the checkpoint."""
import json
import random

import pytest
import torch
from peft import LoraConfig, get_peft_model

from personal_jev.classify import classify
from personal_jev.data import load
from personal_jev.formatting import PREFIX, SUFFIX, pair_text
from personal_jev.model import MODEL_ID, MODEL_REVISION, InputTooLong, Scorer
from personal_jev.train import encode_items, grouped_loss, train

TOL = 1e-3  # fp32 logits; batching/padding may move scores by float noise only


@pytest.fixture(scope="module")
def scorer():
    return Scorer(dtype="float32")


def official_scores(model, tokenizer, pairs, max_length=8192):
    """Verbatim logic of the model card's Transformers reference (process_inputs + compute_logits)."""
    tokenizer = tokenizer.__class__.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side="left")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    token_false_id, token_true_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    inputs = tokenizer(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                       max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        batch_scores = model(**inputs).logits[:, -1, :]
    batch_scores = torch.stack([batch_scores[:, token_false_id], batch_scores[:, token_true_id]], dim=1)
    return torch.nn.functional.log_softmax(batch_scores.float(), dim=1)[:, 1].exp().tolist()


PAIRS = [
    pair_text("Given a web search query, retrieve relevant passages that answer the query", "What is the capital of China?",
              "The capital of China is Beijing."),
    pair_text("Given a web search query, retrieve relevant passages that answer the query", "What is the capital of China?",
              "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
              "responsible for the movement of planets around the sun. " * 20),
    pair_text("Judge whether the Document is a complaint.", "complaint", "My order arrived broken and nobody answers my emails."),
]


def test_matches_official_reference(scorer):
    ours, _ = scorer.score(PAIRS)
    ref = official_scores(scorer.model, scorer.tokenizer, PAIRS)
    p = torch.sigmoid(torch.tensor(ours))
    assert torch.allclose(p, torch.tensor(ref), atol=1e-4), (p.tolist(), ref)
    assert ours[0] > 5 and ours[1] < -5  # relevant vs irrelevant, as on the model card


def test_batched_equals_single_across_lengths(scorer):
    texts = PAIRS + [pair_text("Judge.", "q", "word " * n) for n in (3, 400, 1500)]
    batched, stats = scorer.score(texts)
    single = [scorer.score([t])[0][0] for t in texts]
    assert stats["padded_tokens"] > stats["input_tokens"]  # padding really happened
    assert max(abs(a - b) for a, b in zip(batched, single)) < TOL


def test_candidate_order_and_extra_questions_real_model(scorer):
    req = json.load(open("examples/request.json"))
    base = {q["id"]: q for q in classify(scorer, req)["questions"]}
    req["questions"][1]["candidates"].reverse()
    req["questions"].append({"id": "noise", "type": "binary", "instruction": "Is the moon made of cheese?"})
    other = {q["id"]: q for q in classify(scorer, req)["questions"]}
    assert other["team"]["selected"] == base["team"]["selected"]
    for qid in ("team", "tags"):
        a = {c["id"]: c["score"] for c in base[qid]["candidates"]}
        b = {c["id"]: c["score"] for c in other[qid]["candidates"]}
        assert max(abs(a[k] - b[k]) for k in a) < TOL
    assert abs(base["urgent"]["score"] - other["urgent"]["score"]) < TOL


def test_overlength_is_an_error_not_truncation():
    short = Scorer(max_length=128)
    req = {"state": "Checkout is broken. " * 100, "questions": [{"id": "u", "type": "binary", "instruction": "Urgent?"}]}
    with pytest.raises(InputTooLong, match="question 'u'"):
        classify(short, req)
    with pytest.raises(ValueError):
        Scorer(max_length=40000)


def test_lora_gradients_smoke_training_and_reload(tmp_path):
    torch.manual_seed(0)
    sc = Scorer(dtype="float32", max_length=1024)
    sc.model = get_peft_model(sc.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
    items, dropped = encode_items(sc, load("data/dev.jsonl"))
    assert items and not dropped
    items = items[:8]  # small fixed batch keeps memory modest on laptops
    params = [p for p in sc.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4)
    ids = [x for it in items for x in it["ids"]]
    losses = []
    for _ in range(6):
        sc.model.train()
        loss = grouped_loss(sc.forward(ids), items) / len(items)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]  # fits the tiny dev set; NOT evidence of generalisation
    sc.model.eval()
    before, _ = sc.score_ids(ids)
    sc.model.save_pretrained(tmp_path / "adapter")
    reloaded = Scorer(adapter=tmp_path / "adapter", dtype="float32", max_length=1024)
    after, _ = reloaded.score_ids(ids)
    assert max(abs(a - b) for a, b in zip(before, after)) < TOL
    assert reloaded.meta["adapter_sha256"]


def test_train_entrypoint_runs_end_to_end(tmp_path):
    rows = [json.loads(line) for line in open("data/dev.jsonl")]
    rng = random.Random(0)
    for r in rows:
        r["split"] = "train" if rng.random() < 0.6 else "validation"
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    meta = train(None, train_files=[str(data)], val_files=[str(data)], out_dir=str(tmp_path / "run"), max_steps=3,
                 eval_every=2, max_batch_tokens=2048, grad_accum=1, max_length=1024, gradient_checkpointing=True)
    assert meta["steps"] == 3 and meta["reload_check"]["max_abs_score_diff"] < TOL
    assert meta["lora"]["trainable_params"] > 0 and (tmp_path / "run/adapter/adapter_model.safetensors").exists()
