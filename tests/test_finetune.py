"""RLCD's objective is proper: a policy trained on it reports calibrated probabilities. The metrics helper agrees with hand values."""
import torch

from personal_jev.finetune import metrics, rlcd_loss


def test_rlcd_reports_the_base_rate():
    torch.manual_seed(0)
    for typ, targets, want in (("binary", [i < 70 for i in range(100)], 0.7),  # 70% yes -> p(yes) = 0.7
                               ("multiclass", [0] * 60 + [1] * 30 + [2] * 10, 0.6)):  # p(option 0) = 0.6
        n = 1 if typ == "binary" else 3
        theta = torch.zeros(n, requires_grad=True)  # one shared question: its best report is the base rate
        items = [{"type": typ, "ids": list(range(n)), "target": t, "ref": [0.0] * n} for t in targets]
        opt = torch.optim.Adam([theta], lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            rlcd_loss(theta.repeat(len(items)), items, {"log": 1, "brier": 1, "spherical": 1}, samples=8, sigma=0.2, beta=0.0).div(len(items)).backward()
            opt.step()
        p = (theta.sigmoid()[0] if typ == "binary" else theta.softmax(0)[0]).item()
        assert abs(float(p) - want) < 0.05, (typ, float(p))


def test_soft_targets_are_learned():
    """Label mixed 50/50 with a teacher: binary yes + teacher 0.4 -> 0.7; choice 0 + teacher (0.2, 0.8) -> 0.6. Both the
    fine-tune loss and RLCD's reward should land there on a single question."""
    from personal_jev.finetune import log_loss, soft_target
    torch.manual_seed(0)
    for typ, target, soft, want in (("binary", True, 0.4, 0.7), ("multiclass", 0, {"a": 0.2, "b": 0.8}, 0.6)):
        n = 1 if typ == "binary" else 2
        it = {"type": typ, "ids": list(range(n)), "target": target, "candidate_ids": [] if n == 1 else ["a", "b"], "ref": [0.0] * n}
        it["y"] = soft_target(it, soft, 0.5)
        for name, loss in (("finetune", lambda t: log_loss(t, [it])),
                           ("rlcd", lambda t: rlcd_loss(t, [it], {"log": 1, "brier": 1, "spherical": 1}, samples=64, sigma=0.2, beta=0.0))):
            theta = torch.zeros(n, requires_grad=True)
            opt = torch.optim.Adam([theta], lr=0.05)
            for _ in range(400):
                opt.zero_grad()
                loss(theta).backward()
                opt.step()
            p = (theta.sigmoid()[0] if typ == "binary" else theta.softmax(0)[0]).item()
            assert abs(p - want) < 0.05, (typ, name, p)


def test_confident_miss_cost_hedges():
    """Target 95% yes: the log score alone reports 0.95; with a confident-miss cost of 5 (5 x 5% expected loss above 0.9)
    the best report drops just under 0.9."""
    torch.manual_seed(0)
    for weights, lo, hi in (({"log": 1}, 0.93, 0.97), ({"log": 1, "confident_miss": 5}, 0.8, 0.9)):
        it = {"type": "binary", "ids": [0], "target": True, "y": [[0.95, 0.05]], "ref": [0.0]}
        theta = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=0.05)
        for _ in range(600):
            opt.zero_grad()
            rlcd_loss(theta, [it], weights, samples=64, sigma=0.2, beta=0.0).backward()
            opt.step()
        assert lo < theta.sigmoid().item() < hi, (weights, theta.sigmoid().item())


def test_metrics_by_hand():
    items = [{"type": "binary", "ids": [0], "target": True}, {"type": "multiclass", "ids": [0, 1], "target": 1}]
    m = metrics(items, [[2.0], [3.0, 0.0]])  # a right yes at p 0.88, a wrong confident choice at p 0.95
    assert m["accuracy"] == 0.5 and m["confidently_wrong"] == 1 and m["n"] == 2
    brier_yes = 2 * (1 - torch.sigmoid(torch.tensor(2.0))) ** 2  # (p - 1)^2 + ((1 - p) - 0)^2
    brier_choice = 2 * torch.softmax(torch.tensor([3.0, 0.0]), 0)[0] ** 2  # p0^2 + (p1 - 1)^2, p1 = 1 - p0
    assert abs(m["brier"] - float(brier_yes + brier_choice) / 2) < 1e-5


def test_finetune_then_rlcd_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    """The whole loop (encode, LoRA, tree batches, loss, validation, checkpoints) on a tiny random Qwen3.5 (CPU)."""
    import json
    import random

    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from personal_jev import challengers, finetune

    class Tiny:  # the parts of ChallengerScorer that finetune.train uses, with a character-level "tokenizer"
        pad, device, max_length, meta = 0, "cpu", 4096, {"model": "tiny", "revision": "none", "prompt": "tiny"}

        def __init__(self, base):
            torch.manual_seed(0)
            self.model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
                vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                linear_num_value_heads=4))

        def entry(self, state, q):
            tok = lambda t: [3 + ord(c) % 290 for c in t]  # noqa: E731
            answers = ["Yes"] if q.type == "binary" else [c.description for c in q.candidates]
            e = {"root": [1] + tok(state), "branches": [tok(q.instruction[:24]) + [2] + tok(a)[:6] for a in answers], "n": len(answers)}
            e["length"] = len(e["root"]) + max(map(len, e["branches"]))
            return e

        def base(self):
            return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

        def decoder(self):
            return self.base().model

        def readout(self, h):
            z = h.float() @ self.base().lm_head.weight[[1, 2]].float().T
            return z[..., 0] - z[..., 1]

    monkeypatch.setattr(challengers, "ChallengerScorer", Tiny)
    rnd, rows = random.Random(0), []
    for i in range(40):
        state = f"ticket {i}: " + " ".join(rnd.choice(["refund", "late", "broken", "thanks"]) for _ in range(6))
        c = [{"id": x, "description": f"team {x}"} for x in "abc"]
        rows.append([{"state": state, "question": {"type": "binary", "instruction": "Is it urgent?"}, "target": i % 2 == 0},
                     {"state": state, "question": {"type": "multiclass", "instruction": "Which team?", "candidates": c}, "target": "abc"[i % 3]},
                     {"state": state, "question": {"type": "multilabel", "instruction": "Which apply?", "candidates": c}, "target": ["a"] if i % 2 else []}][i % 3])
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ft = finetune.train("finetune", str(data), tmp_path / "ft", batch_tokens=512, grad_accum=2, eval_every=5, lora_r=4)
    assert ft["train_questions"] == 38 and ft["val_questions"] == 2 and (tmp_path / "ft/adapter/adapter_model.safetensors").exists()
    rl = finetune.train("rlcd", str(data), tmp_path / "rl", init=str(tmp_path / "ft/adapter"), batch_tokens=512, grad_accum=2, eval_every=5)
    assert rl["select_by"] == "validation brier" and {"ece", "brier", "confidently_wrong"} <= set(rl["best"])
    assert json.loads((tmp_path / "rl/train_meta.json").read_text())["rlcd"]["reward"] == {"log": 1.0, "brier": 1.0, "spherical": 1.0}
    for r in rows:  # a teacher's probabilities on every row (scripts/jev_soft_targets.py's format)
        r["soft"] = 0.6 if r["question"]["type"] == "binary" else {"a": 0.5, "b": 0.3, "c": 0.2}
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for mode in ("finetune", "rlcd"):
        m = finetune.train(mode, str(data), tmp_path / f"soft_{mode}", init=str(tmp_path / "ft/adapter"), batch_tokens=512, grad_accum=2, eval_every=5)
        assert m["soft_targets"] == 38 and m["best"]["n"] == 2
