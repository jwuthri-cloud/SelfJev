"""Pure-logic tests (no model): schemas, output modes, grouping, losses, metrics, calibration safeguards.
A fake scorer is used ONLY to test grouping/mapping; it says nothing about model quality."""
import itertools
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch

from personal_jev import calibration
from personal_jev.classify import classify, classify_many, decide
from personal_jev.data import expand_source, split_for, validate_example
from personal_jev.evaluate import auroc, macro_f1, reliability
from personal_jev.formatting import question_pairs
from personal_jev.model import InputTooLong, Scorer
from personal_jev.schemas import ValidationError, parse_question, parse_request
from personal_jev.train import grouped_loss, micro_batches, shuffle_candidates

REQ = json.load(open("examples/request.json"))


class FakeScorer:
    """Deterministic pseudo-score per pair text; lets tests check that scores land on the right candidate."""
    meta = {"model": "fake", "revision": "x", "adapter": None, "adapter_sha256": None}

    @staticmethod
    def f(text):
        return (sum(map(ord, text)) % 997) / 50 - 10

    def score(self, texts):
        return [self.f(t) for t in texts], {"pairs": len(texts)}


# --- schemas -------------------------------------------------------------------------------------------------------

def test_valid_request_parses():
    r = parse_request(REQ)
    assert [q.type for q in r.questions] == ["binary", "multiclass", "multilabel"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r.update(state="  "), "state"),
    (lambda r: r.update(questions=[]), "questions"),
    (lambda r: r["questions"].append(dict(r["questions"][0])), "duplicate question"),
    (lambda r: r["questions"][1]["candidates"].append({"id": "billing", "description": "x"}), "duplicate candidate"),
    (lambda r: r["questions"][1].update(candidates=[]), "non-empty list"),
    (lambda r: r["questions"][1].update(candidates=r["questions"][1]["candidates"][:1]), "at least 2"),
    (lambda r: r["questions"][0].update(threshold="0.5"), "threshold"),
    (lambda r: r["questions"][0].update(threshold=True), "threshold"),
    (lambda r: r["questions"][0].update(threshold=1.5), "threshold"),
    (lambda r: r["questions"][0].update(threshold=float("nan")), "threshold"),
    (lambda r: r["questions"][0].update(candidates=[{"id": "a", "description": "b"}]), "no candidates"),
    (lambda r: r["questions"][1].update(threshold=0.3), "abstain_below"),
    (lambda r: r["questions"][2]["candidates"][0].update(description=""), "description"),
    (lambda r: r["questions"][0].update(treshold=0.5), "unknown field"),
    (lambda r: r["questions"][0].update(type="ordinal"), "type"),
])
def test_invalid_requests(mutate, msg):
    r = json.loads(json.dumps(REQ))
    mutate(r)
    with pytest.raises(ValidationError, match=msg):
        parse_request(r)


# --- output modes ----------------------------------------------------------------------------------------------------

def test_binary_complement_and_threshold_sources():
    q = parse_question({"id": "b", "type": "binary", "instruction": "x"})
    for s in (-40.0, -2.3, 0.0, 1.7, 35.0):
        out = decide(q, [s])
        assert math.isclose(out["p_yes"] + out["p_no"], 1.0, abs_tol=1e-12)
        assert out["selected"] == (out["p_yes"] >= 0.5) and out["threshold_source"] == "default"
    assert decide(q, [0.0], {"threshold": {"binary": 0.7}})["threshold_source"] == "validation"
    assert decide(parse_question({"id": "b", "type": "binary", "instruction": "x", "threshold": 0.2}), [0.0])["threshold_source"] == "request"


def test_multiclass_normalises_within_question_and_multilabel_does_not():
    mc = parse_question(REQ["questions"][1])
    out = decide(mc, [3.0, 3.0, -1.0])
    assert math.isclose(sum(c["probability"] for c in out["candidates"]), 1.0, abs_tol=1e-12)
    ml = parse_question(REQ["questions"][2])
    out = decide(ml, [3.0, 3.0, 3.0])
    assert sum(c["probability"] for c in out["candidates"]) > 2.5  # independent marginals, no sum-to-one
    assert out["selected"] == ["revenue_loss", "refund", "blocked"]


def test_abstention_and_temperature():
    q = parse_question(REQ["questions"][1] | {"abstain_below": 0.9})
    out = decide(q, [1.0, 0.8, 0.5])
    assert out["selected"] is None and out["abstained"]
    hot = decide(parse_question(REQ["questions"][1]), [1.0, 0.0, 0.0], {"temperature": {"multiclass": 2.0}})
    cold = decide(parse_question(REQ["questions"][1]), [1.0, 0.0, 0.0])
    assert hot["calibration"] == "heldout_temperature_scaled" and cold["calibration"] == "uncalibrated"
    assert hot["candidates"][0]["probability"] < cold["candidates"][0]["probability"]


# --- grouping / mapping ----------------------------------------------------------------------------------------------

def test_scores_map_back_to_their_candidates():
    res = classify(FakeScorer(), REQ)
    req = parse_request(REQ)
    for q, out in zip(req.questions, res["questions"]):
        expected = [FakeScorer.f(t) for t in question_pairs(q, req.state)]
        got = [out["score"]] if q.type == "binary" else [c["score"] for c in out["candidates"]]
        assert got == expected


def test_permutations_and_extra_questions_do_not_change_answers():
    base = {q["id"]: q for q in classify(FakeScorer(), REQ)["questions"]}
    r = json.loads(json.dumps(REQ))
    r["questions"][1]["candidates"].reverse()
    r["questions"].insert(0, {"id": "irrelevant", "type": "binary", "instruction": "Is the sky green?"})
    other = {q["id"]: q for q in classify(FakeScorer(), r)["questions"]}
    for qid in base:
        if base[qid]["type"] == "binary":
            assert other[qid]["p_yes"] == base[qid]["p_yes"]
        else:
            assert {c["id"]: c["probability"] for c in other[qid]["candidates"]} == pytest.approx(
                {c["id"]: c["probability"] for c in base[qid]["candidates"]})
            assert sorted(other[qid]["selected"] if isinstance(other[qid]["selected"], list) else [other[qid]["selected"]]) == \
                sorted(base[qid]["selected"] if isinstance(base[qid]["selected"], list) else [base[qid]["selected"]])


def test_length_sorted_batches_cover_every_pair_once_under_budget():
    fake = SimpleNamespace(max_batch_tokens=100, max_batch_size=4)
    lengths = [random.Random(i).randint(5, 60) for i in range(50)] + [250]  # one pair alone exceeds the budget
    batches = list(Scorer.batches(fake, lengths))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 4 and (len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 100)


def test_overlength_error_names_the_offender():
    class Short(FakeScorer):
        def score(self, texts):
            raise InputTooLong([(len(texts) - 1, 999)], 128)
    with pytest.raises(InputTooLong, match="question 'tags' candidate 'blocked': 999 tokens"):
        classify(Short(), REQ)


# --- training losses -------------------------------------------------------------------------------------------------

ITEMS = [{"type": "multiclass", "ids": [[1]] * 3, "target": 2}, {"type": "binary", "ids": [[1]], "target": True},
         {"type": "multilabel", "ids": [[1]] * 4, "target": [True, False, False, True]}, {"type": "multiclass", "ids": [[1]] * 2, "target": 0}]


def test_grouped_loss_matches_per_question_definitions():
    s = torch.tensor([0.5, -1.0, 2.0, 0.3, 1.0, -2.0, 0.0, 3.0, -0.5, 0.7])
    F = torch.nn.functional
    want = (F.cross_entropy(s[None, 0:3], torch.tensor([2])) + F.binary_cross_entropy_with_logits(s[3:4], torch.tensor([1.0]))
            + F.binary_cross_entropy_with_logits(s[4:8], torch.tensor([1.0, 0, 0, 1])) + F.cross_entropy(s[None, 8:10], torch.tensor([0])))
    assert torch.allclose(grouped_loss(s, ITEMS), want)


def test_no_normalisation_across_questions():
    """Changing another question's scores must not change a multiclass question's loss (no cross-question softmax)."""
    s1 = torch.tensor([0.5, -1.0, 2.0, 0.3, 1.0, -2.0, 0.0, 3.0, -0.5, 0.7])
    s2 = s1.clone()
    s2[3:10] += 50.0
    one = lambda s: grouped_loss(s[0:3], ITEMS[:1])
    assert one(s1) == one(s2)
    with pytest.raises(AssertionError):
        grouped_loss(s1[:9], ITEMS)  # misaligned scores are rejected, never silently padded


def test_shuffle_candidates_remaps_targets():
    rng = random.Random(0)
    item = {"type": "multilabel", "ids": [[1], [2], [3], [4]], "target": [True, False, False, True]}
    mc = {"type": "multiclass", "ids": [[1], [2], [3]], "target": 1}
    for _ in range(20):
        a, b = shuffle_candidates(item, rng), shuffle_candidates(mc, rng)
        assert {i[0] for i, t in zip(a["ids"], a["target"]) if t} == {1, 4}
        assert b["ids"][b["target"]] == [2]


def test_micro_batches_keep_questions_whole():
    items = [{"ids": [[0] * random.Random(i).randint(10, 200)] * (1 + i % 5)} for i in range(200)]
    batches = micro_batches(items, 1000, random.Random(1))
    assert sorted(i for b in batches for i in b) == list(range(200))


# --- metrics ---------------------------------------------------------------------------------------------------------

def test_auroc_matches_brute_force_with_ties():
    rng = random.Random(3)
    ys = [rng.random() < 0.4 for _ in range(300)]
    ps = [round(rng.random(), 1) for _ in range(300)]
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    brute = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))
    assert auroc(ys, ps) == pytest.approx(brute)


def test_reliability_and_macro_f1():
    ece, rows = reliability([0.95, 0.95, 0.05, 0.05], [True, False, False, False])
    assert ece == pytest.approx(0.5 * 0.45 + 0.5 * 0.05)
    assert macro_f1([("a", "a"), ("b", "a"), ("b", "b")]) == pytest.approx((2 / 3 + 2 / 3) / 2)


# --- calibration -----------------------------------------------------------------------------------------------------

def _report(tmp_path, split, preds, name="r.json", **meta):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"model": "m", "revision": "r", "adapter": None, "adapter_sha256": None, "prompt_sha": "p",
                                      "data": [], "splits": [split], "calibration": None} | meta, "predictions": preds}))
    return p


def test_temperature_recovers_known_scale(tmp_path):
    rng = random.Random(0)
    preds = []
    for _ in range(3000):
        z = rng.gauss(0, 2)
        y = rng.random() < 1 / (1 + math.exp(-z))
        preds.append({"split": "calibration", "type": "binary", "scores": [3.0 * z], "target": y})  # overconfident by 3x
    cal = calibration.fit(_report(tmp_path, "calibration", preds))
    assert cal["temperature"]["binary"] == pytest.approx(3.0, rel=0.1)
    assert "multiclass" in cal["fit"]["skipped"]


def test_calibration_refuses_test_labels(tmp_path):
    preds = [{"split": "test", "type": "binary", "scores": [1.0], "target": True}] * 50
    with pytest.raises(calibration.LeakageError):
        calibration.fit(_report(tmp_path, "test", preds))
    mixed = _report(tmp_path, "calibration", [p | {"split": "calibration"} for p in preds] + preds[:1], "mixed.json")
    with pytest.raises(calibration.LeakageError):
        calibration.fit(mixed)
    ok = _report(tmp_path, "calibration", [p | {"split": "calibration"} for p in preds], "ok.json")
    with pytest.raises(calibration.LeakageError):
        calibration.fit(ok, threshold_report=_report(tmp_path, "test", preds, "t.json"))


def test_calibration_file_is_bound_to_its_model(tmp_path):
    preds = [{"split": "calibration", "type": "binary", "scores": [float(i % 7 - 3)], "target": i % 2 == 0} for i in range(60)]
    out = tmp_path / "cal.json"
    calibration.fit(_report(tmp_path, "calibration", preds), out=out)
    good = {"model": "m", "revision": "r", "adapter": None, "adapter_sha256": None}
    assert calibration.load(out, good, "p")["fit"]["file"] == str(out)
    with pytest.raises(ValueError):
        calibration.load(out, good | {"adapter_sha256": "abc"}, "p")
    with pytest.raises(ValueError):
        calibration.load(out, good, "other-prompt")


def test_best_threshold_maximises_f1():
    items = [(2.0, True), (1.0, True), (0.5, False), (-1.0, True), (-2.0, False)]
    t = calibration.best_threshold(items, 1.0)
    p = lambda s: 1 / (1 + math.exp(-s))
    assert p(-2.0) < t < p(-1.0)  # predicting the top 4 positive gives F1 = 6/7, the maximum


# --- data ------------------------------------------------------------------------------------------------------------

def test_example_validation_and_source_expansion():
    src = {"source_id": "s1", "family": "f", "provenance": "test", "state": "x", "questions": [
        {"type": "binary", "instruction": "q?", "target": False, "hard_cases": ["negation"]},
        {"type": "multilabel", "instruction": "q?", "candidates": [{"id": "a", "description": "A"}], "target": []}]}
    exs = expand_source(src)
    assert [e["id"] for e in exs] == ["s1-q0", "s1-q1"] and exs[0]["hard_cases"] == ["negation"]
    with pytest.raises(ValidationError):
        validate_example(exs[1] | {"target": ["zzz"]})
    with pytest.raises(ValidationError):
        validate_example(exs[0] | {"target": "yes"})


def test_split_for_is_deterministic_and_roughly_proportional():
    w = {"validation": 0.3, "calibration": 0.2, "test": 0.5}
    assert split_for("abc", w) == split_for("abc", w)
    counts = {k: 0 for k in w}
    for i in range(5000):
        counts[split_for(f"s{i}", w)] += 1
    assert all(abs(counts[k] / 5000 - w[k]) < 0.03 for k in w)
