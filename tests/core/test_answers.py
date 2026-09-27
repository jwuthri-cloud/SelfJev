"""Scores -> typed answers: probabilities, decisions, thresholds, abstention, and mapping back to candidates."""

import json
import math
from pathlib import Path

import pytest

from selfjev.core.answers import classify, decide
from selfjev.core.schemas import parse_question, parse_request
from tests.fakes import FakeScorer, pair_text

REQ = json.loads(Path("examples/request.json").read_text())


def test_binary_complement_and_threshold_sources():
    q = parse_question({"id": "b", "type": "binary", "instruction": "x"})
    for s in (-40.0, -2.3, 0.0, 1.7, 35.0):
        out = decide(q, [s])
        assert math.isclose(out["p_yes"] + out["p_no"], 1.0, abs_tol=1e-12)
        assert out["selected"] == (out["p_yes"] >= 0.5) and out["threshold_source"] == "default"
    assert decide(q, [0.0], {"threshold": {"binary": 0.7}})["threshold_source"] == "validation"
    assert (
        decide(parse_question({"id": "b", "type": "binary", "instruction": "x", "threshold": 0.2}), [0.0])["threshold_source"] == "request"
    )


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


def test_scores_map_back_to_their_candidates():
    res = classify(FakeScorer(), REQ)
    req = parse_request(REQ)
    for q, out in zip(req.questions, res["questions"]):
        expected = [FakeScorer.f(pair_text(req.state, q, c)) for c in q.candidates] or [FakeScorer.f(pair_text(req.state, q))]
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
                {c["id"]: c["probability"] for c in base[qid]["candidates"]}
            )
            assert sorted(other[qid]["selected"] if isinstance(other[qid]["selected"], list) else [other[qid]["selected"]]) == sorted(
                base[qid]["selected"] if isinstance(base[qid]["selected"], list) else [base[qid]["selected"]]
            )
