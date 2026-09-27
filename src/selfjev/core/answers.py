"""Scores -> typed answers. A scorer implements score_requests(requests) -> (per request, per question, the
candidate scores s = z_yes - z_no; stats): selfjev.engine's TreeServer and VllmScorer do.

binary:     p_yes = sigmoid(s / T); selected = p_yes >= threshold
multiclass: p = softmax(scores / T) within the question; selected = argmax (or None when abstaining)
multilabel: p = sigmoid(scores / T) elementwise, no sum constraint; selected = [p >= threshold]
T = 1 unless a calibration file supplies a held-out temperature for that question type.
"""

import math
import time

from .schemas import Question, Request, parse_request


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def _threshold(q: Question, calibration):
    if q.threshold is not None:
        return q.threshold, "request"
    t = (calibration or {}).get("threshold", {}).get(q.type)
    return (t, "validation") if t is not None else (0.5, "default")


def decide(q: Question, scores: list[float], calibration=None) -> dict:
    T = (calibration or {}).get("temperature", {}).get(q.type)
    out = {"id": q.id, "type": q.type, "calibration": "heldout_temperature_scaled" if T else "uncalibrated"}
    if T:
        out["temperature"] = T
    T = T or 1.0
    if q.type == "multiclass":
        probs = softmax([s / T for s in scores])
        best = max(range(len(probs)), key=lambda i: (probs[i], q.candidates[i].id))  # exact ties: by id, not input order
        abstain = q.abstain_below is not None and probs[best] < q.abstain_below
        return out | {
            "selected": None if abstain else q.candidates[best].id,
            "abstained": abstain,
            "abstain_below": q.abstain_below,
            "candidates": [{"id": c.id, "score": s, "probability": p} for c, s, p in zip(q.candidates, scores, probs)],
        }
    threshold, source = _threshold(q, calibration)
    probs = [sigmoid(s / T) for s in scores]
    if q.type == "binary":
        return out | {
            "score": scores[0],
            "p_yes": probs[0],
            "p_no": 1 - probs[0],
            "selected": probs[0] >= threshold,
            "threshold": threshold,
            "threshold_source": source,
        }
    return out | {
        "selected": [c.id for c, p in zip(q.candidates, probs) if p >= threshold],
        "threshold": threshold,
        "threshold_source": source,
        "candidates": [{"id": c.id, "score": s, "probability": p} for c, s, p in zip(q.candidates, scores, probs)],
    }


def classify_many(scorer, requests, calibration=None) -> tuple[list[list[dict]], dict]:
    """Every request in one scorer call (each distinct state encoded once), then one decision per question."""
    reqs = [r if isinstance(r, Request) else parse_request(r) for r in requests]
    per_request, stats = scorer.score_requests(reqs)
    return [[decide(q, s, calibration) for q, s in zip(r.questions, qs)] for r, qs in zip(reqs, per_request)], stats


def run_meta(scorer, calibration=None) -> dict:
    return scorer.meta | {"calibration": (calibration or {}).get("fit")}


def classify(scorer, request, calibration=None) -> dict:
    t0 = time.perf_counter()
    [res], stats = classify_many(scorer, [request], calibration)
    return {"questions": res, "meta": run_meta(scorer, calibration) | stats | {"total_ms": 1e3 * (time.perf_counter() - t0)}}
