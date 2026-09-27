"""Jev's wire format <-> the internal schema (docs/api.md). Our computation for each question type:

noul    no criteria -> binary yes/no, noul = p_yes; {"true": d1, "false": d2} -> a 2-way choice, noul = P(true)
choice  {key: description} -> multiclass over "key: description" (the key alone when the description is None)
score   [level_0, ..., level_k] -> multiclass over the levels; score = sum_i i * p_i in level units, legend, confidence
multi   {key: description} -> multilabel (an independent yes/no per option); multi = options with p >= 0.5
"""

import json

from ..types import Choice, ChoiceAnswer, DecisionRequest, MultiAnswer, Noul, NoulAnswer, Score, ScoreAnswer, confidence


def _label(key: str, description: str | None) -> str:
    text = key.replace("_", " ")
    return f"{text}: {description}" if description else text


def to_native(req: DecisionRequest) -> dict:
    """-> a request in the internal schema (selfjev.core.schemas); objects and arrays in `state` become JSON text."""
    state = req.state if isinstance(req.state, str) else json.dumps(req.state, ensure_ascii=False)
    questions = []
    for qid, q in req.questions.items():
        base = {"id": qid, "instruction": q.instructions}
        if isinstance(q, Noul) and q.criteria is None:
            questions.append(base | {"type": "binary"})
        elif isinstance(q, Noul):
            cands = [{"id": "true", "description": q.criteria["true"]}, {"id": "false", "description": q.criteria["false"]}]
            questions.append(base | {"type": "multiclass", "candidates": cands})
        elif isinstance(q, Score):
            questions.append(
                base | {"type": "multiclass", "candidates": [{"id": str(i), "description": v} for i, v in enumerate(q.criteria)]}
            )
        else:
            kind = "multiclass" if isinstance(q, Choice) else "multilabel"
            questions.append(base | {"type": kind, "candidates": [{"id": k, "description": _label(k, d)} for k, d in q.criteria.items()]})
    return {"state": state, "questions": questions}


def to_answers(req: DecisionRequest, results: list[dict]) -> dict:
    """Internal results (selfjev.core.answers.decide), in question order -> Jev-shaped answers."""
    out = {}
    for (qid, q), r in zip(req.questions.items(), results, strict=True):
        probs = {c["id"]: c["probability"] for c in r.get("candidates", ())}
        if isinstance(q, Noul):
            out[qid] = NoulAnswer(noul=r["p_yes"] if r["type"] == "binary" else probs["true"])
        elif isinstance(q, Score):
            p = [probs[str(i)] for i in range(len(q.criteria))]
            out[qid] = ScoreAnswer(
                score=sum(i * v for i, v in enumerate(p)),
                probabilities={str(i): v for i, v in enumerate(p)},
                legend={str(i): v for i, v in enumerate(q.criteria)},
                confidence=confidence(p),
            )
        elif isinstance(q, Choice):
            out[qid] = ChoiceAnswer(choice=r["selected"], probabilities=probs, confidence=confidence(list(probs.values())))
        else:
            out[qid] = MultiAnswer(multi=r["selected"], probabilities=probs)
    return out
