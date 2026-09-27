"""Request parsing and validation. Our own schema; not Jev's."""
from dataclasses import dataclass

TYPES = ("binary", "multiclass", "multilabel")


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instruction: str
    candidates: tuple[Candidate, ...] = ()
    threshold: float | None = None  # binary / multilabel decision threshold
    abstain_below: float | None = None  # multiclass: return no selection if max p < this


@dataclass(frozen=True)
class Request:
    state: str
    questions: tuple[Question, ...]


def _text(d, key, where):
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValidationError(f"{where}: '{key}' must be a non-empty string")
    return v


def _prob(v, key, where):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:  # also rejects NaN
        raise ValidationError(f"{where}: '{key}' must be a number in [0, 1], got {v!r}")
    return float(v)


def _keys(d, allowed, where):
    if not isinstance(d, dict):
        raise ValidationError(f"{where}: must be an object")
    extra = set(d) - set(allowed)
    if extra:
        raise ValidationError(f"{where}: unknown field(s) {sorted(extra)}")


def parse_question(d, where="question") -> Question:
    _keys(d, {"id", "type", "instruction", "candidates", "threshold", "abstain_below"}, where)
    qid, qtype = _text(d, "id", where), d.get("type")
    where = f"question '{qid}'"
    if qtype not in TYPES:
        raise ValidationError(f"{where}: 'type' must be one of {TYPES}, got {qtype!r}")
    instruction = _text(d, "instruction", where)
    threshold, abstain = _prob(d.get("threshold"), "threshold", where), _prob(d.get("abstain_below"), "abstain_below", where)
    raw = d.get("candidates")
    if qtype == "binary":
        if raw is not None:
            raise ValidationError(f"{where}: binary questions take no candidates")
        if abstain is not None:
            raise ValidationError(f"{where}: 'abstain_below' is multiclass-only")
        return Question(qid, qtype, instruction, threshold=threshold)
    if not isinstance(raw, list) or not raw:
        raise ValidationError(f"{where}: 'candidates' must be a non-empty list")
    if qtype == "multiclass" and len(raw) < 2:
        raise ValidationError(f"{where}: multiclass needs at least 2 candidates")
    if qtype == "multiclass" and threshold is not None:
        raise ValidationError(f"{where}: multiclass uses 'abstain_below', not 'threshold'")
    if qtype == "multilabel" and abstain is not None:
        raise ValidationError(f"{where}: 'abstain_below' is multiclass-only")
    cands = []
    for i, c in enumerate(raw):
        cw = f"{where} candidate {i}"
        _keys(c, {"id", "description"}, cw)
        cands.append(Candidate(_text(c, "id", cw), _text(c, "description", cw)))
    ids = [c.id for c in cands]
    if len(set(ids)) != len(ids):
        raise ValidationError(f"{where}: duplicate candidate ids {sorted({i for i in ids if ids.count(i) > 1})}")
    return Question(qid, qtype, instruction, tuple(cands), threshold, abstain)


def parse_request(d) -> Request:
    _keys(d, {"state", "questions"}, "request")
    state = _text(d, "state", "request")
    qs = d.get("questions")
    if not isinstance(qs, list) or not qs:
        raise ValidationError("request: 'questions' must be a non-empty list")
    questions = tuple(parse_question(q, f"question {i}") for i, q in enumerate(qs))
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValidationError(f"request: duplicate question ids {sorted({i for i in ids if ids.count(i) > 1})}")
    return Request(state, questions)
