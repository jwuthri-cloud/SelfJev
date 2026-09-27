"""The internal request schema: valid requests parse, invalid ones fail with a message that names the problem."""

import json
from pathlib import Path

import pytest

from selfjev.core.schemas import ValidationError, parse_request

REQ = json.loads(Path("examples/request.json").read_text())


def test_valid_request_parses():
    r = parse_request(REQ)
    assert [q.type for q in r.questions] == ["binary", "multiclass", "multilabel"]


@pytest.mark.parametrize(
    "mutate,msg",
    [
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
    ],
)
def test_invalid_requests(mutate, msg):
    r = json.loads(json.dumps(REQ))
    mutate(r)
    with pytest.raises(ValidationError, match=msg):
        parse_request(r)
