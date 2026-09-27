"""Decisions-API shape compatibility, with a fake scorer (mapping and HTTP plumbing only, no model quality)."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from personal_jev.schemas import ValidationError
from personal_jev.server import answers_from, compat_to_request, make_handler
from tests.test_logic import FakeScorer

EXAMPLE = {  # the request shape from the OpenRouter decisions example
    "model": "any-model-id",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this message convey urgency?",
                      "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
        "plain": {"type": "noul", "instructions": "Is money involved?"},
    },
}


def test_example_request_maps_and_answers_have_the_documented_shape():
    from personal_jev.classify import classify
    req, decode = compat_to_request(EXAMPLE)
    a = answers_from(classify(FakeScorer(), req), decode)
    assert list(a) == list(EXAMPLE["questions"])
    assert 0 <= a["is_urgent"]["noul"] <= 1 and 0 <= a["plain"]["noul"] <= 1
    assert a["department"]["choice"] in EXAMPLE["questions"]["department"]["criteria"]
    assert sum(a["department"]["probabilities"].values()) == pytest.approx(1)
    probs = a["frustration"]["probabilities"]
    assert list(probs) == ["Calm", "Frustrated", "Very angry"] and sum(probs.values()) == pytest.approx(1)
    assert a["frustration"]["score"] == pytest.approx(probs["Frustrated"] * 0.5 + probs["Very angry"])


@pytest.mark.parametrize("q", [
    {"type": "ordinal", "instructions": "x", "criteria": ["a", "b"]},
    {"type": "choice", "instructions": "x", "criteria": {"only": "one"}},
    {"type": "score", "instructions": "x", "criteria": ["same", "same"]},
    {"type": "noul", "instructions": "x", "criteria": {"yes": "a", "no": "b"}},
    {"type": "choice", "instructions": " ", "criteria": {"a": "x", "b": "y"}},
])
def test_invalid_compat_questions(q):
    with pytest.raises(ValidationError):
        compat_to_request({"state": "s", "questions": {"q": q}})


def test_http_routes():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeScorer()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        code, body = post("/api/alpha/decisions", EXAMPLE)
        assert code == 200 and set(body["answers"]) == set(EXAMPLE["questions"]) and "not the hosted model" in body["meta"]["note"]
        code, body = post("/classify", json.load(open("examples/request.json")))
        assert code == 200 and [q["id"] for q in body["questions"]] == ["urgent", "team", "tags"]
        assert post("/api/alpha/decisions", {"state": "", "questions": EXAMPLE["questions"]})[0] == 400
        assert post("/nope", {})[0] == 404
    finally:
        httpd.shutdown()


def test_options_in_question_reaches_the_scorer():
    """--options-in-question: choice/score questions get their option list in the instruction; yes/no ones do not."""
    seen = []

    class Recording(FakeScorer):
        def score(self, texts, *a, **k):
            seen.extend(texts)
            return super().score(texts, *a, **k)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Recording(), options_in_question=True))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/api/alpha/decisions", json.dumps(EXAMPLE).encode(),
                                     {"Content-Type": "application/json"})
        json.loads(urllib.request.urlopen(req).read())
    finally:
        httpd.shutdown()
    dept = [t for t in seen if "Which team should handle this?" in t]
    assert dept and all("Options (exactly one is correct):" in t and "- billing: Payments, invoicing, refunds" in t for t in dept)
    assert not any("Options (" in t for t in seen if "Is money involved?" in t)
