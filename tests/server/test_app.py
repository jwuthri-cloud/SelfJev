"""The HTTP server with a fake scorer: Jev's request/answer shape, errors, auth, batching isolation, metadata routes.
No model quality is tested here."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from selfjev.core.schemas import InputTooLong
from selfjev.server.app import create_app
from selfjev.server.batching import Batcher
from tests.fakes import FakeScorer

EXAMPLE = {  # Jev's documented request shape, plus a multi question
    "model": "selfjev-4b",
    "state": "Help! My payouts have been failing for 3 days and the invoice was charged twice.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "plain": {"type": "noul", "instructions": "Is money involved?"},
        "topics": {
            "type": "multi",
            "instructions": "Which topics?",
            "criteria": {"payouts": "payouts", "invoice": "an invoice", "login": None},
        },
    },
}


def client(scorer=None, **kw):
    return TestClient(create_app(scorer or FakeScorer(), **kw))


def test_answers_have_jevs_shape():
    with client() as c:
        r = c.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200 and r.headers["x-request-id"].startswith("req_")
    body = r.json()
    a = body["answers"]
    assert body["id"].startswith("dec_") and body["model"] == "selfjev-4b" and body["usage"]["input_tokens"] > 0
    assert list(a) == list(EXAMPLE["questions"])
    assert set(a["is_urgent"]) == {"type", "noul"} and 0 <= a["is_urgent"]["noul"] <= 1 and 0 <= a["plain"]["noul"] <= 1
    dept = a["department"]
    assert dept["choice"] in EXAMPLE["questions"]["department"]["criteria"] and sum(dept["probabilities"].values()) == pytest.approx(1)
    assert dept["confidence"] == pytest.approx((3 * max(dept["probabilities"].values()) - 1) / 2)
    score = a["frustration"]
    p = [score["probabilities"][k] for k in ("0", "1", "2")]
    assert score["score"] == pytest.approx(p[1] + 2 * p[2]) and score["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    multi = a["topics"]
    assert set(multi["multi"]) <= {"payouts", "invoice", "login"}
    assert multi["multi"] == [k for k, v in multi["probabilities"].items() if v >= 0.5]


def test_openrouter_and_v1_paths_give_the_same_answers():
    with client() as c:
        answers = [c.post(p, json=EXAMPLE).json()["answers"] for p in ("/v1/systemone", "/api/alpha/decisions", "/v1/decisions")]
    assert answers[0] == answers[1] == answers[2]


@pytest.mark.parametrize(
    "mutate,param",
    [
        (lambda b: b["questions"]["department"].update(criteria={"only": None}), "questions.department.criteria"),
        (lambda b: b["questions"]["frustration"].update(criteria=["one"]), "questions.frustration.criteria"),
        (lambda b: b["questions"]["plain"].update(type="ordinal"), "questions.plain"),
        (lambda b: b["questions"]["plain"].update(instructions="  "), "questions.plain.instructions"),
        (lambda b: b.pop("state"), "state"),
        (lambda b: b.update(questions={}), None),
    ],
)
def test_invalid_requests_are_422_with_the_offending_field(mutate, param):
    body = json.loads(json.dumps(EXAMPLE))
    mutate(body)
    with client() as c:
        r = c.post("/v1/systemone", json=body)
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request_error"
    if param:
        assert r.json()["error"]["param"].startswith(param)


def test_model_names():
    with client() as c:
        assert c.post("/v1/systemone", json=EXAMPLE | {"model": "jev-latest"}).status_code == 200  # Jev clients work unchanged
        r = c.post("/v1/systemone", json=EXAMPLE | {"model": "gpt-99"})
    assert r.status_code == 404 and r.json()["error"] == {
        "type": "not_found_error",
        "message": r.json()["error"]["message"],
        "param": "model",
    }


def test_api_keys():
    with client(api_keys=["secret"]) as c:
        assert c.post("/v1/systemone", json=EXAMPLE).status_code == 401
        assert (
            c.post("/v1/systemone", json=EXAMPLE, headers={"Authorization": "Bearer nope"}).json()["error"]["type"]
            == "authentication_error"
        )
        assert c.post("/v1/systemone", json=EXAMPLE, headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/health").status_code == 200  # load balancers check it without a key


def test_option_lists_reach_the_scorer_and_objects_are_serialized():
    seen = []

    class Recording(FakeScorer):
        def score_requests(self, reqs):
            seen.extend((r.state, q.instruction) for r in reqs for q in r.questions)
            return super().score_requests(reqs)

    with client(Recording()) as c:
        c.post("/v1/systemone", json=EXAMPLE | {"state": {"ticket": 4411, "text": "charged twice"}})
    dept = [i for _, i in seen if i.startswith("Which team")]
    assert dept and "Options (exactly one is correct):" in dept[0] and "- billing: Payments, invoicing, refunds" in dept[0]
    assert not any("Options (" in i for _, i in seen if i.startswith("Is money"))
    assert json.loads(seen[0][0]) == {"ticket": 4411, "text": "charged twice"}


def test_too_long_input_is_422_and_a_full_queue_is_529():
    class Short(FakeScorer):
        def score_requests(self, reqs):
            raise InputTooLong([(0, 40000)], 32768)

    with client(Short()) as c:
        r = c.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 422 and "32768" in r.json()["error"]["message"]
    with client(max_queue=0) as c:
        r = c.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 529 and r.headers["retry-after"] == "1" and r.json()["error"]["type"] == "overloaded_error"


def test_a_bug_is_a_json_500_with_the_request_id():
    class Broken(FakeScorer):
        def score_requests(self, reqs):
            raise RuntimeError("boom")

    with client(Broken()) as c:
        r = c.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 500 and r.json()["error"]["type"] == "api_error" and r.headers["x-request-id"] in r.json()["error"]["message"]


def test_internal_schema_route_models_health_metrics():
    req = json.loads(Path("examples/request.json").read_text())
    with client() as c:
        r = c.post("/classify", json=req)
        assert r.status_code == 200 and [q["id"] for q in r.json()["questions"]] == [q["id"] for q in req["questions"]]
        assert c.get("/v1/models").json()["data"][0]["id"] == "selfjev-4b"
        assert c.get("/health").json()["status"] == "ok"
        text = c.get("/metrics").text
    assert 'selfjev_requests_total{route="/classify",status="200"} 1' in text and "selfjev_queue_depth 0" in text


def test_batcher_isolates_a_failing_request():
    def fn(items):
        if "bad" in items:
            raise ValueError("bad item")
        return [i.upper() for i in items]

    async def main():
        b = Batcher(fn, max_batch_requests=8, max_wait_ms=50)
        await b.start()
        try:
            return await asyncio.gather(b.submit("good"), b.submit("bad"), b.submit("fine"), return_exceptions=True)
        finally:
            await b.stop()

    good, bad, fine = asyncio.run(main())
    assert good == "GOOD" and fine == "FINE" and isinstance(bad, ValueError)
