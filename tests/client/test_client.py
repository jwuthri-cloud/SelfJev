"""The SDK against a mock transport: request shape, typed answers, errors, retries, async, validation before sending."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

import selfjev.client as client_module
from selfjev import (
    AsyncSelfJev,
    AuthenticationError,
    Choice,
    InvalidRequestError,
    Multi,
    Noul,
    OverloadedError,
    Score,
    SelfJev,
)

QUESTIONS = {
    "refund": Noul("Does the customer ask for a refund?"),
    "team": Choice("Which team?", {"billing": "billing and refunds", "tech": "outages and bugs"}),
    "urgency": Score("How urgent?", ["not urgent", "this week", "today"]),
    "topics": {"type": "multi", "instructions": "Which topics?", "criteria": {"invoice": None, "shipping": "a delivery"}},
}
ANSWERS = {
    "id": "dec_1",
    "model": "selfjev-4b-2026-09-26",
    "answers": {
        "refund": {"type": "noul", "noul": 0.97},
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.99, "tech": 0.01}, "confidence": 0.98},
        "urgency": {"type": "score", "score": 1.43, "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
                    "legend": {"0": "not urgent", "1": "this week", "2": "today"}, "confidence": 0.36},
        "topics": {"type": "multi", "multi": ["invoice"], "probabilities": {"invoice": 0.98, "shipping": 0.02}},
    },
    "usage": {"input_tokens": 296, "output_tokens": 0},
}  # fmt: skip


def mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_request_shape_and_typed_answers():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers.get("authorization"), json.loads(request.content)
        return httpx.Response(200, json=ANSWERS)

    res = SelfJev(api_key="k", base_url="http://x/", http_client=mock(handler)).system_one("Ticket 4411: charged twice.", QUESTIONS)
    assert seen["url"] == "http://x/v1/systemone" and seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["model"] == "selfjev-4b" and body["state"] == "Ticket 4411: charged twice."
    assert body["questions"]["team"] == {"type": "choice", "instructions": "Which team?", "criteria": QUESTIONS["team"].criteria}
    assert body["questions"]["topics"]["criteria"] == {"invoice": None, "shipping": "a delivery"}  # None keys survive
    assert res.choices["team"].choice == "billing" and res.nouls["refund"].noul == 0.97
    assert res.scores["urgency"].legend["2"] == "today" and res.multis["topics"].multi == ["invoice"]
    assert res.usage.input_tokens == 296 and set(res.answers) == set(QUESTIONS)


def test_errors_carry_type_message_and_request_id():
    err = {"error": {"type": "invalid_request_error", "message": "choice needs 2 options", "param": "questions.team.criteria"}}
    c = SelfJev(http_client=mock(lambda r: httpx.Response(422, json=err, headers={"x-request-id": "req_9"})))
    with pytest.raises(InvalidRequestError) as e:
        c.system_one("x", {"q": Noul("?")})
    assert (e.value.status, e.value.type, e.value.param, e.value.request_id) == (
        422,
        "invalid_request_error",
        "questions.team.criteria",
        "req_9",
    )
    c = SelfJev(http_client=mock(lambda r: httpx.Response(401, text="no key")))
    with pytest.raises(AuthenticationError, match="no key"):
        c.system_one("x", {"q": Noul("?")})


def test_overloaded_is_retried_then_raised(monkeypatch):
    monkeypatch.setattr(client_module, "_delay", lambda attempt, r: 0)
    calls = []

    def flaky(request):
        calls.append(1)
        return (
            httpx.Response(529, json={"error": {"type": "overloaded_error", "message": "busy"}})
            if len(calls) < 3
            else httpx.Response(200, json=ANSWERS)
        )

    assert SelfJev(max_retries=2, http_client=mock(flaky)).system_one("x", QUESTIONS).model == ANSWERS["model"] and len(calls) == 3
    calls.clear()
    with pytest.raises(OverloadedError):
        SelfJev(max_retries=1, http_client=mock(flaky)).system_one("x", QUESTIONS)
    assert len(calls) == 2


def test_retry_after_header_is_respected():
    assert client_module._delay(0, httpx.Response(429, headers={"retry-after": "3"})) == 3.0
    assert 0 < client_module._delay(2, None) <= 2.0


def test_invalid_questions_fail_before_any_request():
    c = SelfJev(http_client=mock(lambda r: pytest.fail("no request should be sent")))
    with pytest.raises(ValidationError):
        c.system_one("x", {"team": Choice("Which team?", {"only": None})})
    with pytest.raises(ValidationError):
        c.system_one("x", {"u": {"type": "score", "instructions": "?", "criteria": ["one"]}})
    with pytest.raises(ValidationError):
        Multi("Topics?", {})


def test_environment_configures_the_client(monkeypatch):
    monkeypatch.setenv("SELFJEV_BASE_URL", "http://env:9")
    monkeypatch.setenv("SELFJEV_API_KEY", "envkey")
    c = SelfJev()
    assert c.base_url == "http://env:9" and c._headers()["Authorization"] == "Bearer envkey"


def test_async_client():
    async def run():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=ANSWERS))
        async with AsyncSelfJev(http_client=httpx.AsyncClient(transport=transport)) as c:
            return await c.system_one("x", QUESTIONS)

    assert asyncio.run(run()).choices["team"].confidence == 0.98


def test_image_parts_become_data_urls(tmp_path):
    from selfjev.client import _part

    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
    (tmp_path / "cat.png").write_bytes(png)
    assert _part(tmp_path / "cat.png") == _part(png) == "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAA=="
    assert _part("text") == "text"
    with pytest.raises(ValueError):
        _part(b"not an image")


def test_rows_upload_with_images_and_are_checked_first(tmp_path):
    """upload_file takes rows: images (Paths, bytes) become data URLs, a bad row fails before anything is sent."""
    from selfjev import TrainingRow

    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
    (tmp_path / "cat.png").write_bytes(png)
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "file_1", "bytes": 1, "created_at": 0, "filename": "rows.jsonl",
                                         "purpose": "fine-tune", "rows": 2, "questions": 2})  # fmt: skip

    q = {"cat": Noul("Is there a cat?")}
    rows = [{"state": [tmp_path / "cat.png", "a photo"], "questions": q, "answers": {"cat": True}},
            TrainingRow(state=[png], questions=q, answers={"cat": False})]  # fmt: skip
    assert SelfJev(http_client=mock(handler)).upload_file(rows).id == "file_1"
    lines = [json.loads(x) for x in seen["body"].replace(b"\r\n", b"\n").split(b"\n") if x.startswith(b'{"state"')]
    url = "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAA=="
    assert [r["state"] for r in lines] == [[url, "a photo"], [url]]
    assert lines[0]["questions"] == {"cat": {"type": "noul", "instructions": "Is there a cat?"}}
    with pytest.raises(ValidationError):  # the answer does not fit the question
        SelfJev(http_client=mock(lambda r: pytest.fail("nothing is sent"))).upload_file([rows[0] | {"answers": {"cat": "yes"}}])


def test_wait_fine_tuning_job_polls_until_it_ends(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)
    states = iter(["queued", "running", "running", "succeeded"])

    def handler(request):
        s = next(states)
        done = "selfjev-4b:ft-x" if s == "succeeded" else None
        return httpx.Response(200, json={"id": "ftjob_1", "model": "selfjev-4b", "status": s, "created_at": 0, "training_file": "file_1",
                                         "method": {"type": "supervised"}, "fine_tuned_model": done})  # fmt: skip

    job = SelfJev(http_client=mock(handler)).wait_fine_tuning_job("ftjob_1", poll=0)
    assert job.status == "succeeded" and job.fine_tuned_model == "selfjev-4b:ft-x"
    states = iter(["running"] * 100)
    with pytest.raises(TimeoutError):
        SelfJev(http_client=mock(handler)).wait_fine_tuning_job("ftjob_1", poll=0, timeout=0)
