"""TypeSafe's own Python SDK (typesafe-sdk) against a selfjev server: only TYPESAFE_BASE_URL and TYPESAFE_API_KEY change."""

import socket
import threading
import time

import pytest
import uvicorn

from selfjev.server.app import create_app
from tests.fakes import FakeScorer

typesafe = pytest.importorskip("typesafe_sdk")


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(FakeScorer(), api_keys=["secret"]), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


@pytest.fixture
def client(base_url, monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", base_url)
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    return typesafe.TypeSafeClient()  # configured by the environment alone, as a Jev user's code is


def test_every_question_type_answers_with_the_default_jev_model_name(client):
    from typesafe_sdk import Choice, Noul, Score

    res = client.system_one(
        state={"subject": "Duplicate charge", "message": "I was charged twice. Refund the duplicate."},
        questions={
            "refund": Noul(instructions="Does the customer want a refund?"),
            "billing": Noul(criteria={"true": "about a payment"}),  # one side and no instructions, as Jev allows
            "team": Choice(instructions={"task": "route the ticket"}, criteria={"billing": {"covers": "payments"}, "support": None}),
            "urgency": Score(instructions="How urgent?", criteria=["low", "medium", "high"]),
            "plain": {"type": "noul", "instructions": "Is this a plain dict question?"},
        },
    )
    assert res.model == "selfjev-4b"  # asked for jev-latest, TypeSafe's default
    a = res.answers
    assert 0 <= a["refund"].noul <= 1 and 0 <= a["billing"].noul <= 1 and 0 <= a["plain"].noul <= 1
    assert a["team"].choice in {"billing", "support"}
    assert set(a["urgency"].legend) == {0, 1, 2}


def test_models_list_parses(client):
    names = {m.name for m in client.models.list().models}
    assert {"selfjev-4b", "jev-latest"} <= names


def test_errors_map_to_typesafe_exceptions(client, base_url):
    from typesafe_sdk import Noul, TypeSafeAuthenticationError, TypeSafeUnprocessableEntityError

    with pytest.raises(TypeSafeAuthenticationError):
        typesafe.TypeSafeClient(api_key="wrong", base_url=base_url).system_one(state="x", questions={"q": Noul(instructions="?")})
    with pytest.raises(TypeSafeUnprocessableEntityError, match="instructions or criteria"):
        client.system_one(state="x", questions={"q": Noul()})
