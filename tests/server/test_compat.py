"""Jev's looser question schema -> the internal one: JSON text, optional instructions, a noul described on one side."""

import json

import pytest
from pydantic import ValidationError

from selfjev.server.compat import DEFAULT_INSTRUCTIONS, to_native
from selfjev.types import Choice, DecisionRequest, Noul


def _native(**questions):
    return {q["id"]: q for q in to_native(DecisionRequest(state="s", questions=questions))["questions"]}


def test_json_instructions_and_criteria_become_json_text():
    criteria = {"billing": {"covers": "payments"}, "other": None}
    q = _native(t={"type": "choice", "instructions": {"task": "route"}, "criteria": criteria})["t"]
    assert json.loads(q["instruction"]) == {"task": "route"}
    assert [c["description"] for c in q["candidates"]] == ['billing: {"covers": "payments"}', "other"]


def test_missing_instructions_fall_back_to_the_criteria():
    q = _native(t={"type": "choice", "criteria": {"a": None, "b": None}})["t"]
    assert q["instruction"] == DEFAULT_INSTRUCTIONS[Choice]


def test_a_noul_described_on_one_side_negates_it_for_the_other():
    q = _native(n={"type": "noul", "criteria": {"true": "spam", "false": None}})["n"]
    assert q["instruction"] == DEFAULT_INSTRUCTIONS[Noul]
    assert [(c["id"], c["description"]) for c in q["candidates"]] == [("true", "spam"), ("false", "not: spam")]


def test_a_noul_with_nothing_to_ask_is_refused():
    with pytest.raises(ValidationError, match="instructions or criteria"):
        Noul(criteria={"true": None})
    assert Noul("Is it spam?", {"true": None, "false": None}).criteria is None  # null sides are no criteria: a plain yes/no


def test_image_state_parts_stay_parts():
    from selfjev.core.schemas import parse_request

    url = "data:image/png;base64,iVBORw0KGgo="
    q = {"q": {"type": "noul", "instructions": "A cat?"}}
    native = to_native(DecisionRequest(state=[url, {"caption": "pet"}], questions=q))
    assert native["state"] == [url, '{"caption": "pet"}']
    assert parse_request(native).state == (url, '{"caption": "pet"}')
    assert to_native(DecisionRequest(state=["a", "b"], questions=q))["state"] == '["a", "b"]'  # no image: JSON text, as before
