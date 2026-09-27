"""Request builders and answer readers shared by the data and evaluation scripts (no network)."""

import json

from selfjev.data.providers import NOUL_CRITERIA, call_cost, decide, jev_request, llm_request, to_openai

EXS = [
    {"id": "s-q0", "question": {"type": "binary", "instruction": "Refund?"}},
    {
        "id": "s-q1",
        "question": {
            "type": "multiclass",
            "instruction": "Team?",
            "candidates": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
        },
    },
    {
        "id": "s-q2",
        "question": {
            "type": "multilabel",
            "instruction": "Tags?",
            "candidates": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}],
        },
    },
]


def test_call_cost_adds_byok_upstream_cost():
    assert call_cost({"usage": {"cost": 0.5}}) == 0.5
    assert call_cost({"usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.2}}}) == 0.2
    assert call_cost({"usage": {"cost": 0.1, "cost_details": {"upstream_inference_cost": 0.2}}}) == 0.1  # not BYOK
    assert call_cost({}) == 0.0


def test_jev_request_maps_types_to_jev_questions():
    body, keys = jev_request("m", "text", EXS)
    qs = body["questions"]
    assert body["model"] == "m" and body["state"] == "text"
    assert qs["q0"] == {"type": "noul", "instructions": "Refund?", "criteria": NOUL_CRITERIA}
    assert qs["q1"]["type"] == "choice" and qs["q1"]["criteria"] == {"a": "A", "b": "B"}
    assert [qs[k]["type"] for k in ("q2_0", "q2_1")] == ["noul", "noul"]  # multilabel: one noul per candidate
    assert keys == [("s-q0", ["q0"]), ("s-q1", ["q1"]), ("s-q2", ["q2_0", "q2_1"])]


def test_llm_request_schema_and_openai_body():
    body, keys = llm_request("openai/gpt", "text", EXS, "low")
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["q0", "q1", "q2"] and schema["properties"]["q1"]["required"] == ["a", "b"]
    assert keys == [("s-q0", ["q0"]), ("s-q1", ["q1"]), ("s-q2", ["q2"])]
    user = json.loads(body["messages"][1]["content"].split("QUESTIONS (JSON):\n", 1)[1])
    assert user[1]["candidates"] == {"a": "A", "b": "B"}
    oa_body = to_openai(body, "gpt", "low")
    assert oa_body["model"] == "gpt" and oa_body["reasoning_effort"] == "low" and oa_body["max_completion_tokens"] == body["max_tokens"]
    assert not {"usage", "reasoning", "max_tokens"} & set(oa_body)


def test_decide_reads_judge_probabilities_in_the_target_type():
    assert decide(EXS[0], {"p_yes": 0.2}) == (False, 0.8)
    assert decide(EXS[1], {"a": 0.25, "b": 0.75}) == ("b", 0.75)
    assert decide(EXS[1], {"a": 0.5, "b": 0.5})[0] == "a"  # ties go to the first candidate
    assert decide(EXS[2], {"x": 0.9, "y": 0.3}) == (["x"], 0.7)
