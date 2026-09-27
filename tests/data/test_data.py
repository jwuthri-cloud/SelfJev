"""Dataset rows: validation, source expansion, deterministic splits."""

import pytest

from selfjev.core.schemas import ValidationError
from selfjev.data import expand_source, split_for, validate_example


def test_example_validation_and_source_expansion():
    src = {
        "source_id": "s1",
        "family": "f",
        "provenance": "test",
        "state": "x",
        "questions": [
            {"type": "binary", "instruction": "q?", "target": False, "hard_cases": ["negation"]},
            {"type": "multilabel", "instruction": "q?", "candidates": [{"id": "a", "description": "A"}], "target": []},
        ],
    }
    exs = expand_source(src)
    assert [e["id"] for e in exs] == ["s1-q0", "s1-q1"] and exs[0]["hard_cases"] == ["negation"]
    with pytest.raises(ValidationError):
        validate_example(exs[1] | {"target": ["zzz"]})
    with pytest.raises(ValidationError):
        validate_example(exs[0] | {"target": "yes"})


def test_split_for_is_deterministic_and_roughly_proportional():
    w = {"validation": 0.3, "calibration": 0.2, "test": 0.5}
    assert split_for("abc", w) == split_for("abc", w)
    counts = dict.fromkeys(w, 0)
    for i in range(5000):
        counts[split_for(f"s{i}", w)] += 1
    assert all(abs(counts[k] / 5000 - w[k]) < 0.03 for k in w)
