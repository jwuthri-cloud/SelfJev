"""Calibration: temperatures recover a known scale, test labels are refused, a file is bound to its model."""

import json
import math
import random

import pytest

from selfjev.evaluation import calibration


def _report(tmp_path, split, preds, name="r.json", **meta):
    p = tmp_path / name
    p.write_text(
        json.dumps(
            {
                "meta": {
                    "model": "m",
                    "revision": "r",
                    "adapter": None,
                    "adapter_sha256": None,
                    "prompt_sha": "p",
                    "data": [],
                    "splits": [split],
                    "calibration": None,
                }
                | meta,
                "predictions": preds,
            }
        )
    )
    return p


def test_temperature_recovers_known_scale(tmp_path):
    rng = random.Random(0)
    preds = []
    for _ in range(3000):
        z = rng.gauss(0, 2)
        y = rng.random() < 1 / (1 + math.exp(-z))
        preds.append({"split": "calibration", "type": "binary", "scores": [3.0 * z], "target": y})  # overconfident by 3x
    cal = calibration.fit(_report(tmp_path, "calibration", preds))
    assert cal["temperature"]["binary"] == pytest.approx(3.0, rel=0.1)
    assert "multiclass" in cal["fit"]["skipped"]


def test_calibration_refuses_test_labels(tmp_path):
    preds = [{"split": "test", "type": "binary", "scores": [1.0], "target": True}] * 50
    with pytest.raises(calibration.LeakageError):
        calibration.fit(_report(tmp_path, "test", preds))
    mixed = _report(tmp_path, "calibration", [p | {"split": "calibration"} for p in preds] + preds[:1], "mixed.json")
    with pytest.raises(calibration.LeakageError):
        calibration.fit(mixed)
    ok = _report(tmp_path, "calibration", [p | {"split": "calibration"} for p in preds], "ok.json")
    with pytest.raises(calibration.LeakageError):
        calibration.fit(ok, threshold_report=_report(tmp_path, "test", preds, "t.json"))


def test_calibration_file_is_bound_to_its_model(tmp_path):
    preds = [{"split": "calibration", "type": "binary", "scores": [float(i % 7 - 3)], "target": i % 2 == 0} for i in range(60)]
    out = tmp_path / "cal.json"
    calibration.fit(_report(tmp_path, "calibration", preds), out=out)
    good = {"model": "m", "revision": "r", "adapter": None, "adapter_sha256": None}
    assert calibration.load(out, good, "p")["fit"]["file"] == str(out)
    with pytest.raises(ValueError):
        calibration.load(out, good | {"adapter_sha256": "abc"}, "p")
    with pytest.raises(ValueError):
        calibration.load(out, good, "other-prompt")


def test_best_threshold_maximises_f1():
    items = [(2.0, True), (1.0, True), (0.5, False), (-1.0, True), (-2.0, False)]
    t = calibration.best_threshold(items, 1.0)
    p = lambda s: 1 / (1 + math.exp(-s))
    assert p(-2.0) < t < p(-1.0)  # predicting the top 4 positive gives F1 = 6/7, the maximum
