"""Metrics against brute-force and hand-computed values."""

import itertools
import random

import pytest

from selfjev.evaluation.evaluate import auroc, macro_f1, reliability


def test_auroc_matches_brute_force_with_ties():
    rng = random.Random(3)
    ys = [rng.random() < 0.4 for _ in range(300)]
    ps = [round(rng.random(), 1) for _ in range(300)]
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    brute = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))
    assert auroc(ys, ps) == pytest.approx(brute)


def test_reliability_and_macro_f1():
    ece, _rows = reliability([0.95, 0.95, 0.05, 0.05], [True, False, False, False])
    assert ece == pytest.approx(0.5 * 0.45 + 0.5 * 0.05)
    assert macro_f1([("a", "a"), ("b", "a"), ("b", "b")]) == pytest.approx((2 / 3 + 2 / 3) / 2)
