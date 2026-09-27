"""Micro-batches keep every question about a text together and cover each item once."""

import random

from selfjev.training.batching import micro_batches


def test_micro_batches_keep_states_whole_and_cover_every_item():
    rng = random.Random(0)
    roots = [[0] * rng.randint(10, 400) for _ in range(40)]
    items = [{"state": s, "q": [1] * 5, "ids": [[2] * 3] * (1 + s % 4)} for s in range(40) for _ in range(1 + s % 3)]
    batches = micro_batches(items, roots, 1500, random.Random(1))
    assert sorted(i for b in batches for i in b) == list(range(len(items)))
    owner = {items[i]["state"]: n for n, b in enumerate(batches) for i in b}
    assert all(owner[items[i]["state"]] == n for n, b in enumerate(batches) for i in b)  # a state never straddles two batches
