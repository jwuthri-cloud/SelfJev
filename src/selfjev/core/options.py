"""Every option listed in the question text ("all options"): each leaf still judges one candidate but now sees all the
alternatives. Adapters trained with it (`selfjev finetune`, selfjev-4b) need the same transform on every request:
`selfjev serve` and `selfjev classify` apply it (with_option_lists), `selfjev eval` reads copies that have it (data/ova/).
Options are listed in a fixed random order seeded by the given key.

The "option pointers" variant (numbered options, "option k" leaves) was a dead end; it is at tag
archive/pre-cleanup-2026-09-27."""

import hashlib
import random

HEAD = {"multiclass": "Options (exactly one is correct):", "multilabel": "Options (any number can be correct, possibly none):"}


def with_options(q: dict, seed: str) -> dict:
    """Question dict (type, instruction, candidates) -> the same question with its options in the instruction."""
    if q["type"] == "binary":
        return q
    opts = list(q["candidates"])
    random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)).shuffle(opts)
    return q | {"instruction": f"{q['instruction']}\n{HEAD[q['type']]}\n" + "\n".join(f"- {c['description']}" for c in opts)}


def with_option_lists(request: dict) -> dict:
    """A request in the internal schema with every question's options listed, each seeded by its question id."""
    return request | {"questions": [with_options(q, str(q.get("id"))) if isinstance(q, dict) else q for q in request.get("questions", [])]}
