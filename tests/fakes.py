"""Test doubles. FakeScorer implements the engine protocol (score_requests) with a deterministic pseudo-score per
(state, question, candidate), so tests can check that scores land on the right candidate. It says nothing about quality."""

from typing import ClassVar


def pair_text(state, q, c=None):
    return f"{state}|{q.instruction}|{c.description if c else 'Yes'}"


class FakeScorer:
    meta: ClassVar[dict] = {"model": "fake", "revision": "x", "adapter": None, "adapter_sha256": None, "prompt": "fake", "prompt_sha": "f"}

    @staticmethod
    def f(text):
        return (sum(map(ord, text)) % 997) / 50 - 10

    def score_requests(self, reqs):
        per = [
            [[self.f(pair_text(r.state, q, c)) for c in q.candidates] or [self.f(pair_text(r.state, q))] for q in r.questions] for r in reqs
        ]
        return per, {"pairs": sum(len(qs) for r in per for qs in r)}
