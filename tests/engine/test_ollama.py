"""The Ollama engine: its prompts equal the tree engine's tokens, and its readout against a fake Ollama."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from selfjev.core.schemas import InputTooLong, parse_question, parse_request
from selfjev.engine.ollama import INSTRUCTION, OllamaScorer, prompts


def test_prompts_equal_the_tree_engines_text():
    """Same text, not same tokens: the tree engine tokenizes the space after "Question:" alone, Ollama merges it into the next word.
    Needs the Qwen3.5 tokenizer in the local Hugging Face cache (no download); skipped otherwise."""
    from transformers import AutoTokenizer

    from selfjev.engine.qwen35 import BASE, Qwen35Scorer

    try:
        tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer not cached")
    sc = object.__new__(Qwen35Scorer)
    sc.tokenizer, sc.max_length = tok, 10**9
    assert INSTRUCTION == __import__("selfjev.engine.qwen35", fromlist=["x"]).INSTRUCTION
    q = parse_question(
        {
            "id": "q",
            "type": "multiclass",
            "instruction": "Pick one.",
            "candidates": [{"id": "a", "description": "é 日本"}, {"id": "b", "description": "b"}],
        }
    )
    for state in ["The doc.\nTwo lines", ("part one", "part two")]:
        root, _, _ = sc.root(state)
        entry = sc.entry(state, q)
        assert prompts(state, q) == [tok.decode(root + b) for b in entry["branches"]]


class FakeOllama(BaseHTTPRequestHandler):
    seen: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append(body)
        if self.path == "/api/show":
            out = {}
        else:  # "yes" 3 nats above "no" for the answer "Yes", the reverse otherwise; a long prompt overflows the context
            yes = "Proposed answer: Yes" in body["prompt"]
            top = [{"token": "yes", "logprob": -0.1 if yes else -3.1}, {"token": "no", "logprob": -3.1 if yes else -0.1}]
            out = {"logprobs": [{"top_logprobs": top}], "prompt_eval_count": 40000 if "OVERFLOW" in body["prompt"] else 10}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *a):
        pass


def test_readout_and_overflow():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sc = OllamaScorer("m", f"http://127.0.0.1:{srv.server_port}")
    cands = [{"id": "a", "description": "alpha"}, {"id": "b", "description": "Yes"}]
    req = parse_request(
        {"state": "doc", "questions": [{"id": "b1", "type": "binary", "instruction": "Is it?"},
                                       {"id": "m", "type": "multiclass", "instruction": "Which?", "candidates": cands}]}
    )  # fmt: skip
    (per,), stats = sc.score_requests([req])
    assert per[0] == [pytest.approx(3.0)] and per[1] == [pytest.approx(-3.0), pytest.approx(3.0)]
    assert stats["pairs"] == 3 and stats["tokens_per_request"] == [10]
    assert FakeOllama.seen[-1]["options"]["num_ctx"] == 32768 and FakeOllama.seen[-1]["raw"] is True
    with pytest.raises(InputTooLong):
        sc.score_requests([parse_request({"state": "OVERFLOW", "questions": [{"id": "b", "type": "binary", "instruction": "?"}]})])
    srv.shutdown()
