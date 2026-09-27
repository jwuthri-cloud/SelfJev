"""The command line on a fake scorer: `selfjev classify` lists every option in the question, as selfjev-4b was trained."""

import json

from selfjev import cli
from tests.fakes import FakeScorer


def test_classify_lists_the_options_unless_told_not_to(monkeypatch, capsys):
    seen = []

    class Recording(FakeScorer):
        def score_requests(self, reqs):
            seen.extend(q.instruction for r in reqs for q in r.questions)
            return super().score_requests(reqs)

    monkeypatch.setattr(cli, "_scorer", lambda a: Recording())
    cli.main(["classify", "examples/request.json"])
    assert json.loads(capsys.readouterr().out)["questions"] and any("Options (" in i for i in seen)
    seen.clear()
    cli.main(["classify", "examples/request.json", "--no-options-in-question"])
    assert seen and not any("Options (" in i for i in seen)
