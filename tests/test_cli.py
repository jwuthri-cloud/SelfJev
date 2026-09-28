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


def test_a_missing_default_adapter_downloads_the_published_one(monkeypatch, tmp_path):
    import sys
    import types

    got = []
    hub = types.SimpleNamespace(snapshot_download=lambda repo, allow_patterns: got.append((repo, allow_patterns)) or str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.chdir(tmp_path)  # no weights/ here, as after `pip install selfjev[serve]`
    assert cli._adapter(cli.DEFAULT_ADAPTER) == str(tmp_path)
    assert got == [(cli.HUB_ADAPTER, cli.ADAPTER_FILES)]
    (tmp_path / "mine").mkdir()
    assert cli._adapter("mine") == "mine"  # a local directory is used as is
    try:
        cli._adapter("./runs/missing")
    except SystemExit as e:
        assert "adapter not found" in str(e)
    else:
        raise AssertionError("a missing local path must not be sent to the Hub")


def test_a_missing_extra_names_the_pip_install(monkeypatch):
    def run(a):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    monkeypatch.setattr(cli, "_run", run)
    try:
        cli.main(["classify", "examples/request.json", "--adapter", "examples"])
    except SystemExit as e:
        assert "pip install 'selfjev[serve]'" in str(e)
    else:
        raise AssertionError("expected SystemExit")
