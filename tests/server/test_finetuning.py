"""The fine-tuning API end to end through the SDK: uploads, the job queue, serving the new adapter, and the command line a
job runs. Training itself is a stand-in script (no GPU, no model)."""

import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from selfjev import InvalidRequestError, SelfJev
from selfjev.data import validate_example
from selfjev.server import finetuning
from selfjev.server.app import create_app
from selfjev.types import FineTuningJob, Hyperparameters, Method
from tests.fakes import FakeScorer

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "tech": None}},
    "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    "topics": {"type": "multi", "instructions": "Which topics?", "criteria": {"invoice": None, "login": None}},
    "paid": {"type": "noul", "instructions": "Paid?", "criteria": {"true": "a paying customer", "false": "a free user"}},
}
ROW = {"state": "Charged twice, fix it now!", "questions": QUESTIONS,
       "answers": {"urgent": True, "team": "billing", "anger": 2, "topics": ["invoice", "invoice"], "paid": False}}  # fmt: skip
TRAIN_OK = (
    "import json, pathlib, sys; d = pathlib.Path(sys.argv[1]); (d / 'adapter').mkdir(parents=True); "
    "(d / 'train_meta.json').write_text(json.dumps({'log': [{'step': 10, 'validation': {'accuracy': 0.9}}]}))"
)
TRAIN_OOM = "print('CUDA out of memory'); raise SystemExit(1)"
TRAIN_SLOW = "import time; time.sleep(60)"


class AdapterScorer(FakeScorer):
    """FakeScorer plus the multi-adapter hooks of TreeServer(merge=False); records which adapter answered."""

    def __init__(self):
        self.active, self.loaded, self.answered_by = "default", {}, []

    def load_adapter(self, name, path):
        assert Path(path).is_dir()
        self.loaded[name] = path

    def set_adapter(self, name):
        assert name == "default" or name in self.loaded
        self.active = name

    def score_requests(self, reqs):
        self.answered_by.append(self.active)
        return super().score_requests(reqs)


def runner(*scripts):
    """Each job runs the next script instead of `selfjev finetune|rlcd`, with the run directory as its argument."""
    todo = list(scripts)

    def run(job, train, val, init, run_dir, log):
        with log.open("w") as out:
            return subprocess.Popen(
                [sys.executable, "-c", todo.pop(0), str(run_dir)], stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )  # cancel kills its group

    return run


@pytest.fixture
def server(tmp_path, monkeypatch):
    with ExitStack() as stack:

        def start(*scripts):
            monkeypatch.setattr(finetuning, "run_training", runner(*scripts))
            scorer = AdapterScorer()
            http = stack.enter_context(TestClient(create_app(scorer, fine_tuning_home=tmp_path, init_adapter="weights/selfjev_4b")))
            return SelfJev(base_url="http://testserver", http_client=http, max_retries=0), scorer

        yield start


def upload(client, tmp_path, rows, name="train.jsonl"):
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return client.upload_file(p)


def wait(client, job_id, *statuses):
    t0 = time.time()
    while (job := client.fine_tuning_job(job_id)).status not in statuses:
        assert time.time() - t0 < 20, job
        time.sleep(0.05)
    return job


def test_uploads_become_training_rows_and_bad_lines_are_named(server, tmp_path):
    client, _ = server()
    f = upload(client, tmp_path, [ROW, "", ROW | {"state": {"ticket": 7}}])
    assert (f.rows, f.questions, f.filename) == (2, 10, "train.jsonl")
    assert [x.id for x in client.fine_tuning_jobs()] == [] and f.id in (tmp_path / "files" / f"{f.id}.json").read_text()
    rows = [json.loads(line) for line in (tmp_path / "files" / f"{f.id}.train.jsonl").read_text().splitlines()]
    for r in rows:
        validate_example(r)  # what `selfjev finetune --data` reads
    assert [r["target"] for r in rows[:5]] == [True, "billing", "2", ["invoice"], "false"]

    bad = ROW | {"answers": ROW["answers"] | {"team": "sales"}}
    with pytest.raises(InvalidRequestError) as e:
        upload(client, tmp_path, [ROW, bad])
    assert e.value.param == "file.line_2" and "team" in e.value.message
    for rows, param in (([ROW | {"state": " "}], "file.line_1"), ([], "file")):
        with pytest.raises(InvalidRequestError) as e:
            upload(client, tmp_path, rows)
        assert e.value.param == param
    (tmp_path / "latin1.jsonl").write_bytes(b"\xff\xfe")
    with pytest.raises(InvalidRequestError, match="UTF-8"):
        client.upload_file(tmp_path / "latin1.jsonl")


def test_a_finished_job_is_served_as_a_new_model_and_survives_a_restart(server, tmp_path):
    client, scorer = server(TRAIN_OK)
    f = upload(client, tmp_path, [ROW])
    job = client.create_fine_tuning_job(f.id, method="rlcd", hyperparameters={"reward": {"log": 1, "confident_miss": 2}}, suffix="acme")
    assert job.status == "queued" and job.method.hyperparameters.reward == {"log": 1, "confident_miss": 2}
    job = wait(client, job.id, "succeeded", "failed")
    assert job.status == "succeeded" and job.fine_tuned_model == f"selfjev-4b:ft-acme-{job.id[6:14]}"
    assert job.fine_tuned_model in [m["id"] for m in client.models()]

    res = client.system_one(ROW["state"], QUESTIONS, model=job.fine_tuned_model)
    assert res.model == job.fine_tuned_model and set(res.answers) == set(QUESTIONS)
    assert scorer.answered_by[-1] == job.fine_tuned_model and scorer.active == "default"
    client.system_one(ROW["state"], QUESTIONS)
    assert scorer.answered_by[-1] == "default"
    messages = [e.message for e in client.fine_tuning_events(job.id)]
    assert messages == ["queued: rlcd on 5 questions", "running", f"succeeded: {job.fine_tuned_model}", "validation at step 10"]

    client, scorer = server()  # a restart on the same home serves the model again
    assert job.fine_tuned_model in scorer.loaded and job.fine_tuned_model in [m["id"] for m in client.models()]


def test_failed_and_cancelled_jobs(server, tmp_path):
    client, _ = server(TRAIN_OOM, TRAIN_SLOW, TRAIN_OK)
    f = upload(client, tmp_path, [ROW])
    failed = wait(client, client.create_fine_tuning_job(f.id).id, "succeeded", "failed")
    assert failed.status == "failed" and "CUDA out of memory" in failed.error

    slow = wait(client, client.create_fine_tuning_job(f.id).id, "running")
    assert client.cancel_fine_tuning_job(slow.id).status == "cancelled"
    after = wait(client, client.create_fine_tuning_job(f.id).id, "succeeded", "failed")  # the queue moved on: the process died
    assert after.status == "succeeded" and client.fine_tuning_job(slow.id).status == "cancelled"
    with pytest.raises(ValueError, match="suffix"):  # checked by the SDK before sending
        client.create_fine_tuning_job(f.id, suffix="Not Valid")
    with pytest.raises(Exception, match="404"):
        client.create_fine_tuning_job("file_missing")


def test_a_job_runs_the_cli_with_its_hyperparameters(tmp_path, monkeypatch):
    from selfjev import cli

    seen, got = {}, {}
    monkeypatch.setattr(finetuning.subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd))
    monkeypatch.setattr("selfjev.training.finetune.train", lambda *a, **kw: got.update(args=a, kw=kw))
    hp = Hyperparameters(epochs=2, learning_rate=1e-5, reward={"log": 1, "confident_miss": 2}, samples=4)
    job = FineTuningJob(id="ftjob_x", model="selfjev-4b", status="queued", created_at=0, training_file="file_a",
                        method=Method(type="rlcd", hyperparameters=hp))  # fmt: skip
    finetuning.run_training(job, tmp_path / "t.jsonl", tmp_path / "v.jsonl", "weights/selfjev_4b", tmp_path / "run", tmp_path / "log")
    assert seen["cmd"][:3] == [sys.executable, "-m", "selfjev.cli"]
    cli.main(seen["cmd"][3:])
    a = got["args"]
    assert a[:5] == ("rlcd", str(tmp_path / "t.jsonl"), str(tmp_path / "run"), str(tmp_path / "v.jsonl"), "weights/selfjev_4b")
    assert (a[7], a[8]) == (2, 1e-5) and got["kw"]["reward_weights"] == {"log": 1.0, "confident_miss": 2.0} and got["kw"]["samples"] == 4
