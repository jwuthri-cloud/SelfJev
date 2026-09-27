"""Fine-tuning and RLCD over HTTP (docs/api.md, "Fine-tuning"): files, jobs, and the models they produce.

Jobs run one at a time on this machine as `selfjev finetune` / `selfjev rlcd` subprocesses, starting from the served
adapter; a finished job's model (`selfjev-4b:ft-<suffix>-<id>`) is served at once as another LoRA on the same weights.
Training needs GPU memory next to serving: a 48 GB card, or a separate box. State lives under `home`
(SELFJEV_HOME, default ~/.selfjev/server): files/ (uploads and their converted rows) and jobs/ (one folder per job).
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import ValidationError as PydanticError

from ..types import DecisionRequest, FileObject, FineTuningJob, FineTuningJobRequest, JobEvent, TrainingRow
from .compat import to_native

MAX_FILE_BYTES = 512 * 2**20


class FineTuningError(Exception):
    def __init__(self, status: int, type: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status, self.type, self.message, self.param = status, type, message, param


def _targets(row: TrainingRow, native: dict) -> list:
    """The expected answers in the internal schema, question by question (see server.compat)."""
    out = []
    for (qid, q), nq in zip(row.questions.items(), native["questions"], strict=True):
        a = row.answers[qid]
        if q.type == "noul":
            out.append(a if nq["type"] == "binary" else ("true" if a else "false"))
        elif q.type == "score":
            out.append(str(a))
        else:
            out.append(list(dict.fromkeys(a)) if isinstance(a, list) else a)
    return out


class Store:
    """Uploaded files: the JSONL as sent, and its rows converted to what `selfjev finetune --data` reads."""

    def __init__(self, home: Path):
        self.dir = home / "files"
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, content: bytes, filename: str, purpose: str) -> FileObject:
        if purpose != "fine-tune":
            raise FineTuningError(422, "invalid_request_error", "purpose must be 'fine-tune'", "purpose")
        if len(content) > MAX_FILE_BYTES:
            raise FineTuningError(413, "invalid_request_error", f"file over {MAX_FILE_BYTES // 2**20} MB", "file")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise FineTuningError(422, "invalid_request_error", "the file is not UTF-8 JSONL", "file") from None
        fid, rows, converted = "file_" + uuid.uuid4().hex[:24], 0, []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = TrainingRow.model_validate_json(line)
                native = to_native(DecisionRequest(state=row.state, questions=row.questions))
            except PydanticError as e:
                err = e.errors()[0]
                where = ".".join(map(str, err["loc"]))
                raise FineTuningError(
                    422, "invalid_request_error", f"line {n}: {where + ': ' if where else ''}{err['msg']}", f"file.line_{n}"
                ) from None
            for q, target in zip(native["questions"], _targets(row, native), strict=True):
                converted.append({"id": f"{fid}-{n}-{q['id']}", "source_id": f"{fid}-{n}", "family": "file", "provenance": filename,
                                  "state": native["state"], "question": q, "target": target})  # fmt: skip
            rows += 1
        if not rows:
            raise FineTuningError(422, "invalid_request_error", "the file has no rows", "file")
        obj = FileObject(
            id=fid, bytes=len(content), created_at=int(time.time()), filename=filename, purpose=purpose, rows=rows, questions=len(converted)
        )
        (self.dir / f"{fid}.jsonl").write_bytes(content)
        (self.dir / f"{fid}.train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in converted))
        (self.dir / f"{fid}.json").write_text(obj.model_dump_json())
        return obj

    def get(self, fid: str) -> FileObject:
        p = self.dir / f"{fid}.json"
        if not fid.startswith("file_") or not p.exists():
            raise FineTuningError(404, "not_found_error", f"no file '{fid}'", "file_id")
        return FileObject.model_validate_json(p.read_text())

    def all(self) -> list[FileObject]:
        return [FileObject.model_validate_json(p.read_text()) for p in sorted(self.dir.glob("file_*.json"))]

    def delete(self, fid: str) -> dict:
        self.get(fid)
        for p in self.dir.glob(f"{fid}*"):
            p.unlink()
        return {"id": fid, "object": "file", "deleted": True}

    def training_path(self, fid: str) -> Path:
        self.get(fid)
        return self.dir / f"{fid}.train.jsonl"


def run_training(job: FineTuningJob, train: Path, val: Path | None, init: str, run_dir: Path, log: Path) -> subprocess.Popen:
    """Start `selfjev finetune|rlcd` for a job; returns the process (its exit code decides the job's status)."""
    hp = job.method.hyperparameters
    cmd = [sys.executable, "-m", "selfjev.cli", "finetune" if job.method.type == "supervised" else "rlcd"]
    cmd += ["--data", str(train), "--out", str(run_dir), "--init", init, "--epochs", str(hp.epochs)]
    cmd += ["--val", str(val)] if val else []
    cmd += ["--lr", str(hp.learning_rate)] if hp.learning_rate else []
    if job.method.type == "rlcd":
        cmd += ["--reward", ",".join(f"{k}={v}" for k, v in hp.reward.items())] if hp.reward else []
        cmd += [x for k in ("samples", "sigma", "beta") if getattr(hp, k) is not None for x in (f"--{k}", str(getattr(hp, k)))]
    with log.open("w") as out:  # the child keeps its own copy of the handle
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


class Jobs:
    """A FIFO of fine-tuning jobs run by one worker thread. `on_model(name, adapter_path)` hears about new models."""

    def __init__(self, home: Path, store: Store, base_model: str, init_adapter: str, on_model: Callable | None = None, runner=None):
        self.dir, self.store, self.base_model, self.init = home / "jobs", store, base_model, init_adapter
        self.dir.mkdir(parents=True, exist_ok=True)
        self.on_model, self.runner = on_model or (lambda name, path: None), runner or run_training
        self.lock, self.wake, self.procs = threading.Lock(), threading.Event(), {}
        for job in self.all():  # a restart forgets processes: requeue what was running, re-announce what succeeded
            if job.status == "running":
                self._save(job.model_copy(update={"status": "queued"}))
            elif job.status == "succeeded":
                self._announce(job)
        threading.Thread(target=self._worker, daemon=True, name="selfjev-fine-tuning").start()

    def _save(self, job: FineTuningJob) -> FineTuningJob:
        d = self.dir / job.id
        d.mkdir(exist_ok=True)
        tmp = d / f".job.{threading.get_ident()}"  # write then rename: readers never see half a file
        tmp.write_text(job.model_dump_json(indent=1))
        tmp.replace(d / "job.json")
        return job

    def _event(self, job_id: str, message: str, level="info", data=None):
        e = JobEvent(created_at=int(time.time()), level=level, message=message, data=data)
        with (self.dir / job_id / "events.jsonl").open("a") as f:
            f.write(e.model_dump_json() + "\n")

    def create(self, req: FineTuningJobRequest) -> FineTuningJob:
        if req.model != self.base_model:
            raise FineTuningError(404, "not_found_error", f"fine-tune '{self.base_model}'; got '{req.model}'", "model")
        self.store.training_path(req.training_file)
        if req.validation_file:
            self.store.training_path(req.validation_file)
        job = FineTuningJob(id="ftjob_" + uuid.uuid4().hex[:20], model=req.model, status="queued", created_at=int(time.time()),
                            training_file=req.training_file, validation_file=req.validation_file, method=req.method)  # fmt: skip
        self._save(job)
        (self.dir / job.id / "request.json").write_text(req.model_dump_json())
        self._event(job.id, f"queued: {req.method.type} on {self.store.get(req.training_file).questions} questions")
        self.wake.set()
        return job

    def get(self, job_id: str) -> FineTuningJob:
        p = self.dir / job_id / "job.json"
        if not job_id.startswith("ftjob_") or not p.exists():
            raise FineTuningError(404, "not_found_error", f"no job '{job_id}'", "job_id")
        return FineTuningJob.model_validate_json(p.read_text())

    def all(self) -> list[FineTuningJob]:
        jobs = [FineTuningJob.model_validate_json(p.read_text()) for p in self.dir.glob("ftjob_*/job.json")]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def events(self, job_id: str) -> list[JobEvent]:
        self.get(job_id)
        p = self.dir / job_id / "events.jsonl"
        out = [JobEvent.model_validate_json(line) for line in p.read_text().splitlines()] if p.exists() else []
        meta = self.dir / job_id / "run" / "train_meta.json"
        if meta.exists():  # validation after every checkpoint, from the trainer
            out += [
                JobEvent(created_at=int(meta.stat().st_mtime), message=f"validation at step {v['step']}", data=v["validation"])
                for v in json.loads(meta.read_text()).get("log", [])
            ]
        return out

    def cancel(self, job_id: str) -> FineTuningJob:
        with self.lock:
            job = self.get(job_id)
            if job.status in ("succeeded", "failed", "cancelled"):
                return job
            if job_id in self.procs:
                os.killpg(self.procs[job_id].pid, signal.SIGTERM)
            job = self._save(job.model_copy(update={"status": "cancelled", "finished_at": int(time.time())}))
        self._event(job_id, "cancelled")
        return job

    def _announce(self, job: FineTuningJob):
        try:
            self.on_model(job.fine_tuned_model, str(self.dir / job.id / "run" / "adapter"))
        except Exception as e:  # a bad adapter must stop neither the server nor the queue
            self._event(job.id, f"could not serve {job.fine_tuned_model}: {e!r}", "error")

    def _worker(self):  # ponytail: the queue is the job files, rescanned; a real queue if jobs ever number in the thousands
        while True:
            queued = sorted((j for j in self.all() if j.status == "queued"), key=lambda j: j.created_at)
            if not queued:
                self.wake.wait(timeout=5)
                self.wake.clear()
                continue
            try:
                self._run(queued[0])
            except Exception as e:  # e.g. the trainer could not start: fail this job, keep the queue going
                self._save(queued[0].model_copy(update={"status": "failed", "error": repr(e), "finished_at": int(time.time())}))
                self._event(queued[0].id, "failed", "error", {"error": repr(e)})

    def _run(self, job: FineTuningJob):
        d = self.dir / job.id
        with self.lock:
            if self.get(job.id).status != "queued":
                return
            val = self.store.training_path(job.validation_file) if job.validation_file else None
            proc = self.runner(job, self.store.training_path(job.training_file), val, self.init, d / "run", d / "train.log")
            self.procs[job.id] = proc
            self._save(job.model_copy(update={"status": "running"}))
        self._event(job.id, "running")
        code = proc.wait()
        with self.lock:
            self.procs.pop(job.id, None)
            if self.get(job.id).status == "cancelled":
                return
            if code == 0 and (d / "run" / "adapter").exists():
                req = FineTuningJobRequest.model_validate_json((d / "request.json").read_text())
                name = f"{job.model}:ft-{req.suffix or 'model'}-{job.id[6:14]}"
                done = self._save(job.model_copy(update={"status": "succeeded", "fine_tuned_model": name, "finished_at": int(time.time())}))
            else:
                tail = (d / "train.log").read_text()[-2000:] if (d / "train.log").exists() else ""
                done = self._save(
                    job.model_copy(update={"status": "failed", "error": tail or f"exit code {code}", "finished_at": int(time.time())})
                )
        if done.status == "succeeded":
            self._event(job.id, f"succeeded: {done.fine_tuned_model}")
            self._announce(done)
        else:
            self._event(job.id, "failed", "error", {"log_tail": done.error[-500:]})


def router(store: Store, jobs: Jobs) -> APIRouter:
    r = APIRouter(tags=["fine-tuning"])

    @r.post("/v1/files", response_model=FileObject)
    async def upload(file: UploadFile = File(...), purpose: str = Form("fine-tune")):
        return store.add(await file.read(), file.filename or "upload.jsonl", purpose)

    @r.get("/v1/files")
    async def list_files():
        return {"object": "list", "data": store.all()}

    @r.get("/v1/files/{file_id}", response_model=FileObject)
    async def get_file(file_id: str):
        return store.get(file_id)

    @r.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str):
        return store.delete(file_id)

    @r.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
    async def create_job(req: FineTuningJobRequest):
        return jobs.create(req)

    @r.get("/v1/fine_tuning/jobs")
    async def list_jobs():
        return {"object": "list", "data": jobs.all()}

    @r.get("/v1/fine_tuning/jobs/{job_id}", response_model=FineTuningJob)
    async def get_job(job_id: str):
        return jobs.get(job_id)

    @r.get("/v1/fine_tuning/jobs/{job_id}/events")
    async def job_events(job_id: str):
        return {"object": "list", "data": jobs.events(job_id)}

    @r.post("/v1/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJob)
    async def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    return r
