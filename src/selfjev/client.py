"""The selfjev SDK: a client for the decisions API, in Jev's shape (docs/api.md). Works against a selfjev server and
against Jev itself (TypeSafe's /v1/systemone, or OpenRouter with path="/api/alpha/decisions").

    from selfjev import SelfJev, Choice, Noul

    client = SelfJev(base_url="http://localhost:8000")           # or SELFJEV_BASE_URL / SELFJEV_API_KEY
    res = client.system_one(
        state="Ticket 4411: the invoice was charged twice.",
        questions={"refund": Noul("Does the customer ask for a refund?"),
                   "team": Choice("Which team?", {"billing": "billing and refunds", "tech": "outages and bugs"})},
    )
    res.choices["team"].choice, res.nouls["refund"].noul

Images (the default tree engine): `state=[Path("cat.jpg")]`, or a PIL image or image bytes, alone or next to text parts.

Fine-tuning (a selfjev server started with --fine-tuning; docs/api.md "Fine-tuning"):

    f = client.upload_file("train.jsonl")                         # one decisions request + its answers per line
    job = client.create_fine_tuning_job(f.id, method="rlcd", suffix="acme")
    client.fine_tuning_job(job.id).fine_tuned_model               # then system_one(..., model=that name)

Needs only httpx and pydantic. Retries 429, 529, 5xx and connection errors with exponential backoff (Retry-After wins).
"""

import asyncio
import base64
import io
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import TypeAdapter

from .types import (
    DecisionRequest,
    DecisionResponse,
    ErrorResponse,
    FileObject,
    FineTuningJob,
    FineTuningJobRequest,
    JobEvent,
    Question,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PATH = "/v1/systemone"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
_QUESTIONS = TypeAdapter(dict[str, Question])
_JOBS, _EVENTS = TypeAdapter(list[FineTuningJob]), TypeAdapter(list[JobEvent])
JOBS = "/v1/fine_tuning/jobs"


class SelfJevError(Exception):
    """Base class of every SDK error."""


class APIConnectionError(SelfJevError):
    """The server could not be reached (after retries)."""


class APIError(SelfJevError):
    """The server answered with an error status. `type` and `message` come from its JSON error body."""

    def __init__(self, status: int, message: str, type: str = "api_error", param: str | None = None, request_id: str | None = None):
        super().__init__(f"{status} {type}: {message}")
        self.status, self.message, self.type, self.param, self.request_id = status, message, type, param, request_id


class AuthenticationError(APIError):
    pass


class NotFoundError(APIError):
    pass


class InvalidRequestError(APIError):
    pass


class RateLimitError(APIError):
    pass


class OverloadedError(APIError):
    pass


_BY_STATUS = {401: AuthenticationError, 403: AuthenticationError, 404: NotFoundError, 400: InvalidRequestError,
              422: InvalidRequestError, 429: RateLimitError, 529: OverloadedError}  # fmt: skip


def _error(r: httpx.Response) -> APIError:
    try:
        e = ErrorResponse.model_validate(r.json()).error
        message, kind, param = e.message, e.type, e.param
    except Exception:  # not our JSON error body: keep the raw text
        message, kind, param = r.text[:500] or r.reason_phrase, "api_error", None
    return _BY_STATUS.get(r.status_code, APIError)(r.status_code, message, kind, param, r.headers.get("x-request-id"))


def _delay(attempt: int, r: httpx.Response | None) -> float:
    after = r.headers.get("retry-after") if r is not None else None
    if after:
        try:
            return min(float(after), 60.0)
        except ValueError:
            pass
    return min(8.0, 0.5 * 2**attempt) * (0.5 + random.random() / 2)


_MAGIC = {b"\x89PNG": "png", b"\xff\xd8\xff": "jpeg", b"GIF8": "gif", b"RIFF": "webp"}


def _part(p):
    """An image in `state` (a PIL image, image file bytes, or a Path to an image file) -> a base64 data URL."""
    if isinstance(p, Path):
        p = p.read_bytes()
    if hasattr(p, "save"):  # PIL.Image
        buf = io.BytesIO()
        p.save(buf, format="PNG")
        p = buf.getvalue()
    if not isinstance(p, bytes):
        return p
    kind = next((k for m, k in _MAGIC.items() if p.startswith(m)), None)
    if kind is None:
        raise ValueError("state: bytes must be a PNG, JPEG, GIF or WebP image")
    return f"data:image/{kind};base64," + base64.b64encode(p).decode()


class _Base:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0, max_retries=2, path=DEFAULT_PATH):
        self.api_key = api_key if api_key is not None else os.environ.get("SELFJEV_API_KEY")
        self.base_url = (base_url or os.environ.get("SELFJEV_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("SELFJEV_DEFAULT_MODEL") or "selfjev-4b"
        self.timeout, self.max_retries, self.path = timeout, max_retries, path

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": "selfjev-python"}  # httpx sets Content-Type: JSON bodies and multipart uploads differ
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, state, questions, model, extra_body) -> dict:
        state = [_part(p) for p in state] if isinstance(state, list) else _part(state)
        req = DecisionRequest(model=model or self.model, state=state, questions=_QUESTIONS.validate_python(questions))
        return req.model_dump(mode="json", exclude_none=True) | (extra_body or {})

    @staticmethod
    def _upload(path, purpose) -> dict:  # bytes, not a file handle, so a retry resends the whole file
        return {"files": {"file": (Path(path).name, Path(path).read_bytes())}, "data": {"purpose": purpose}}

    @staticmethod
    def _job(training_file, method, hyperparameters, validation_file, suffix, model) -> dict:
        req = FineTuningJobRequest.model_validate(
            {"training_file": training_file, "validation_file": validation_file, "suffix": suffix,
             "method": {"type": method, "hyperparameters": hyperparameters or {}}} | ({"model": model} if model else {})
        )  # fmt: skip
        return {"json": req.model_dump(mode="json", exclude_none=True)}


class SelfJev(_Base):
    """Synchronous client. Reuse one instance: it keeps a connection pool."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0, max_retries=2, path=DEFAULT_PATH, http_client=None):
        super().__init__(api_key, base_url, model, timeout, max_retries, path)
        self._http = http_client or httpx.Client(timeout=timeout)

    def system_one(self, state, questions: dict[str, Any], model: str | None = None, extra_body: dict | None = None) -> DecisionResponse:
        """Answer every question about `state`. Questions: Noul / Choice / Score / Multi objects or plain dicts."""
        return DecisionResponse.model_validate(self._post(self.path, self._body(state, questions, model, extra_body)))

    def models(self) -> list[dict]:
        return self._request("GET", "/v1/models").json()["data"]

    def upload_file(self, path: str | Path, purpose: str = "fine-tune") -> FileObject:
        """Upload a fine-tuning file: JSONL, one {"state", "questions", "answers"} object per line."""
        return FileObject.model_validate(self._request("POST", "/v1/files", **self._upload(path, purpose)).json())

    def create_fine_tuning_job(
        self,
        training_file: str,
        method: Literal["supervised", "rlcd"] = "supervised",
        hyperparameters: dict | None = None,
        validation_file: str | None = None,
        suffix: str | None = None,
        model: str | None = None,
    ) -> FineTuningJob:
        """Queue a fine-tune (supervised) or RLCD job; its model is served under job.fine_tuned_model once it succeeds."""
        body = self._job(training_file, method, hyperparameters, validation_file, suffix, model)
        return FineTuningJob.model_validate(self._request("POST", JOBS, **body).json())

    def fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return FineTuningJob.model_validate(self._request("GET", f"{JOBS}/{job_id}").json())

    def fine_tuning_jobs(self) -> list[FineTuningJob]:
        return _JOBS.validate_python(self._request("GET", JOBS).json()["data"])

    def fine_tuning_events(self, job_id: str) -> list[JobEvent]:
        return _EVENTS.validate_python(self._request("GET", f"{JOBS}/{job_id}/events").json()["data"])

    def cancel_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return FineTuningJob.model_validate(self._request("POST", f"{JOBS}/{job_id}/cancel").json())

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, json=body).json()

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.request(method, self.base_url + path, headers=self._headers(), **kw)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise APIConnectionError(f"{method} {self.base_url + path}: {e}") from e
                time.sleep(_delay(attempt, None))
                continue
            if r.status_code < 400:
                return r
            if r.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                raise _error(r)
            time.sleep(_delay(attempt, r))
        raise AssertionError("unreachable")

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncSelfJev(_Base):
    """Asynchronous client: `await client.system_one(...)`."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0, max_retries=2, path=DEFAULT_PATH, http_client=None):
        super().__init__(api_key, base_url, model, timeout, max_retries, path)
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def system_one(
        self, state, questions: dict[str, Any], model: str | None = None, extra_body: dict | None = None
    ) -> DecisionResponse:
        body = self._body(state, questions, model, extra_body)
        return DecisionResponse.model_validate((await self._request("POST", self.path, json=body)).json())

    async def models(self) -> list[dict]:
        return (await self._request("GET", "/v1/models")).json()["data"]

    async def upload_file(self, path: str | Path, purpose: str = "fine-tune") -> FileObject:
        return FileObject.model_validate((await self._request("POST", "/v1/files", **self._upload(path, purpose))).json())

    async def create_fine_tuning_job(
        self,
        training_file: str,
        method: Literal["supervised", "rlcd"] = "supervised",
        hyperparameters: dict | None = None,
        validation_file: str | None = None,
        suffix: str | None = None,
        model: str | None = None,
    ) -> FineTuningJob:
        body = self._job(training_file, method, hyperparameters, validation_file, suffix, model)
        return FineTuningJob.model_validate((await self._request("POST", JOBS, **body)).json())

    async def fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return FineTuningJob.model_validate((await self._request("GET", f"{JOBS}/{job_id}")).json())

    async def fine_tuning_jobs(self) -> list[FineTuningJob]:
        return _JOBS.validate_python((await self._request("GET", JOBS)).json()["data"])

    async def fine_tuning_events(self, job_id: str) -> list[JobEvent]:
        return _EVENTS.validate_python((await self._request("GET", f"{JOBS}/{job_id}/events")).json()["data"])

    async def cancel_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return FineTuningJob.model_validate((await self._request("POST", f"{JOBS}/{job_id}/cancel")).json())

    async def _request(self, method: str, path: str, **kw) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http.request(method, self.base_url + path, headers=self._headers(), **kw)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise APIConnectionError(f"{method} {self.base_url + path}: {e}") from e
                await asyncio.sleep(_delay(attempt, None))
                continue
            if r.status_code < 400:
                return r
            if r.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                raise _error(r)
            await asyncio.sleep(_delay(attempt, r))
        raise AssertionError("unreachable")

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()
