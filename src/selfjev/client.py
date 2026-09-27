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

Needs only httpx and pydantic. Retries 429, 529, 5xx and connection errors with exponential backoff (Retry-After wins).
"""

import asyncio
import os
import random
import time
from typing import Any

import httpx
from pydantic import TypeAdapter

from .types import DecisionRequest, DecisionResponse, ErrorResponse, Question

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PATH = "/v1/systemone"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
_QUESTIONS = TypeAdapter(dict[str, Question])


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


class _Base:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0, max_retries=2, path=DEFAULT_PATH):
        self.api_key = api_key if api_key is not None else os.environ.get("SELFJEV_API_KEY")
        self.base_url = (base_url or os.environ.get("SELFJEV_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("SELFJEV_DEFAULT_MODEL") or "selfjev-4b"
        self.timeout, self.max_retries, self.path = timeout, max_retries, path

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "selfjev-python"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, state, questions, model, extra_body) -> dict:
        req = DecisionRequest(model=model or self.model, state=state, questions=_QUESTIONS.validate_python(questions))
        return req.model_dump(mode="json", exclude_none=True) | (extra_body or {})


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
