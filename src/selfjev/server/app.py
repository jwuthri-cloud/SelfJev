"""The HTTP server: Jev's decisions API (docs/api.md) on a selfjev.engine scorer.

  POST /v1/systemone, /api/alpha/decisions, /v1/decisions   Jev's request and answers, plus `multi`
  POST /classify                                            the internal schema (examples/request.json)
  GET  /v1/models, /health, /metrics

Concurrent requests are batched into shared forward passes (selfjev.server.batching). Authentication: Bearer keys from
SELFJEV_API_KEYS (comma-separated) or `api_keys`; none configured means open, for local self-hosting.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from ..core.answers import classify_many, run_meta
from ..core.options import with_options
from ..core.schemas import InputTooLong, ValidationError, parse_request
from ..types import DecisionRequest, DecisionResponse, Usage
from .batching import Batcher, Overloaded
from .compat import to_answers, to_native
from .metrics import Metrics

JEV_ALIASES = frozenset({"jev-latest", "typesafe/jev-latest", "~typesafe/jev-latest"})  # Jev clients keep their model name
OPEN_PATHS = frozenset({"/health", "/metrics"})


class APIException(Exception):
    def __init__(self, status: int, type: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status, self.type, self.message, self.param = status, type, message, param


def _error(status: int, type: str, message: str, param: str | None = None, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"type": type, "message": message, "param": param}}, status, headers=headers)


def create_app(
    scorer,
    calibration=None,
    options_in_question=True,
    model_name="selfjev-4b",
    api_keys=None,
    max_batch_requests=32,
    max_wait_ms=5.0,
    max_queue=256,
) -> FastAPI:
    """scorer: TreeServer, Qwen35Scorer or VllmScorer (anything with score_requests and meta)."""
    keys = set(api_keys if api_keys is not None else filter(None, os.environ.get("SELFJEV_API_KEYS", "").split(",")))
    bearer = {f"Bearer {k}" for k in keys}
    metrics = Metrics()

    def run(requests):
        results, stats = classify_many(scorer, requests, calibration)
        return list(zip(results, stats.get("tokens_per_request") or [0] * len(results), strict=True))

    batcher = Batcher(run, max_batch_requests, max_wait_ms, max_queue)

    @asynccontextmanager
    async def lifespan(_):
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(
        title="selfjev", summary="Jev's decisions API on selfjev-4b", version=__import__("selfjev").__version__, lifespan=lifespan
    )

    def prepare(native: dict):
        if options_in_question:  # adapters trained with every option listed in the question need the same transform
            native = native | {
                "questions": [with_options(q, str(q.get("id"))) if isinstance(q, dict) else q for q in native.get("questions", [])]
            }
        return parse_request(native)

    @app.middleware("http")
    async def request_id_auth_metrics(request: Request, call_next):
        rid, t0 = "req_" + uuid.uuid4().hex[:20], time.perf_counter()
        if bearer and request.url.path not in OPEN_PATHS and request.headers.get("authorization") not in bearer:
            response = _error(401, "authentication_error", "missing or unknown API key")
        else:
            response = await call_next(request)
        response.headers["x-request-id"] = rid
        metrics.observe(request.url.path, response.status_code, time.perf_counter() - t0)
        return response

    @app.exception_handler(APIException)
    async def _api(_, e: APIException):
        return _error(e.status, e.type, e.message, e.param)

    @app.exception_handler(RequestValidationError)
    async def _schema(_, e: RequestValidationError):
        first = e.errors()[0]
        loc = [str(x) for x in first["loc"][1:] if x not in ("noul", "choice", "score", "multi")]  # drop "body" and the union tag
        return _error(422, "invalid_request_error", first["msg"], ".".join(loc) or None)

    @app.exception_handler(ValidationError)
    async def _internal_schema(_, e: ValidationError):
        return _error(422, "invalid_request_error", str(e))

    @app.exception_handler(InputTooLong)
    async def _too_long(_, e: InputTooLong):
        return _error(422, "invalid_request_error", str(e), "state")

    @app.exception_handler(Overloaded)
    async def _overloaded(_, e: Overloaded):
        return _error(529, "overloaded_error", f"server busy ({e}); retry with backoff", headers={"retry-after": "1"})

    async def decisions(body: DecisionRequest) -> DecisionResponse:
        if body.model != model_name and body.model not in JEV_ALIASES:
            raise APIException(404, "not_found_error", f"unknown model '{body.model}'; this server has '{model_name}'", "model")
        results, tokens = await batcher.submit(prepare(to_native(body)))
        metrics.add_usage(tokens, len(body.questions))
        return DecisionResponse(
            id="dec_" + uuid.uuid4().hex[:24], model=model_name, answers=to_answers(body, results), usage=Usage(input_tokens=tokens)
        )

    for path in ("/v1/systemone", "/api/alpha/decisions", "/v1/decisions"):
        app.post(path, response_model=DecisionResponse, response_model_exclude_none=True, tags=["decisions"])(decisions)

    @app.post("/classify", tags=["internal"])
    async def classify(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise APIException(422, "invalid_request_error", "body must be a JSON object")
        results, tokens = await batcher.submit(prepare(body))
        return {"questions": results, "meta": run_meta(scorer, calibration) | {"input_tokens": tokens}}

    @app.get("/v1/models", tags=["meta"])
    async def models():
        keep = ("model", "revision", "adapter", "adapter_sha256", "prompt", "max_length")
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "selfjev", "meta": {k: scorer.meta.get(k) for k in keep}}],
        }

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "model": model_name, "queue": batcher.depth}

    @app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
    async def prometheus():
        return metrics.render(batcher.depth)

    return app


def serve(scorer, host="127.0.0.1", port=8000, calibration=None, options_in_question=True, **kw):
    import uvicorn

    uvicorn.run(create_app(scorer, calibration, options_in_question, **kw), host=host, port=port)
