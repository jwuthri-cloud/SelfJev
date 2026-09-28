"""The HTTP server: Jev's decisions API (docs/api.md) on a selfjev.engine scorer.

  POST /v1/systemone, /api/alpha/decisions, /v1/decisions   Jev's request and answers, plus `multi`
  POST /classify                                            the internal schema (examples/request.json)
  GET  /v1/models, /health, /metrics
  /v1/files, /v1/fine_tuning/jobs[...]                      fine-tuning and RLCD (with fine_tuning_home; server.finetuning)

Concurrent requests are batched into shared forward passes (selfjev.server.batching). Authentication: Bearer keys from
SELFJEV_API_KEYS (comma-separated) or `api_keys`; none configured means open, for local self-hosting.
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from ..core.answers import classify_many, run_meta
from ..core.options import with_option_lists
from ..core.schemas import InputTooLong, ValidationError, parse_request
from ..types import DecisionRequest, DecisionResponse, Usage
from .batching import Batcher, Overloaded
from .compat import to_answers, to_native
from .finetuning import FineTuningError, Jobs, Store, router
from .metrics import Metrics

JEV_ALIASES = frozenset({"jev-latest", "typesafe/jev-latest", "~typesafe/jev-latest"})  # Jev clients keep their model name
OPEN_PATHS = frozenset({"/health", "/metrics"})
log = logging.getLogger("selfjev.server")


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
    fine_tuning_home: Path | None = None,
    init_adapter: str | None = None,
    warmup: bool = False,
) -> FastAPI:
    """scorer: TreeServer or VllmScorer (anything with score_requests and meta). fine_tuning_home turns on
    the fine-tuning routes; their models need a scorer that can load adapters (TreeServer(merge=False))."""
    keys = set(api_keys if api_keys is not None else filter(None, os.environ.get("SELFJEV_API_KEYS", "").split(",")))
    bearer = {f"Bearer {k}" for k in keys}
    metrics = Metrics()

    served, model_lock = {model_name: "default"}, threading.Lock()  # model id -> adapter name

    def run(items):
        """items: (adapter, request) pairs; each adapter's requests share one scorer call."""
        out = [None] * len(items)
        with model_lock:
            for adapter in dict.fromkeys(a for a, _ in items):
                idx = [i for i, (a, _) in enumerate(items) if a == adapter]
                if hasattr(scorer, "set_adapter"):
                    scorer.set_adapter(adapter)
                results, stats = classify_many(scorer, [items[i][1] for i in idx], calibration)
                for i, res, tok in zip(idx, results, stats.get("tokens_per_request") or [0] * len(idx), strict=True):
                    out[i] = (res, tok)
            if hasattr(scorer, "set_adapter"):
                scorer.set_adapter("default")
        return out

    def register(name: str, path: str):  # a fine-tuning job finished: serve its adapter next to the base one
        with model_lock:
            scorer.load_adapter(name, path)
        served[name] = name

    batcher = Batcher(run, max_batch_requests, max_wait_ms, max_queue)

    def warm_up():  # the first calls compile GPU kernels (39 s cold on an L40S): pay that before /health answers
        q = [
            {"id": "a", "type": "binary", "instruction": "Is this a test?"},
            {
                "id": "b",
                "type": "multiclass",
                "instruction": "Which?",
                "candidates": [{"id": "x", "description": "x"}, {"id": "y", "description": "y"}],
            },
        ]
        for n in (1, 16):
            run([("default", prepare({"state": f"Warm-up text {i}.", "questions": q})) for i in range(n)])

    @asynccontextmanager
    async def lifespan(_):
        if warmup:
            t0 = time.perf_counter()
            await asyncio.to_thread(warm_up)
            log.info("warmed up in %.1f s", time.perf_counter() - t0)
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(
        title="selfjev", summary="Jev's decisions API on selfjev-4b", version=__import__("selfjev").__version__, lifespan=lifespan
    )

    def prepare(native: dict):  # adapters trained with every option listed in the question need the same transform
        return parse_request(with_option_lists(native) if options_in_question else native)

    @app.middleware("http")
    async def request_id_auth_metrics(request: Request, call_next):
        rid, t0 = "req_" + uuid.uuid4().hex[:20], time.perf_counter()
        if bearer and request.url.path not in OPEN_PATHS and request.headers.get("authorization") not in bearer:
            response = _error(401, "authentication_error", "missing or unknown API key")
        else:
            try:
                response = await call_next(request)
            except Exception:  # a bug: JSON like every other error, with the id that finds it in the log
                log.exception("request %s failed", rid)
                response = _error(500, "api_error", f"internal error; report it with request id {rid}")
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

    @app.exception_handler(FineTuningError)
    async def _fine_tuning(_, e: FineTuningError):
        return _error(e.status, e.type, e.message, e.param)

    @app.exception_handler(Overloaded)
    async def _overloaded(_, e: Overloaded):
        return _error(529, "overloaded_error", f"server busy ({e}); retry with backoff", headers={"retry-after": "1"})

    async def decisions(body: DecisionRequest) -> DecisionResponse:
        model = model_name if body.model in JEV_ALIASES else body.model
        if model not in served:
            raise APIException(404, "not_found_error", f"unknown model '{body.model}'; this server has {sorted(served)}", "model")
        results, tokens = await batcher.submit((served[model], prepare(to_native(body))))
        metrics.add_usage(tokens, len(body.questions))
        return DecisionResponse(
            id="dec_" + uuid.uuid4().hex[:24], model=model, answers=to_answers(body, results), usage=Usage(input_tokens=tokens)
        )

    for path in ("/v1/systemone", "/api/alpha/decisions", "/v1/decisions"):
        app.post(path, response_model=DecisionResponse, response_model_exclude_none=True, tags=["decisions"])(decisions)

    @app.post("/classify", tags=["internal"])
    async def classify(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise APIException(422, "invalid_request_error", "body must be a JSON object")
        results, tokens = await batcher.submit(("default", prepare(body)))
        return {"questions": results, "meta": run_meta(scorer, calibration) | {"input_tokens": tokens}}

    @app.get("/v1/models", tags=["meta"])
    async def models():
        keep = ("model", "revision", "adapter", "adapter_sha256", "prompt", "max_length")
        meta = {k: scorer.meta.get(k) for k in keep}
        data = [{"id": m, "object": "model", "owned_by": "selfjev", "meta": meta} for m in served]
        # `models` is TypeSafe's shape (their SDK's models.list()); `data` is OpenAI's, read by selfjev's SDK.
        jev = [{"name": m, "description": f"selfjev: {m}, self-hosted", "release_date": ""} for m in [*served, *sorted(JEV_ALIASES)]]
        return {"object": "list", "data": data, "models": jev}

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "model": model_name, "queue": batcher.depth}

    @app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
    async def prometheus():
        return metrics.render(batcher.depth)

    if fine_tuning_home is not None:
        store = Store(Path(fine_tuning_home))
        jobs = Jobs(Path(fine_tuning_home), store, model_name, init_adapter or scorer.meta.get("adapter"), register)
        app.include_router(router(store, jobs))
    return app


def serve(scorer, host="127.0.0.1", port=8000, calibration=None, options_in_question=True, **kw):
    import uvicorn

    uvicorn.run(create_app(scorer, calibration, options_in_question, warmup=True, **kw), host=host, port=port)
