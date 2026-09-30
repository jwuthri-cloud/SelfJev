# API

The selfjev API is Jev's decisions API: same endpoint, same request, same answers. Code written for Jev (TypeSafe or
OpenRouter) works by changing the base URL and the key. One extension, `multi` (select all that apply), and fine-tuning
endpoints come on top.

Implemented by `selfjev serve` (server: `src/selfjev/server/`, SDK: `src/selfjev/client.py`). The fine-tuning routes
need `selfjev serve --fine-tuning`.

## With TypeSafe's SDK

TypeSafe's Python SDK (`pip install typesafe-sdk`) works against a selfjev server with two environment variables and no
code change: `TYPESAFE_BASE_URL` (your server) and `TYPESAFE_API_KEY` (a key from `SELFJEV_API_KEYS`). Its default model,
`jev-latest`, is answered by `selfjev-4b`; `client.models.list()` reads the `models` field of `GET /v1/models`. The server
takes Jev's whole question schema: `instructions` may be left out (the criteria carry the question), instructions and
criteria may be JSON (read as JSON text, like `state`), and a `noul` may describe only one side (the other becomes its
negation). `tests/server/test_typesafe_sdk.py` runs the real SDK against the server in CI.

## Endpoints

| method | path | what |
|---|---|---|
| POST | `/v1/systemone` | answer questions about a state (TypeSafe's path) |
| POST | `/api/alpha/decisions`, `/v1/decisions` | the same (OpenRouter's path; `/v1/decisions` is an alias) |
| POST | `/classify` | the internal schema, with raw scores and run metadata ([how it works](how_it_works.md#request-and-response)) |
| GET | `/v1/models` | served models: `selfjev-4b` and fine-tunes |
| GET | `/health` | 200 once the model is loaded |
| GET | `/metrics` | Prometheus metrics |
| POST | `/v1/files` | upload a JSONL training file |
| POST, GET | `/v1/fine_tuning/jobs`, `/v1/fine_tuning/jobs/{id}` | start and follow a fine-tune or RLCD job |
| GET | `/v1/fine_tuning/jobs/{id}/events` | validation metrics as training runs |
| POST | `/v1/fine_tuning/jobs/{id}/cancel` | stop a job |

Authentication: `Authorization: Bearer <key>`. A self-hosted server without configured keys (`SELFJEV_API_KEYS`) accepts
every request.

You create the server key yourself; it does not come from Hugging Face or TypeSafe. Generate a long random value,
for example `python -c 'import secrets; print(secrets.token_urlsafe(32))'`, set it as `SELFJEV_API_KEYS` on the server,
then give the same value to the client through `api_key=` or `SELFJEV_API_KEY`. The server accepts any key in its
comma-separated list. `selfjev deploy aws up` generates a key if `--api-key` is omitted, prints it once, and saves it
in `~/.selfjev/deployments/<name>.json`.

## Request

```json
{
  "model": "selfjev-4b",
  "state": "Ticket 4411: the invoice was charged twice and the customer wants the second charge back.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "billing and refunds", "tech": "outages and bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["not urgent", "this week", "today"]},
    "topics": {"type": "multi", "instructions": "Which topics does it mention?",
               "criteria": {"invoice": "an invoice", "shipping": "a delivery", "account": "account access"}}
  }
}
```

- `state`: a string, or any JSON object or array (serialized as JSON before reading). State plus the longest question:
  at most 32,768 tokens (the model is trained on up to 16K; longer is accepted, not validated).
- Images (an extension; Jev takes none): `state` may be an image as a base64 data URL (`"data:image/jpeg;base64,…"`), or
  an array of parts mixing text and images, read in order. The image is encoded once by Qwen3.5's own vision tower
  (resized to at most 1,024 × 1,024 pixels, one token per 32 × 32) and every question is answered against it, as with
  text. Default engine only (not `--engine vllm`). The SDK turns a `Path`, image bytes or a PIL image in `state` into a
  data URL: `client.system_one(state=[Path("cat.jpg")], questions={"breed": Choice("What breed is it?", {...})})`.
  `selfjev-4b` was never trained on images: accuracy is in [experiments.md](experiments.md).
- `questions`: up to 64, keyed by your ids. Each is answered in isolation against the same state; the state is read once.

| type | criteria | answers |
|---|---|---|
| `noul` | optional `{"true": …, "false": …}` descriptions | the probability of yes (or of `true`) |
| `choice` | `{key: description or null}`, 2 to 255 options | one option |
| `score` | `[level_0, …, level_k]`, 2 to 10 ordered levels | a position on the scale |
| `multi` (extension) | `{key: description or null}`, 1 to 255 options | every option that applies, possibly none |

## Response

```json
{
  "id": "dec_01J9Z3…",
  "model": "selfjev-4b",
  "answers": {
    "refund": {"type": "noul", "noul": 0.97},
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.99, "tech": 0.01}, "confidence": 0.98},
    "urgency": {"type": "score", "score": 1.43, "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
                "legend": {"0": "not urgent", "1": "this week", "2": "today"}, "confidence": 0.36},
    "topics": {"type": "multi", "multi": ["invoice"], "probabilities": {"invoice": 0.98, "shipping": 0.02, "account": 0.04}}
  },
  "usage": {"input_tokens": 296, "output_tokens": 0}
}
```

- `noul`: P(yes). No confidence (as in Jev).
- `choice`: the most probable option (ties broken by key), the distribution, and `confidence = (K · p_max − 1) / (K − 1)`:
  1 when all probability is on one option, 0 when it is uniform.
- `score`: `score = Σ i · p_i` in level units (0 to k), `probabilities` and `legend` keyed by the level index as a string,
  `confidence` as for `choice` over the levels.
- `multi`: every option with P ≥ 0.5, and each option's own probability (they do not sum to 1).
- `model` names the model that answered: `selfjev-4b`, or a fine-tune. Jev's names (`jev-latest`,
  `typesafe/jev-latest`, `~typesafe/jev-latest`) are accepted and answered by `selfjev-4b`. `output_tokens` is always
  0: nothing is generated.

## Errors

`{"error": {"type": "…", "message": "…", "param": "questions.team.criteria"}}`

| status | type | when |
|---|---|---|
| 401 | `authentication_error` | missing or unknown key |
| 404 | `not_found_error` | unknown model, file or job |
| 413 | `invalid_request_error` | an uploaded file over 512 MB |
| 422 | `invalid_request_error` | schema violations, too few options, input over the token limit, a bad training line |
| 429 | `rate_limit_error` | from Jev or a proxy in front (selfjev queues instead); `Retry-After` says when to retry |
| 529 | `overloaded_error` | the request queue is full; retry with backoff |
| 500 | `api_error` | a bug: please report it with the request id (in the message and the `x-request-id` header) |

## Fine-tuning

The shape of OpenAI's fine-tuning API, on a server started with `selfjev serve --fine-tuning` (keeps the LoRA unmerged so
fine-tunes can be served next to `selfjev-4b`; training shares the GPU, so use a 48 GB card such as the L40S, or train on
another box with `selfjev finetune`). Jobs run one at a time; state lives in `--home` (`SELFJEV_HOME`, default
`~/.selfjev/server`) and survives restarts.

**Training file:** JSONL, one decisions request plus the expected answers per line, so logged traffic becomes training
data:

```json
{"state": "…", "questions": {"team": {"type": "choice", "instructions": "…", "criteria": {"billing": "…", "tech": "…"}}},
 "answers": {"team": "billing"}}
```

`answers` covers every question: `true`/`false` for `noul`, a key for `choice`, a level index for `score` and a list of
keys for `multi`. Upload it as multipart form data (`file`, `purpose=fine-tune`): `POST /v1/files` checks every line
and names the first bad one (`param: "file.line_12"`). `GET /v1/files`, `GET` and `DELETE /v1/files/{id}` manage uploads.

**Images:** `state` may be an image data URL or a list of text and image parts, exactly as in a request; the vision tower
stays frozen and the language LoRA learns from it. Files hold the images inline (base64), limit 512 MB per file (about
4,000 photos at 512 px). Ready-made rows from Hugging Face datasets: [image datasets](image_datasets.md).

**Job:**

```json
POST /v1/fine_tuning/jobs
{"model": "selfjev-4b", "training_file": "file_…", "validation_file": "file_…", "suffix": "support",
 "method": {"type": "rlcd", "hyperparameters": {"epochs": 1, "reward": {"log": 1, "brier": 1, "spherical": 1, "confident_miss": 2}}}}
```

- `method.type`: `supervised` (cross-entropy on the answers) or `rlcd` (proper-scoring-rule rewards, optionally a cost
  per confident mistake). See [fine-tune and RLCD](finetune.md) for what each does and does not buy.
- `hyperparameters`: `epochs` (1 to 10), `learning_rate` (default 2e-4 supervised, 5e-5 RLCD); RLCD only: `reward`
  (weights of `log`, `brier`, `spherical`, `accuracy`, `confident_miss`), `samples`, `sigma`, `beta`.
- Training starts from the served `selfjev-4b` adapter. Without a `validation_file`, 5% of the training file (at most
  1,000 questions) is held out.
- `status`: `queued`, `running`, `succeeded`, `failed` (with `error`: the end of the training log) or `cancelled`.
  `GET /v1/fine_tuning/jobs/{id}/events` lists progress and the validation metrics of every checkpoint.
- A job that succeeds names its model, `selfjev-4b:ft-support-<id>`, in `fine_tuned_model`. It is served at once:
  pass it as `model`.

With the SDK:

```python
from selfjev import SelfJev

client = SelfJev(base_url="http://localhost:8000")
f = client.upload_file("train.jsonl")
job = client.create_fine_tuning_job(f.id, method="rlcd", suffix="support")
job = client.fine_tuning_job(job.id)          # poll until job.status == "succeeded"
client.system_one(state, questions, model=job.fine_tuned_model)
```
