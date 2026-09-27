# API

The selfjev API is Jev's decisions API: same endpoint, same request, same answers. Code written for Jev (TypeSafe or
OpenRouter) works by changing the base URL and the key. One extension, `multi` (select all that apply), and fine-tuning
endpoints come on top.

Status: specification for the server being built (`selfjev serve`); `selfjev serve` today implements
`/api/alpha/decisions` with `noul`, `choice` and `score`.

## Endpoints

| method | path | what |
|---|---|---|
| POST | `/v1/systemone` | answer questions about a state (TypeSafe's path) |
| POST | `/api/alpha/decisions` | the same (OpenRouter's path) |
| GET | `/v1/models` | served models: `selfjev-4b` and fine-tunes |
| GET | `/health` | 200 once the model is loaded |
| GET | `/metrics` | Prometheus metrics |
| POST | `/v1/files` | upload a JSONL training file |
| POST, GET | `/v1/fine_tuning/jobs`, `/v1/fine_tuning/jobs/{id}` | start and follow a fine-tune or RLCD job |
| GET | `/v1/fine_tuning/jobs/{id}/events` | validation metrics as training runs |
| POST | `/v1/fine_tuning/jobs/{id}/cancel` | stop a job |

Authentication: `Authorization: Bearer <key>`. A self-hosted server without configured keys (`SELFJEV_API_KEYS`) accepts
every request.

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
  "model": "selfjev-4b-2026-09-26",
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
- `model` names the exact version that answered. `output_tokens` is always 0: nothing is generated.

## Errors

`{"error": {"type": "…", "message": "…", "param": "questions.team.criteria"}}`

| status | type | when |
|---|---|---|
| 401 | `authentication_error` | missing or unknown key |
| 404 | `not_found_error` | unknown model, file or job |
| 422 | `invalid_request_error` | schema violations, duplicate ids, too few options, input over the token limit |
| 429 | `rate_limit_error` | over the configured rate; `Retry-After` says when to retry |
| 529 | `overloaded_error` | the request queue is full; retry with backoff |
| 500 | `api_error` | a bug: please report it with the `x-request-id` header |

## Fine-tuning

The shape of OpenAI's fine-tuning API. Training rows are decisions requests plus the expected answers, so logged traffic
becomes training data:

```json
{"state": "…", "questions": {"team": {"type": "choice", "instructions": "…", "criteria": {"billing": "…", "tech": "…"}}},
 "answers": {"team": "billing"}}
```

`answers` gives `true`/`false` for `noul`, a key for `choice`, a level index for `score` and a list of keys for `multi`.

```json
POST /v1/fine_tuning/jobs
{"model": "selfjev-4b", "training_file": "file_…", "validation_file": "file_…", "suffix": "support",
 "method": {"type": "supervised", "hyperparameters": {"epochs": 1, "learning_rate": 2e-5}}}
```

- `method.type`: `supervised` (cross-entropy on the targets) or `rlcd` (proper-scoring-rule rewards, optionally a cost per
  confident mistake: `{"reward": {"log": 1, "brier": 1, "spherical": 1, "confident_miss": 0}}`). See
  [fine-tune and RLCD](finetune.md) for what each does and does not buy.
- The job's result is a model, `selfjev-4b:ft-support-<id>`, usable at once in `model`.
