# SelfJev

**SelfJev turns text, images and questions into decisions your code can use.** Route a request, check an AI response, or apply a policy. Get typed answers and probabilities back from selfjev-4b, an open 4B decisions model with Jev's API.

**SelfJev is self-hosted.** There is no SelfJev cloud: you run the model server on your own GPU, and this package is the client for it (plus the server itself, as an extra). Already on Jev? Keep TypeSafe's SDK and change two environment variables.

| You need | API type | What comes back |
|---|---|---|
| Does this need a refund? | `Noul` | Probability of yes |
| Which team should handle it? | `Choice` | One choice and probabilities for every option |
| How urgent is it? | `Score` | A position on your ordered scale |
| Which topics are mentioned? | `Multi` | Every selected option and its probability |

## 1. Run your server

On a Linux machine with an NVIDIA GPU (24 GB is a practical start; see the [hardware guide](https://github.com/Jwuthri/SelfJev/blob/master/website/content/hardware.md)):

```bash
pip install "selfjev[serve,gpu]"
export SELFJEV_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"   # your key, you choose it
selfjev serve --host 0.0.0.0 --port 8000
```

The first start downloads the [selfjev-4b vision adapter](https://huggingface.co/Jwuthrich/selfjev-4b-vision) (230 MB; text and images) and its Qwen3.5-4B base from Hugging Face. `--adapter <dir or repo>` serves your own fine-tune. Docker, AWS, Runpod and GCP recipes are in the [deployment guide](https://github.com/Jwuthri/SelfJev/blob/master/docs/deploy.md).

| Extra | For |
|---|---|
| `selfjev` | the client only: httpx + pydantic, no torch |
| `selfjev[serve]` | the model server (add `gpu` on Linux for the fast kernels) |
| `selfjev[serve,quant]` | `--quantize 4bit` / `--quantized-model` for an 8 GB GPU |
| `selfjev[ollama]` | `selfjev serve --engine ollama` on the GGUF release, no torch |
| `selfjev[train]` | `selfjev finetune` and `selfjev rlcd` |
| `selfjev[deploy]` | `selfjev deploy aws` |

## 2a. Already using Jev? Change two variables

Code written for TypeSafe's `typesafe-sdk` runs unchanged against your server:

```bash
export TYPESAFE_BASE_URL="https://your-selfjev-host:8000"
export TYPESAFE_API_KEY="the key you set in SELFJEV_API_KEYS"
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient()   # reads the two variables above
res = client.system_one(
    state="I was charged twice. Please refund the duplicate payment.",
    questions={
        "refund": Noul(instructions="Does the customer want a refund?"),
        "team": Choice(instructions="Which team?", criteria={"billing": "payments and refunds", "support": "technical issues"}),
    },
)
```

The default model name `jev-latest` is answered by selfjev-4b, and `client.models.list()`, errors and retries behave as they do against Jev. OpenRouter's decisions path (`/api/alpha/decisions`) is served too.

## 2b. Or use the selfjev client

`pip install selfjev` in your application. It adds `Multi` (select all that apply) and the fine-tuning API.

```python
from selfjev import Choice, Multi, Noul, SelfJev

client = SelfJev(base_url="https://your-selfjev-host:8000", api_key="the key you set in SELFJEV_API_KEYS")

result = client.system_one(
    state="I was charged twice. Please refund the duplicate payment.",
    questions={
        "refund": Noul("Does the customer want a refund?"),
        "team": Choice("Which team should handle this?", {"billing": "payments and refunds", "support": "technical issues"}),
        "topics": Multi(
            "Which topics are mentioned?",
            {"payment": "a payment or charge", "refund": "a refund request", "login": "an account access problem"},
        ),
    },
)

print(result.nouls["refund"].noul)   # probability of yes
print(result.choices["team"].choice)  # selected team
print(result.multis["topics"].multi)  # selected topics
```

`AsyncSelfJev` has the same interface with `await`. `SELFJEV_BASE_URL` and `SELFJEV_API_KEY` work in place of the arguments.

**Images** (new in 0.3.0; Jev takes none): put a photo, a screenshot or a scanned document in `state`, alone or next to
text. A `Path`, image bytes or a PIL image all work; on the wire it is a base64 data URL.

```python
from pathlib import Path

result = client.system_one(
    state=[Path("cat.jpg")],
    questions={"breed": Choice("What breed is it?", {"persian": "a Persian cat", "siamese": "a Siamese cat"})},
)
```

The image is read once and every question is answered against it: 163 ms for one image and one question on an L40S,
136 ms for text.

**Fine-tuning** (a server started with `selfjev serve --fine-tuning`): upload rows, start a job, use its model. A folder
of labelled photos is a few lines:

```python
rows = [{"state": [p], "questions": {"breed": Choice("What breed is it?", {"persian": None, "siamese": None})},
         "answers": {"breed": p.parent.name}} for p in Path("photos").glob("*/*.jpg")]   # photos/persian/1.jpg, ...
f = client.upload_file(rows)                     # or a JSONL file; every row is checked first
job = client.wait_fine_tuning_job(client.create_fine_tuning_job(f.id, suffix="pets").id)
client.system_one(state=[Path("cat.jpg")], questions=rows[0]["questions"], model=job.fine_tuned_model)
```

## Measured

SelfJev-4B Vision (the default since 0.3.0) served by TreeServer, compared with Jev on the same questions:

| Suite | Questions | SelfJev-4B Vision | Jev |
|---|---:|---:|---:|
| Text decisions | 1,991 | **96.1%** | 97.2% |
| AI response review | 946 | **92.5%** | 92.5% |
| Broader text tasks (development benchmark) | 3,471 | **84.1%** | 82.7% |
| Images (10 datasets, 4 never trained on) | 2,002 | **90.4%** | no image input |

These are project evaluations, not a universal ranking. See the [results and limitations](https://github.com/Jwuthri/SelfJev/blob/master/docs/experiments.md) and the [Decision Bench](https://huggingface.co/datasets/Jwuthrich/selfjev-decision-bench) evaluation set.

## Links

- [API reference](https://github.com/Jwuthri/SelfJev/blob/master/docs/api.md) · [Fine-tuning](https://github.com/Jwuthri/SelfJev/blob/master/docs/finetune.md) · [Deployment](https://github.com/Jwuthri/SelfJev/blob/master/docs/deploy.md)
- [Model: selfjev-4b vision](https://huggingface.co/Jwuthrich/selfjev-4b-vision) · [text-only](https://huggingface.co/Jwuthrich/selfjev-4b) · [merged weights (text-only)](https://huggingface.co/Jwuthrich/selfjev-4b-merged)
- [Source](https://github.com/Jwuthri/SelfJev) · [Research notebook](https://jwuthri.github.io/SelfJev/)

The package code is Apache-2.0. The model weights carry their own licenses; see each model card.
