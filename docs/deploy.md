# Deploy

`selfjev serve` runs [the API](api.md) on one NVIDIA GPU with at least 16 GB (selfjev-4b is Qwen3.5-4B in bf16 plus a
LoRA). Three ways to run it, from the most hands-on to the least.

## On a GPU machine you have

```bash
git clone https://github.com/Jwuthri/SelfJev.git && cd SelfJev
git lfs pull --include "weights/selfjev_4b/*"
uv sync --extra serve --extra gpu
SELFJEV_API_KEYS=my-key uv run selfjev serve --host 0.0.0.0 --port 8000
```

The base model (about 9 GB) downloads from Hugging Face on first start, at a pinned revision. Without
`SELFJEV_API_KEYS` (comma-separated) the server accepts every request, which suits a laptop or a private network.
The `my-key` shown above is a placeholder: choose a long random secret and pass the same value as `api_key` in the SDK
or `SELFJEV_API_KEY` in your client environment. Hugging Face tokens and Jev API keys are unrelated to this secret.

Useful flags: `--max-length` (state plus longest question, default 32,768 tokens), `--max-batch-tokens` (tokens packed
per forward pass), `--calibration` (temperatures from `selfjev calibrate`), `--engine vllm --model-dir <merged>` (vLLM on
weights merged by `selfjev merge`: fast for one question per request, slower for many; the default tree engine reads
the text once for all questions, see [speed](speed.md)).

## Docker

```bash
git lfs pull --include "weights/selfjev_4b/*"
docker build -f deploy/Dockerfile -t selfjev .
docker run --gpus all -p 8000:8000 -e SELFJEV_API_KEYS=my-key selfjev
```

The image bakes the base model in (`--build-arg BAKE_MODEL=0` to download it at start instead) and has a health check
on `/health`. `docker compose -f deploy/docker-compose.yml up -d` does the same with `SELFJEV_API_KEYS` from the
environment or an `.env` file.

## AWS, one command

Needs AWS credentials and `pip install "selfjev[deploy]"` (or `uv sync --extra deploy`).

```bash
selfjev deploy aws machines                                   # presets and prices
selfjev deploy aws up --name prod --instance g6.xlarge        # prints the endpoint and a new API key
selfjev deploy aws status --name prod                         # health, uptime, cost so far
selfjev deploy aws list
selfjev deploy aws down --name prod                           # terminates the box, deletes its security group
```

`up` starts NVIDIA's Deep Learning Base AMI and, on first boot, installs this repository at `--ref` (default `master`)
with only the selfjev-4b weights, pre-downloads the base model and runs `selfjev serve` as a systemd service on port
8000 behind the key. Setup installs packages and fetches about 9 GB of model; `up` waits for `/health` (up to 30
minutes) unless `--no-wait`. The deployment
record, key included, is kept in `~/.selfjev/deployments/<name>.json`; every resource is tagged `Project=selfjev`.

| preset | GPU | $/h (us-east-2, on demand) | for |
|---|---|---|---|
| `g6.xlarge` (default) | L4, 24 GB | 0.805 | the cheapest 24 GB GPU; serving on it is not measured yet |
| `g5.xlarge` | A10G, 24 GB | 1.006 | when L4 capacity is short |
| `g6e.xlarge` | L40S, 48 GB | 1.861 | long texts, more traffic, fine-tuning on the box |
| `g6e.2xlarge` | L40S, 48 GB | 2.242 | as g6e.xlarge with more CPU |
| `p5.4xlarge` | H100, 80 GB | 6.88 | lowest latency; often out of capacity |

Other options: `--region` (default `us-east-2`), `--allow-cidr` (who may reach the port, default everyone; the key still
applies), `--api-key` (bring your own), `--max-hours` (the box terminates itself after that long: a cost cap for
trials), `--ssh` (a key pair and port 22 from your IP, to read `/var/log/selfjev-setup.log`).

There is no pause: `down` terminates the box and billing stops; `up` builds a fresh one.

## Fine-tuning on the server

`selfjev serve --fine-tuning` (or `selfjev deploy aws up --fine-tuning`) adds the
[fine-tuning routes](api.md#fine-tuning): upload a JSONL of requests with their expected answers, start a supervised or
RLCD job, and the fine-tuned model is served next to `selfjev-4b` as soon as the job succeeds. Jobs train on the same
GPU as serving, one at a time. Our training runs used 48 GB L40S cards (`g6e.2xlarge`; `g6e.xlarge` has the same GPU); a
24 GB card next to serving is untested and likely too small. Job state
lives in `--home` (`SELFJEV_HOME`, default `~/.selfjev/server`). In Docker, mount a volume there and add the flag:
`docker run --gpus all -p 8000:8000 -v selfjev:/root/.selfjev selfjev uv run --no-sync selfjev serve --host 0.0.0.0 --fine-tuning`.

To train elsewhere (a bigger box, a notebook), run `selfjev finetune` or `selfjev rlcd` there
([fine-tune and RLCD](finetune.md)) and serve the resulting adapter with `selfjev serve --adapter <run>/adapter`.

## Operating it

- `GET /health`: 200 with the queue depth once the model is loaded (open, for load balancers).
- `GET /metrics`: Prometheus text: requests by route and status, latency, questions and tokens, queue depth (open).
- Every response carries `x-request-id`; errors are JSON (`{"error": {"type", "message", "param"}}`). An unhandled
  error is a 500 whose message names the request id; the server logs its traceback under that id.
- Concurrent requests are batched into shared forward passes (up to 32 requests, 5 ms wait). When 256 requests are
  waiting the server answers 529 with `Retry-After`; the SDK retries 429, 529 and 5xx with backoff.
