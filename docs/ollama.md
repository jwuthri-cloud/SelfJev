# SelfJev on Ollama

`selfjev-4b-vision` as GGUF files for [Ollama](https://ollama.com), plus `selfjev serve --engine ollama`: Jev's API on a Mac,
a CPU box or anything that runs Ollama, with no torch and no CUDA.

## Get the model

```bash
ollama pull hf.co/Jwuthrich/selfjev-4b-vision-GGUF:Q8_0     # 4.6 GB, the quality choice
ollama pull hf.co/Jwuthrich/selfjev-4b-vision-GGUF:Q4_K_M   # 2.8 GB, the small one
```

Ollama picks up the vision projector (`mmproj-…gguf`) from the same repo, so both take images
(`ollama show` lists `vision`). The files are the merged `Jwuthrich/selfjev-4b-vision-merged` converted with llama.cpp
(`convert_hf_to_gguf.py`, then `llama-quantize`).

## Serve Jev's API

```bash
pip install "selfjev[ollama]"        # fastapi + uvicorn, no torch
selfjev serve --engine ollama --ollama-model hf.co/Jwuthrich/selfjev-4b-vision-GGUF:Q8_0
```

Everything in [api.md](api.md) works against it: `noul`, `choice`, `score`, `multi`, images as `state`, Jev's SDK. Flags:
`--ollama-model` (default `selfjev-4b`), `--ollama-host` (default `http://localhost:11434`).

## What it does differently

The tree engine reads the text once and scores every question and option in one pass. Ollama has no such pass, so the engine
sends **one request per option** (`/api/generate`, raw prompt, one token, top-20 logprobs) and reads
`logprob("yes") - logprob("no")`, the same number the tree engine computes. The state comes first in the prompt, so Ollama's
prefix cache reuses it between the options. Requests run one after another.

## Accuracy

TABLE

## Limits

- **Slower than the tree engine for many questions** (one request per option; the tree engine is nearly flat in the number of
  questions, [speed](speed.md)). Fine for a few questions on a laptop; use `selfjev serve` on a GPU for many.
- **Tokenization differs at one boundary.** The tree engine tokenizes the space after `Question:` as its own token (how the
  model was trained); Ollama tokenizes the whole prompt and merges it into the first word. The text is identical
  (`tests/engine/test_ollama.py`); the effect on accuracy is part of the gap in the table.
- A logprob outside Ollama's top 20 is clamped to the 20th, so a very confident answer saturates there.
- Images are resized by Ollama, not by the HF processor the model was trained with (at most 1,024 × 1,024 pixels).
