# personal-jev

Fast, instruction-conditioned **binary / multiclass / multilabel** classification with
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B): send a text (the *state*),
natural-language questions and optional candidate labels with descriptions; get typed decisions and
numerical scores from one batched forward pass per batch, with no text generation.

Inspired by the *interface* of TypeSafe's Jev (typed decisions over a state). It is **not** a reproduction of
Jev's undisclosed architecture or training (RLCD). Our native schema is our own; `pjev serve` also exposes a
route with the same request/response *shape* as the decisions API (same shape, different model; see
[Decisions-API-shaped endpoint](#decisions-api-shaped-endpoint)).

## How it works, as comics

Generated with OpenAI `gpt-image-2.5-flare` by [scripts/make_comics.py](scripts/make_comics.py) (all prompts are
in the script). Every number on these pages is a measured result from this repo's reports and training run.

**Deep dive (technical):** [1 · scoring](docs/comics/deep-1-scoring.png) ·
[2 · data](docs/comics/deep-2-data.png) · [3 · LoRA training loop](docs/comics/deep-3-training.png) ·
[4 · calibration, results & limits](docs/comics/deep-4-results.png)

**Friendlier tour (same ideas, analogies):** [1 · the judge](docs/comics/friendly-1-the-judge.png) ·
[2 · the study library](docs/comics/friendly-2-the-library.png) ·
[3 · tiny knobs](docs/comics/friendly-3-tiny-knobs.png) · [4 · the report card](docs/comics/friendly-4-report-card.png)

**Qwen + LoRA, step by step:** [1 · how the text reaches Qwen](docs/comics/lora-1-text-to-qwen.png) ·
[2 · where LoRA plugs in](docs/comics/lora-2-where-lora-plugs-in.png)

![How personal-jev scores](docs/comics/deep-1-scoring.png)

## Results (Apple M5 Pro, MPS, measured 2026-09-23)

**Bottom line.** Fine-tuning the reranker with LoRA raises question accuracy on our test split from 61.0% to
73.5%. The adapter is 4.6M trainable parameters and took 1.05 h to train on this laptop. Jev (82.7%) and GPT-6
Astra (85.8%) are clearly better on the same 3,471 questions, above all on reasoning traps: role reversal,
sarcasm, numeric and temporal reasoning, injections, and long texts with distractors. Our model is only
competitive on public datasets close to its training data. The unmodified reranker is a decent intent/topic
matcher but a poor yes/no judge (binary AUROC 0.605).

"Qwen 0.6B + LoRA fine-tune" means: the pinned Qwen3-Reranker-0.6B weights stay frozen, and the only trained
parameters are rank-16 low-rank adapters added to the q/k/v/o attention projections of all 28 layers (0.76%
of parameters). There is no new classifier head, probe or cross-attention: the score is still the model's own
`yes` − `no` logit.

### Full test split: 3,471 questions (`hf.jsonl` test + `eval.jsonl` test), identical for all four models

| | Qwen 0.6B, unmodified | Qwen 0.6B + LoRA fine-tune | Jev (`typesafe/jev-1.13-20260917`) | GPT-6 Astra (reasoning low) |
|---|---|---|---|---|
| question accuracy % | 61.0 | 73.5 | 82.7 | **85.8** |
| binary accuracy % | 55.3 | 77.8 | 93.5 | **93.8** |
| binary F1 % | 55.8 | 75.1 | 93.2 | **93.4** |
| binary AUROC | 0.605 | 0.863 | **0.981** | 0.971 |
| multiclass accuracy % | 73.3 | 78.3 | 84.6 | **87.0** |
| multiclass macro-F1 % | 79.6 | 82.3 | 91.6 | **95.3** |
| multilabel exact match % | 1.2 | 31.1 | 40.1 | **55.8** |
| multilabel micro-F1 % | 23.6 | 62.9 | 66.3 | **74.0** |
| multilabel label AUROC | 0.679 | 0.882 | 0.890 | **0.940** |
| binary ECE | 0.384 | 0.066 | **0.045** | 0.050 |
| binary Brier | 0.397 | 0.155 | **0.051** | 0.058 |
| multiclass ECE (top-label) | **0.026** | 0.027 | 0.099 | 0.087 |
| multiclass log loss | 0.717 | **0.601** | 2.134 | 0.605 |

Paired exact McNemar tests (counts are questions only one of the two models gets right):

| comparison | fine-tune right | other right | p |
|---|---|---|---|
| fine-tune vs unmodified | 605 | 172 | ≈ 3e-57 |
| fine-tune vs Jev | 197 | 516 | ≈ 8e-34 |
| fine-tune vs GPT-6 Astra | 153 | 580 | ≈ 3e-59 |

Notes on method:
- Jev often returns hard 0/1 choice probabilities, so its confident misses give it a high multiclass log loss.
- The decisions API has no multilabel type, so Jev multilabel is one `noul` per candidate.
- GPT-6 Astra ran with `reasoning.effort=low`, JSON-schema output, and our labeling policy in its system prompt.
- Cost: Jev $0.064 on OpenRouter credit. GPT-6 Astra $19.27, billed to the OpenAI key attached to OpenRouter
  (BYOK). Full tables: [reports/external/full/comparison.md](reports/external/full/comparison.md).

**Accuracy % by family.** `heldout_*` families never appear in our training data. `eval_*` families have
17–32 test questions each, so single-family differences under about 20 points are within noise.

| family | n | unmodified | + LoRA | Jev | GPT-6 Astra |
|---|---|---|---|---|---|
| eval_adversarial | 27 | 74.1 | 63.0 | 100.0 | 100.0 |
| eval_agent_output (no training data) | 26 | 30.8 | 46.2 | 92.3 | 100.0 |
| eval_evidence | 17 | 41.2 | 82.4 | 100.0 | 100.0 |
| eval_multilabel | 32 | 12.5 | 34.4 | 100.0 | 100.0 |
| eval_policy | 22 | 31.8 | 40.9 | 72.7 | 100.0 |
| eval_routing | 25 | 48.0 | 76.0 | 100.0 | 100.0 |
| eval_urgency_sentiment | 22 | 40.9 | 54.5 | 95.5 | 100.0 |
| heldout_boolq | 300 | 59.3 | 70.3 | 90.7 | 89.0 |
| heldout_emotion_multiclass | 300 | 51.0 | 56.0 | 57.7 | 62.3 |
| heldout_intent_clinc | 300 | 92.3 | 89.3 | 94.3 | 95.3 |
| heldout_question_type_trec | 300 | 64.3 | 68.0 | 94.0 | 95.3 |
| heldout_sentiment_sst2 | 300 | 50.0 | 76.7 | 96.7 | 97.0 |
| heldout_topic_dbpedia | 300 | 89.3 | 91.7 | 98.0 | 99.0 |
| hf_emotions_multilabel | 300 | 0.0 | 32.3 | 31.7 | 49.3 |
| hf_intent_banking77 | 300 | 89.3 | 92.7 | 95.7 | 96.3 |
| hf_nli | 300 | 56.7 | 89.0 | 92.7 | 93.7 |
| hf_sentiment_tweets | 300 | 54.3 | 66.7 | 64.3 | 68.7 |
| hf_topic_agnews | 300 | 77.3 | 86.7 | 87.3 | 90.0 |

**Read with care.** The `eval_*` families were written and blind-verified by Claude Opus. GPT-6 Astra scores 100%
on all seven and Jev 92–100% on six of them, so these families separate small models from frontier ones rather
than frontier models from each other. On noisy-label public sets (GoEmotions, TweetEval, dair-ai emotion)
every model stays well below 100%.

### Where we fail most ([reports/external/failures.md](reports/external/failures.md))

The fine-tune misses 919 of 3,471 test questions. Jev and GPT-6 Astra both get 480 of those right, and all
three models miss 303. Of those 303, 235 (78%) are in the noisy-label GoEmotions, dair-ai emotion and
TweetEval sets. Where the fine-tune is right,
Jev is wrong on 197.
Largest gaps to Jev by hard-case tag, in error-rate points: sarcasm +67, numeric reasoning +56, role reversal
+53, injection +50, paraphrase +50, distractors +49, lexical overlap +45, long states +44, negation +43,
temporal reasoning +41. On contradiction (+9) and missing-evidence (+3) cases, training closed most of the
gap. [reports/external/failures.jsonl](reports/external/failures.jsonl) lists every miss worst-first with the
unmodified model's, Jev's and GPT-6 Astra's answers. These are **test** questions: use them to design new training data,
never as training data.

### Calibration

- **Base:** raw binary scores are badly calibrated (ECE 0.384). The held-out temperature (T = 21.6) brings ECE
  to 0.025, but only by squashing every probability toward 0.5 (Brier 0.244, near a coin flip's 0.25).
  Calibration cannot add the signal that isn't there.
- **LoRA:** calibrated enough out of the box (binary ECE 0.066). The fitted temperatures (binary 1.16,
  multiclass 1.27, multilabel 1.05) barely change anything.
- **New families:** calibration fitted on in-distribution data does not transfer. Examples: LoRA ECE 0.19 on
  SST-2 and 0.35 on agent-output, and multiclass temperature scaling makes ECE on CLINC and DBpedia *worse*.

### Speed ([reports/bench/](reports/bench/); end-to-end p50, one request at a time)

| request | pairs | fp32 base | bf16 base | fp32 LoRA |
|---|---|---|---|---|
| 512-token state, 1 question × 3 candidates | 3 | 351 ms | **150 ms** | 383 ms |
| 512-token state, 16 × 3 | 48 | 5.4 s | 2.5 s | 6.0 s |
| 2,048-token state, 1 × 3 | 3 | 1.4 s | 0.57 s | 1.5 s |
| 2,048-token state, 16 × 3 | 48 | 23.3 s | 9.6 s | 26.6 s |
| 8,192-token state, 1 × 3 | 3 | 8.6 s | 2.9 s | 9.4 s |
| 8,192-token state, 16 × 3 | 48 | 150 s | 46.7 s | 150 s |
| 16K / 32K-token state, 1 binary question | 1 | 9.1 s / 28.8 s | 2.5 s / 7.3 s | — |

- Throughput is flat per token (fp32 ≈ 5.3K tokens/s at 512-token states and ≈ 2.8K at 8K; bf16 2.3–3.2× faster),
  so latency ≈ pairs × state tokens.
- Peak MPS memory is 2.2–6.8 GB. 32K-token inputs fit. Cold start is 1.1 s to load plus the first request.
- **bf16 costs no quality** on the LoRA model: test accuracy 73.7% vs 73.5% in fp32, identical AUROC 0.863;
  43 of 3,471 decisions flip. Serve in bf16.
- The unmerged adapter adds 8–14% latency; merging it into the weights would remove that.
- We did not benchmark Jev's latency. The spec rules out claiming Jev-like ~100 ms: we only get close for a
  single short question in bf16.

### Training run ([runs/lora_pilot/train_meta.json](runs/lora_pilot/train_meta.json), config [configs/lora_pilot.json](configs/lora_pilot.json))

- **Setup:** LoRA r=16, α=32, dropout 0.05 on `q/k/v/o_proj` (4,587,520 trainable of 600M parameters,
  0.76%). The base model runs in bf16 with fp32 adapters and gradient checkpointing; lr 2e-4 with 5% warmup and
  linear decay; 1 epoch.
- **Data:** 10,112 questions: public datasets capped at 1,600 per family plus 2,112 synthetic.
- **Run:** 183 optimizer steps in 1.05 h on the M5 Pro.
- **Validation:** loss 1.129 at step 0 (unmodified model) → 0.486 → 0.477 → 0.435 at step 150, which was the
  selected checkpoint. Steps 151–183 were never validated because the data ran out before the planned final
  validation; that edge case is now fixed.
- **Reload check:** the saved adapter reloads exactly (0.0 difference with identical batching in bf16;
  0.0000 in fp32). The 0.125 figure the run logged was bf16 batch-composition noise, not a save bug; the check
  now uses identical batching.
- **Prompt:** `task-v1` was chosen on validation data only (0.614 vs 0.597–0.609 for the other three), a small
  margin.

## Scaling up: 4B and 8B (AWS, one NVIDIA L40S, 2026-09-23)

**TL;DR.** Fine-tuning matters far more than model size.
- **Size with LoRA:** 4B (80.3%) and 8B (80.7%) are statistically tied (p = 0.52), and both are well above 0.6B
  (73.5%).
- **Against Jev:** 8B + LoRA is 2.0 points behind (82.7%, p ≈ 0.001) and beats it on multilabel.
- **Best overall:** GPT-6 Astra (85.8%).
- **Still missed:** our models fail most on reasoning traps: numbers, dates, who-did-what, and instructions
  planted in the text.

Same 3,471 test questions for every column:

| | 0.6B | 0.6B + LoRA | 4B | 4B + LoRA | 8B | 8B + LoRA | Jev | GPT-6 Astra |
|---|---|---|---|---|---|---|---|---|
| question accuracy % | 61.0 | 73.5 | 62.8 | 80.3 | 66.2 | 80.7 | 82.7 | **85.8** |
| binary AUROC | 0.605 | 0.863 | 0.604 | 0.945 | 0.658 | 0.945 | **0.981** | 0.971 |
| multiclass macro-F1 % | 79.6 | 82.3 | 83.4 | 86.2 | 86.5 | 88.7 | 91.6 | **95.3** |
| multilabel exact match % | 1.2 | 31.1 | 2.0 | 47.7 | 10.2 | 48.0 | 40.1 | **55.8** |
| binary ECE | 0.384 | 0.066 | 0.357 | 0.056 | 0.364 | 0.051 | **0.045** | 0.050 |

Full tables are in [reports/scale_comparison.md](reports/scale_comparison.md), generated by
[scripts/scale_table.py](scripts/scale_table.py): every metric, per family, per hard-case tag, and paired
McNemar tests.

**8B + LoRA vs Jev**, by question accuracy:
- **Wins:** GoEmotions multilabel 45.7 vs 31.7, tweet sentiment 69.0 vs 64.3, AG News 89.3 vs 87.3,
  missing-evidence cases 87.5 vs 84.6.
- **Losses:** the authored families (agent-output grading 53.8 vs 92.3, policy 50.0 vs 72.7) and the traps
  (numeric 62.5 vs 93.8, temporal 51.7 vs 79.3, role reversal 60 vs 100, injection 70 vs 100).

**How it ran:**
- **Script:** [scripts/run_model.sh](scripts/run_model.sh) on one `g6e.xlarge` (NVIDIA L40S 48 GB) in the
  Connectly AWS account: us-east-1, default VPC, tagged `Project=personal-jev`, and every resource deleted
  afterwards.
- **Steps:** prompt selected per model on validation only (4B picked `answer-v1`, 8B `task-v1`), then baseline
  evals with held-out calibration, then LoRA, then tuned evals.
- **LoRA:** rank 16 on q/k/v/o. That is 11.8M trainable parameters for 4B (0.29%) and 15.3M for 8B (0.19%).
- **Training:** the same 10,112 questions and hyperparameters as the 0.6B, except 16K-token micro-batches × 2
  accumulation. bf16 throughout.
- **Time and result:** 42 min to train the 4B, 50 min for the 8B. The best checkpoint was the last step for
  both, and both adapters reload exactly.
- **Cost:** 3.16 instance-hours ≈ $5.89.
- **Capacity:** g6e was scarce. There was no `g6e.2xlarge` or `g6e.4xlarge` in any us-east-1 zone, and only one
  `g6e.xlarge` in us-east-1a, so the two models ran back to back.
- **Precision caveat:** the 0.6B columns were evaluated in fp32 on the Mac, the 4B/8B columns in bf16 on the GPU.
  On the 0.6B, switching to bf16 moved test accuracy by +0.2 points.

**Speed on the same GPU** (L40S, bf16, end-to-end p50, one request at a time, adapters not merged;
[reports/bench/](reports/bench/)):

| request | pairs | 0.6B | 0.6B + LoRA | 4B | 4B + LoRA | 8B | 8B + LoRA |
|---|---|---|---|---|---|---|---|
| 512-token state, 1 question × 3 candidates | 3 | 28 ms | 43 ms | 99 ms | 116 ms | 162 ms | 188 ms |
| 512-token state, 16 × 3 | 48 | 421 ms | 608 ms | 1.9 s | 2.4 s | 2.9 s | 3.6 s |
| 2,048-token state, 1 × 3 | 3 | 79 ms | 110 ms | 390 ms | 480 ms | 661 ms | 768 ms |
| 2,048-token state, 16 × 3 | 48 | 1.7 s | 2.4 s | 7.1 s | 9.0 s | 11.0 s | 13.2 s |
| 8,192-token state, 1 × 3 | 3 | 409 ms | 553 ms | 1.8 s | 2.2 s | 2.9 s | 3.3 s |
| 8,192-token state, 16 × 3 | 48 | 6.6 s | 8.8 s | 29.1 s | 35.3 s | 43.8 s | 51.9 s |

- **Memory:** peak GPU memory is 1.6 GB (0.6B), 8.7 GB (4B) and 16.9 GB (8B).
- **Throughput:** mean ≈ 66K, 15K and 9.5K tokens/s for 0.6B, 4B and 8B.
- **LoRA overhead:** 15–50% with the adapter unmerged; merging it into the weights removes it.

## Shared-prefix tree scorer (best model so far)

The text is read **once**, and each question and candidate still reads it through **all** layers. One forward
pass runs over a token tree (text → questions → candidates) with a tree attention mask, and yes/no logits are read
at each leaf. Each leaf scores exactly like the standalone `text + question + candidate` sequence; the tests check
this to 1.7e-5 on the real model. Full write-up: [docs/tree_model.md](docs/tree_model.md). Test split, 3,471
questions; speed on one A10G, bf16, both with LoRA:

| | question accuracy % | binary AUROC | multiclass accuracy % | 16 questions × 3 candidates, 8K-token text |
|---|---|---|---|---|
| stock Qwen3-Reranker-4B + LoRA | 80.3 | 0.945 | 82.3 | 90,449 ms |
| **tree Qwen3-Reranker-4B + LoRA** | **81.6** | **0.953** | **83.9** | **2,798 ms** |
| Jev | 82.7 | 0.981 | 84.6 | not measured |

- **Accuracy.** It beats the stock 4B trained on the same data (p = 0.016). The gap to Jev is not significant
  (p = 0.08).
- **Speed.** It is 32–37× faster with 16 × 3 questions on 8K–16K-token texts, and the same speed for a single
  question.
- **Base model.** A general instruct model (Qwen3-4B-Instruct-2507) was better untrained but ended at 80.4% after
  LoRA, so the reranker is the better base.
- **Usage.** `pjev classify request.json --tree --model Qwen/Qwen3-Reranker-4B --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 --adapter runs/tree_4b/adapter --dtype bfloat16`.

### Data and base-model curves (4B): the stock recipe saturates at 80–81%

Ten more Qwen3-Reranker-4B LoRA runs with the `lora_4b` recipe, one A10G each, to see what moves the 80.3%
(tables and paired tests: [reports/curve/summary.md](reports/curve/summary.md), configs in `configs/curve/`,
`scripts/run_curve.sh`):

| lever | runs | test question accuracy % |
|---|---|---|
| more of the same data (nested 25% / 50% / 100% of the 10,112 questions) | 1 epoch | 77.0 / 78.9 / 80.3 |
| | 2 epochs | 79.1 / 79.1 / 79.8 |
| a new task type: BoolQ training questions added (+0 / +100 / +300 / +1,000 / +3,000) | BoolQ test | 83.3 / 84.3 / 83.3 / 84.3 / 86.7 (Jev 90.7) |
| | overall | 80.3 / 80.0 / 80.9 / 80.9 / 80.9 |
| base model: Qwen3-4B-Instruct-2507, same pair format | zero-shot | 71.3 (binary 84.2, AUROC 0.911) vs 62.8 for the reranker |
| | + LoRA | 80.6 vs 80.3 (p = 0.68) |

- Each doubling of the data buys about 1.5 points, and 2 epochs of 25% equal 1 epoch of 50%: matching Jev with more
  of the same data would take several times the current set, if the slope held.
- A new task type needs thousands of labeled examples for a few points, and the overall score does not move
  significantly (p ≥ 0.09 for every BoolQ run).
- The instruct base is much better untrained, above all on yes/no, but identical after LoRA. Its tree-format
  counterparts are in `reports/tree_zeroshot_instruct_4b` and `reports/tree_4b_instruct`.
- Data volume, epochs and base model all land at 80–81%. The tree scorer's 81.6% is the only lever so far that
  moved the test score up.

## Custom shared-state model

A second model follows the v1 spec. It encodes the text **once** with the same Qwen backbone, encodes each
question + candidate separately, and scores them with new cross-attention blocks and small heads, with no yes/no
logits. Full write-up: [docs/custom_model.md](docs/custom_model.md). Test split, 3,471 questions:

| | question accuracy % | multiclass accuracy % | binary AUROC | 16 questions × 3 candidates, 8K-token text (A10G, bf16) |
|---|---|---|---|---|
| stock reranker + LoRA (above) | **73.5** | **78.3** | **0.863** | 24,024 ms |
| custom, as specified | 39.0 | 39.1 | 0.536 | — |
| custom + similarity term + joint LoRA | 58.2 | 70.2 | 0.514 | **626 ms** |

With 16 questions × 3 candidates on one 8K–16K-token text it is 38–43× faster on the same GPU, and it ties or beats
the stock LoRA on Banking77 and AG News. It still loses on unseen label sets and cannot do yes/no questions. Use it with
`pjev classify request.json --checkpoint runs/custom_sim_lora/checkpoint`.

## What is actually computed

For every (state, question, candidate) triple (stock backend):

1. Map the question to the reranker's `(Instruct, Query, Document)` slots using a named prompt mapping
   (`formatting.PROMPTS`; the default was selected on validation data, see
   [reports/prompt_selection.md](reports/prompt_selection.md)). The state is always the Document; stable
   label ids stay in our metadata and only descriptions reach the model.
2. Wrap it in the model card's official template, verbatim: the fixed system prefix, `<Instruct>: … <Query>: … <Document>: …`
   and the `<think>\n\n</think>\n\n` assistant suffix. The template tokens count toward `max_length`.
3. Tokenize like the official Transformers reference (prefix, pair text and suffix are tokenized separately and concatenated).
4. Sort pairs by length and pack them into batches under a padded-token budget, with left padding so
   position −1 is every row's last real token.
5. Run one causal forward pass per batch (`logits_to_keep=1`, `use_cache=False`, no `generate()`), with the
   stock causal mask and attention implementation.
6. Compute `s = z_yes − z_no` from the "yes" (9693) and "no" (2152) logits. `sigmoid(s)` equals the model
   card's `softmax([z_no, z_yes])[1]`, and a test checks this against the reference code to 1e-4.
7. Group scores back to their questions and apply the output rule:

| type | probability | decision |
|---|---|---|
| binary | `p_yes = sigmoid(s / T)`, `p_no = 1 − p_yes` | `p_yes ≥ threshold` (request → validation-selected → 0.5 default; source recorded) |
| multiclass | `softmax(scores / T)` within the question only | argmax (exact ties broken by id, never by input order); optional `abstain_below` → `selected: null` |
| multilabel | `sigmoid(scores / T)` per candidate, no sum constraint (marginals, not a joint distribution) | every candidate with `p ≥ threshold` |

`T = 1` and the status is `uncalibrated` unless a calibration file fitted on a held-out calibration split
supplies a temperature for that type (status `heldout_temperature_scaled`). A multiclass argmax always picks
something: add an explicit "none of the above" candidate or set `abstain_below`.

**Cost model.** Every candidate is its own pair and re-encodes the whole state. A request with 16 multiclass
questions of 3 candidates each is 48 forward rows over the state; question count alone is not the workload.
There is no shared state encoding or KV-prefix reuse in v1.

## Install

```bash
uv sync                  # Python 3.12, torch, transformers, peft, pytest (pinned in uv.lock)
uv sync --group data     # + datasets/pyarrow, only needed to rebuild data/hf.jsonl
```

The checkpoint is pinned to revision `e61197ed45024b0ed8a2d74b80b4d909f1255473` and downloads on first use
(~1.2 GB).

## Use

```bash
uv run pjev classify examples/request.json                        # unmodified base model
uv run pjev classify examples/request.json --adapter runs/lora_pilot/adapter --calibration calib/lora_pilot.json
```

```python
from personal_jev.model import Scorer
from personal_jev.classify import classify

scorer = Scorer()                         # fp32 on mps/cuda/cpu; Scorer(adapter="runs/lora_pilot/adapter") for LoRA
result = classify(scorer, request_dict)   # {"questions": [...], "meta": {...}}
```

The request format is in [examples/request.json](examples/request.json). Validation rejects duplicate
question or candidate ids, empty states, questions or candidate lists, a multiclass question with fewer than 2
candidates, non-numeric or out-of-range thresholds, and unknown fields. Input longer than `max_length`
(default 8192, at most 32768) raises `InputTooLong` naming the question and candidate. Nothing is ever truncated.

Each question in the response carries `id`, `type`, raw `score`(s), probabilities, `selected` (bool, id or
list of ids), `threshold` and `threshold_source`, and `calibration`. `meta` records the model and pinned
revision, adapter path and sha256, prompt name and sha, device, dtype, `max_length`, pair and token counts
(including padding), and tokenize, model and total milliseconds.

## Data

One JSONL line per (state, question, target), with `id`, `source_id`, `family`, `split`, `provenance`,
`state`, `question` (`type`, `instruction`, `candidates`), `target`, `hard_cases` and optional
`paraphrase_group` and `notes`. Authored files use a compact form (one state, several questions) that
`data.load` expands. Check any file with `uv run python -m personal_jev.data check FILE`.

**Labeling policy** (all sources): binary `true` means *the text supports answering yes*. Contradicted and
simply-not-stated are both `false`; `false` is "not supported by the text", not "known false". Instructions
inside the state are data. Negations, hypotheticals and future conditionals are not the thing itself.
Multilabel targets are complete over the listed candidates: candidates with unknown labels are left out
rather than labeled negative.

| file | what | size | provenance |
|---|---|---|---|
| `data/dev.jsonl` | development fixtures, used in tests and for quick checks | 14 states / 28 questions | written in this session by Claude Code; pending human review |
| `data/eval.jsonl` | **evaluation set** over 7 families; hard cases tagged | 155 states / 398 questions: validation 118, calibration 109, test 171 | written by 7 Claude Opus agents from [data/eval/BRIEF.md](data/eval/BRIEF.md), with no access to training data; a separate blind Opus re-labelling agreed on 398/398 ([review](data/eval/review/REVIEW.md)); **not human-reviewed** |
| `data/hf.jsonl` | public human-labeled datasets converted to our schema | 16,800: train 12,000; validation, calibration 750 each; test 1,500 in-distribution + 1,800 held-out | 11 pinned HF revisions, licenses per row ([scripts/build_hf.py](scripts/build_hf.py)); label descriptions written by Claude Sonnet ([data/hf/label_descriptions.json](data/hf/label_descriptions.json)) |
| `data/synthetic.jsonl` | hard, long, trap-heavy **training** data in 6 families | 923 states / 2,405 questions: train 2,080, validation 110, calibration 122, test 93 | written by 12 Claude Sonnet agents from [data/synthetic/BRIEF.md](data/synthetic/BRIEF.md); labels are LLM-intended, not ground truth; generation stopped at about 75% of the plan when a usage limit hit; an Opus label review dropped 43 wrong or ambiguous questions ([review](data/synthetic/review/REVIEW.md)) |

- **Splits** are assigned by hashing `source_id`, so all questions and paraphrases about one state share a
  split. `build_data.py` drops any synthetic state whose word-8-gram containment with an eval state is
  ≥ 0.3. It dropped 0, and 0 HF train states overlap eval states.
- **Held-out families** never appear in training: CLINC150 intents (with an out-of-scope "none" option),
  DBpedia-14 topics, TREC question types, dair-ai emotion, BoolQ and SST-2. The eval family
  `eval_agent_output` (grading AI replies against narrow criteria) is also excluded from all training data.
- **Contamination caveat:** public datasets such as Banking77, AG News and TweetEval may overlap with the
  base model's own training data, which could inflate absolute baseline numbers on them. The authored eval
  set is new text.

## Decisions-API-shaped endpoint

```bash
uv run pjev serve --adapter runs/lora_pilot/adapter --dtype bfloat16      # http://127.0.0.1:8000
```

`POST /api/alpha/decisions` accepts the same body as the decisions API used through OpenRouter
(`state` plus `questions: {id: {type: noul | choice | score, instructions, criteria}}`) and returns
`answers` in the same shape. Client code only changes its base URL (no auth locally). `POST /classify` serves
our native schema. It is the same shape but a **different model**, so its probabilities are not
interchangeable with the hosted service's. The response names our model and says so in `meta.note`. Mapping:

| type | criteria | our computation | answer |
|---|---|---|---|
| `choice` | `{label: description}` | multiclass over `"label: description"` | `choice`, `probabilities` |
| `noul` | `{"true": …, "false": …}` | 2-way choice between the two descriptions | `noul` = P(true) |
| `noul` | none | our binary yes/no | `noul` = p_yes |
| `score` | `[level_0, …, level_k]` (ordered) | distribution over levels | `probabilities` per level; `score` = Σ pᵢ·i/k in [0, 1] (our definition) |

On your example request (payouts failing for 3 days), the LoRA model in bf16 answered in 71 ms: `technical`
0.89, `noul` 0.65, `score` 0.52 (mostly "Frustrated").

## Reproduce

```bash
uv run pytest                                   # 48 tests: 42 without the model + 6 on the real checkpoint (~35 s on M5 Pro)
uv run python scripts/build_hf.py               # data/hf.jsonl from pinned HF revisions
uv run python scripts/build_data.py             # data/eval.jsonl + data/synthetic.jsonl (hash splits, overlap check)
uv run python scripts/select_prompt.py data/hf.jsonl data/eval.jsonl   # validation-only prompt selection
scripts/run_experiments.sh                      # baseline evals, calibration, LoRA training, tuned evals, compare, bench
scripts/run_model.sh 4b Qwen/Qwen3-Reranker-4B 22e683669bc0f0bd69640a1354a6d0aebcfeede5   # same pipeline for 4B/8B (GPU)
scripts/run_model.sh 8b Qwen/Qwen3-Reranker-8B 77d193c791ed757ca307ee72715aa132723da912
uv run python scripts/scale_table.py > reports/scale_comparison.md                         # size comparison table
uv run python scripts/summarize.py              # README-style tables straight from the reports
zsh -ic 'uv run python scripts/compare_external.py --per-hf-family 300 --tag full --only jev --budget 5'
zsh -ic 'uv run python scripts/compare_external.py --per-hf-family 30 --tag subset --budget 20'
uv run python scripts/failures.py               # reports/external/failures.{md,jsonl}
```

`compare_external.py` reads `OPENROUTER_API_KEY` from the environment and never prints it. It caches every
response under `reports/external/cache/`, so reruns cost nothing. It stops at `--budget` USD, counting
OpenRouter charges plus upstream BYOK cost (BYOK calls show $0 on OpenRouter).

`pjev calibrate` fits temperatures only on a report whose every prediction is from the `calibration` split
and selects thresholds only on `validation`. Anything else raises `LeakageError`. A calibration file is
bound to the model revision, adapter sha256 and prompt sha that produced it, and refuses to load for any
other scorer.

## Layout

```
src/personal_jev/  schemas.py (request validation)  formatting.py (official template + prompt mappings)
                   model.py (Scorer: tokenize, batch, yes/no logits)  classify.py (grouping, output modes)
                   data.py (JSONL, splits)  train.py (LoRA)  evaluate.py (metrics, reports)
                   calibration.py (temperature, thresholds)  benchmark.py  server.py (HTTP)  cli.py
                   custom.py (shared-state model)  train_custom.py (its training)  -> docs/custom_model.md
                   tree.py (shared-prefix tree scorer)  train_tree.py (its LoRA training)  -> docs/tree_model.md
scripts/           build_hf.py  build_data.py  select_prompt.py  run_experiments.sh  summarize.py
                   compare_external.py (Jev / GPT-6 Astra via OpenRouter)  failures.py
data/              dev.jsonl  eval.jsonl (+ eval/BRIEF.md, eval/review/)  hf.jsonl (+ hf/)  synthetic.jsonl (+ synthetic/BRIEF.md)
configs/           lora_pilot.json
reports/           prompt selection, eval reports, comparisons, benchmarks, external/ (Jev, GPT-6 Astra, failures)
tests/             test_logic.py (no model)  test_server.py (API shape, no model)  test_model.py (real checkpoint)
```

## Limitations

- **Eval labels:** the eval set is LLM-written and LLM-verified with no human review. Family slices are small
  (17–32 test questions).
- **Synthetic data:** the training data is LLM-authored. Its generator (Claude Sonnet) belongs to the same model
  family as the eval authors, so part of the gain on the `eval_*` families may be style match. Held-out
  public families and `eval_agent_output` are the cleaner generalization signals.
- **Contamination:** public datasets may overlap the base model's own training data.
- **Single run:** one seed, one epoch, one hyperparameter setting, and only 10K training questions.
- **Speed:** measured on Apple MPS only. Every candidate re-reads the whole state, so many questions over a
  long state is this design's worst case.

## Next

- **Pick 4B + LoRA** as the working model. It matches 8B + LoRA on quality (80.3% vs 80.7%, not significant)
  at about 60% of the latency and half the memory.
- **Aim new training data at the failure tags:** numeric and temporal reasoning, role reversal, injection,
  multi-positive multilabel. These must be *new* examples, never the test items in
  `reports/external/failures.jsonl`. A second epoch and a larger rank, or LoRA on the MLP layers, are the cheap
  next ablations.
- **Serving:** merge the adapter into the weights (removes the 15–50% LoRA overhead) and serve in bf16.
- **Many questions over long texts:** see the custom shared-state model notes (KV-prefix reuse).
