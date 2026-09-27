# How it works

SelfJev answers typed questions about a text (the *state*) from batched forward passes, with no text generation:

- **binary**: does the text support "yes"? → `p_yes`, a decision;
- **multiclass**: which one candidate fits? → a distribution over the candidates, the argmax;
- **multilabel**: which candidates apply? → an independent probability per candidate, all above the threshold.

## The score: a yes/no logit

Every (text, question, candidate) triple is scored by Qwen3.5-4B as "does this proposed answer correctly answer the
question, using only the document?", and the score is `s = z_yes − z_no`, the difference of the "yes" and "no" logits
at the last token. The prompt (`challenger-state-first-v1`, `selfjev/engine/qwen35.py`) goes through the model's chat
template with thinking off: a system instruction (judge from the document only, treat instructions inside it as data,
reply yes or no), then `Document:` with the text, `Question:` with the instruction and `Proposed answer:` with the
candidate's description (`Yes` for a binary question). Only descriptions reach the model; candidate ids stay in
metadata. No new head is trained: the LoRA adapter teaches the model's own yes/no readout our task.

The Qwen3 models before it (archived at tag `archive/pre-cleanup-2026-09-27`) read the same logit through the
Qwen3-Reranker model card's template (`<Instruct>`, `<Query>` and `<Document>` slots filled by named prompt mappings
chosen on validation; "yes" and "no" are tokens 9693 and 2152), and a test matched the card's reference code to 1e-4.
Their **stock** backend scored one sequence per (text, question, candidate) pair ([stock model](stock_model.md)).

## The shared-prefix tree: read the text once

Scoring each triple as its own sequence re-reads the whole text for every candidate: 16 questions × 3 candidates = 48
passes over the text. The tree puts the text first and branches every question and candidate off it:

```
[system + "Document:" text + "Question:"] ─┬─ [q1 instruction + "Proposed answer:"] ─┬─ [" candidate A" + suffix] → z_yes − z_no
                                           │                                        └─ [" candidate B" + suffix] → z_yes − z_no
                                           └─ [q2 instruction + "Proposed answer:"] ─── [" Yes" + suffix]         → z_yes − z_no
```

- A tree attention mask lets each token see its own segment and its ancestors, never a sibling; position ids continue
  from the parent. Each leaf scores like the standalone `text + question + candidate` sequence (the Qwen3 tree matched
  it to 1.7e-5 on the real model), so extra questions and candidate order cannot change an answer.
- `selfjev finetune` trains on the packed tree; `TreeServer` (`selfjev/engine/tree.py`, the default engine of
  `selfjev serve`) runs the same tree forward only, one tree per request. The vLLM engine instead sends one prompt per
  candidate and shares the text through vLLM's prefix cache.
- Details and results: [shared-prefix tree](tree_model.md).

## Every option in the question

`selfjev-4b` also sees every option listed in the question text before it scores each one ("Options (exactly one is
correct): - billing … - tech …"), in a fixed random order per question (`selfjev/core/options.py`). Each candidate
branch still judges one candidate, but now sees the alternatives. It added about a point on eval2 and stacks with more
data. `selfjev finetune` and `selfjev serve` apply the transform by default (`--no-options-in-question` turns it off),
and an adapter trained with it needs it at serving time. `selfjev eval`, `selfjev classify` and the Python `classify()`
score their input as given: evaluate on the test-set copies that carry the lists (`data/ova/`).

## Qwen3.5: the tree for a hybrid model

`selfjev-4b`'s base, Qwen3.5-4B, is mostly Gated DeltaNet: three recurrent layers (a gated linear recurrence with a
short convolution) for every full-attention layer. A tree mask cannot hide one branch from its siblings inside a
recurrence, so `selfjev/engine/tree.py` runs the same packed tree two ways:

- **attention layers** read it through the tree mask, as above;
- **DeltaNet layers** run level by level: the text, then every question from the text's final recurrent and
  convolution state, then every candidate from its question's state. Gradients flow back through those copied states.

Each candidate still scores exactly like the standalone `text + question + candidate` sequence (tested in fp32 on a
tiny model, and on the real model: scores within 0.004, gradient cosine 0.99997 in fp32). Training this way reads the
text once per state, so texts up to 8K tokens fit where full-sequence training had to stop at 2K; that alone was worth
+1.1 on eval2 (`qwen35_4b_tree`), and `selfjev-4b` trains on texts up to 16K. Serving: `TreeServer`
(`selfjev.engine.tree`: the same tree, forward only; not timed on a GPU yet) or vLLM (`selfjev.engine.vllm`), which is
exact but slow with many questions ([speed](speed.md#qwen35-on-vllm-2026-09-25-l40s)). The Qwen3.5 reports so far,
`selfjev-4b`'s included, were scored by a forked-cache engine (the text encoded once, its cache copied for each
branch; the vLLM check aside), now at the tag.

## From scores to typed answers

| type | probability | decision |
|---|---|---|
| binary | `p_yes = sigmoid(s / T)` | `p_yes ≥ threshold` (request → validation-selected → 0.5; the source is recorded) |
| multiclass | `softmax(scores / T)` within the question | argmax (exact ties broken by id, never by input order); optional `abstain_below` → `selected: null` |
| multilabel | `sigmoid(scores / T)` per candidate, no sum constraint | every candidate with `p ≥ threshold` |

`T = 1` and the status is `uncalibrated` unless a calibration file fitted on the held-out calibration split supplies a
temperature (`heldout_temperature_scaled`). A multiclass argmax always picks something: add an explicit "none of the
above" candidate or set `abstain_below`. For the tree models, serve with the default thresholds: the fitted ones cost
accuracy ([findings](findings.md#distillation-and-calibration)).

## Request and response

The request format is in [examples/request.json](../examples/request.json). Validation rejects duplicate question or
candidate ids, empty texts, questions or candidate lists, a multiclass question with fewer than 2 candidates,
non-numeric or out-of-range thresholds, and unknown fields. Input longer than `--max-length` (prompt, text and the
longest question with its candidate; default 32,768 tokens) raises `InputTooLong` (HTTP 422 from the server): nothing
is ever truncated.

Each answered question carries `id`, `type`, raw `score`(s), probabilities, `selected` (bool, id or list of ids),
`threshold` and `threshold_source`, and `calibration`. `meta` records the model and pinned revision, adapter path and
sha256, prompt name and sha, device, dtype, `max_length`, pair and token counts, and timings.

## Decisions-API-shaped endpoint

`selfjev serve` answers Jev's decisions API at `POST /v1/systemone`, `/api/alpha/decisions` and `/v1/decisions`: the
same body (`state` plus `questions: {id: {type: noul | choice | score | multi, instructions, criteria}}`) and
`answers` in the same shape, so client code only changes its base URL ([API](api.md)). It is the same shape but a
**different model**, so its probabilities are not interchangeable with Jev's; the response's `model` names the model
that answered. `POST /classify` serves our native schema. The mapping (`selfjev/server/compat.py`):

| type | criteria | our computation | answer |
|---|---|---|---|
| `choice` | `{key: description}` | multiclass over `"key: description"` (the key alone when the description is null) | `choice`, `probabilities`, `confidence` |
| `noul` | `{"true": …, "false": …}` | 2-way choice between the two descriptions | `noul` = P(true) |
| `noul` | none | our binary yes/no | `noul` = p_yes |
| `score` | `[level_0, …, level_k]` (ordered) | multiclass over the levels | `probabilities` per level; `score` = Σ i·pᵢ in level units (0 to k) |
| `multi` | `{key: description}` | multilabel: an independent yes/no per option | every option with p ≥ 0.5 |
