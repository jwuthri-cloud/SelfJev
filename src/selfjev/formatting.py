"""Official Qwen3-Reranker prompt template plus our question -> (instruction, query, document) mappings.

PREFIX/SUFFIX and pair_text() are copied verbatim from the Qwen3-Reranker-0.6B model card's Transformers
reference (identical to the repo's chat_template.jinja). Only the PROMPTS mappings are ours. Every run
records the prompt name and prompt_sha(); editing a mapping changes its sha, so give it a new name.
"""
import hashlib

from .schemas import Question

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EVIDENCE = "Judge whether the proposed answer to the question is correct, using only the Document as evidence."
BINARY_ANSWER = "Yes"  # binary p_yes = P(the Document supports answering "Yes")

# name -> f(question, answer, is_binary) -> (instruct, query). The state is always the Document.
PROMPTS = {
    # Fixed instruction; question and proposed answer both in the query.
    "answer-v1": lambda q, a, b: (EVIDENCE, f"Question: {q}\nProposed answer: {a}"),
    # Question as the instruction, candidate description as the query; binary asks the question as the query.
    "task-v1": lambda q, a, b: ("Answer the question in the Query using only the Document.", q) if b else (q, a),
    # Question quoted inside the instruction, candidate (or "Yes") as the query.
    "task-v2": lambda q, a, b: (f'Judge whether the Document supports answering the question "{q}" with the answer given in the Query.', a),
    # task-v1 for binary, task-v2 for candidates.
    "hybrid-v1": lambda q, a, b: ("Answer the question in the Query using only the Document.", q) if b else
    (f'Judge whether the Document supports answering the question "{q}" with the answer given in the Query.', a),
}
DEFAULT_PROMPT = "task-v1"  # selected on validation splits only; see reports/prompt_selection.md


def pair_text(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def answers(q: Question) -> list[str]:
    """One proposed answer per scored pair: ['Yes'] for binary, candidate descriptions otherwise."""
    return [BINARY_ANSWER] if q.type == "binary" else [c.description for c in q.candidates]


def question_pairs(q: Question, state: str, prompt: str = DEFAULT_PROMPT) -> list[str]:
    f = PROMPTS[prompt]
    return [pair_text(*f(q.instruction, a, q.type == "binary"), state) for a in answers(q)]


def prompt_sha(prompt: str = DEFAULT_PROMPT) -> str:
    f = PROMPTS[prompt]
    rendered = [pair_text(*f("{q}", BINARY_ANSWER, True), "{d}"), pair_text(*f("{q}", "{a}", False), "{d}")]
    return hashlib.sha256("\x1f".join([PREFIX, SUFFIX, *rendered]).encode()).hexdigest()[:12]
