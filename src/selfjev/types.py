"""The wire format of the decisions API (docs/api.md): Jev's request and answers, plus `multi`. Shared by the SDK
(selfjev.client) and the server (selfjev.server), so both validate exactly the same thing. Needs only pydantic.
"""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 64
MAX_OPTIONS = 255


def _as_text(v):
    """Jev accepts JSON (an object or an array) wherever it takes text; the model reads it as JSON text, like `state`."""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else v


Text = Annotated[str, BeforeValidator(_as_text)]


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: Text | None = Field(default=None, description="optional, as in Jev: without it the criteria carry the question")
    weight: float | None = Field(default=None, description="accepted for compatibility with Jev's SDK; not used")

    def __init__(self, instructions: str | None = None, criteria: Any = None, /, **data):
        """Positional shorthand: Choice("Which team?", {"billing": "…", "tech": "…"})."""
        if instructions is not None:
            data["instructions"] = instructions
        if criteria is not None:
            data["criteria"] = criteria
        super().__init__(**data)

    @field_validator("instructions")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("instructions must not be blank")
        return v


def _options(v: dict[str, str | None], low: int) -> dict[str, str | None]:
    if not low <= len(v) <= MAX_OPTIONS:
        raise ValueError(f"criteria needs {low} to {MAX_OPTIONS} options, got {len(v)}")
    if any(not k.strip() for k in v):
        raise ValueError("option keys must not be blank")
    return v


class Noul(_Question):
    """Yes/no: the answer is the probability of yes (or of the `true` description)."""

    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], Text | None] | None = None

    @field_validator("criteria")
    @classmethod
    def _described(cls, v):
        """Either side may be left out or null, as in Jev; with neither described there are no criteria."""
        v = {k: d for k, d in (v or {}).items() if d is not None}
        return v or None

    @model_validator(mode="after")
    def _asks_something(self):
        if self.instructions is None and self.criteria is None:
            raise ValueError("a noul needs instructions or criteria")
        return self


class Choice(_Question):
    """Exactly one option: key -> description (or None to use the key itself)."""

    type: Literal["choice"] = "choice"
    criteria: dict[str, Text | None]

    @field_validator("criteria")
    @classmethod
    def _n(cls, v):
        return _options(v, 2)


class Score(_Question):
    """A position on an ordered scale of 2 to 10 described levels, lowest first."""

    type: Literal["score"] = "score"
    criteria: list[Text]

    @field_validator("criteria")
    @classmethod
    def _levels(cls, v):
        if not 2 <= len(v) <= 10:
            raise ValueError(f"score criteria needs 2 to 10 levels, got {len(v)}")
        if len(set(v)) != len(v) or any(not x.strip() for x in v):
            raise ValueError("score levels must be distinct and not blank")
        return v


class Multi(_Question):
    """Every option that applies, possibly none (a selfjev extension; Jev asks one noul per option)."""

    type: Literal["multi"] = "multi"
    criteria: dict[str, Text | None]

    @field_validator("criteria")
    @classmethod
    def _n(cls, v):
        return _options(v, 1)


Question = Annotated[Noul | Choice | Score | Multi, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown top-level fields are ignored, as Jev does

    model: str = "selfjev-4b"
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _limits(self):
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"questions needs 1 to {MAX_QUESTIONS} entries, got {len(self.questions)}")
        if isinstance(self.state, str) and not self.state.strip():
            raise ValueError("state must not be blank")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


class MultiAnswer(BaseModel):
    type: Literal["multi"] = "multi"
    multi: list[str]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")  # servers may add fields (e.g. OpenRouter's provider)

    id: str | None = None
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)

    def _of(self, kind):
        return {k: a for k, a in self.answers.items() if isinstance(a, kind)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return self._of(NoulAnswer)

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return self._of(ChoiceAnswer)

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return self._of(ScoreAnswer)

    @property
    def multis(self) -> dict[str, MultiAnswer]:
        return self._of(MultiAnswer)


class ErrorDetail(BaseModel):
    type: str
    message: str
    param: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def confidence(probabilities: list[float]) -> float:
    """(K * p_max - 1) / (K - 1): 1 when all probability is on one option, 0 when it is uniform (Jev's definition)."""
    k = len(probabilities)
    return max(0.0, min(1.0, (k * max(probabilities) - 1) / (k - 1))) if k > 1 else 1.0


# --- fine-tuning (docs/api.md, "Fine-tuning") ------------------------------------------------------------------------


class TrainingRow(BaseModel):
    """One line of a fine-tuning file: a decisions request plus the expected answer of every question
    (noul: true/false, choice: a key, score: a level index, multi: a list of keys)."""

    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]
    answers: dict[str, bool | int | str | list[str]]

    @model_validator(mode="after")
    def _answers_match(self):
        if set(self.answers) != set(self.questions):
            raise ValueError(f"answers must cover exactly the questions: {sorted(set(self.questions) ^ set(self.answers))}")
        for qid, q in self.questions.items():
            a = self.answers[qid]
            ok = {
                "noul": isinstance(a, bool),
                "choice": isinstance(a, str) and a in (getattr(q, "criteria", None) or {}),
                "score": isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(getattr(q, "criteria", None) or []),
                "multi": isinstance(a, list) and set(a) <= set(getattr(q, "criteria", None) or {}),
            }[q.type]
            if not ok:
                raise ValueError(f"answer to '{qid}' does not fit a {q.type} question: {a!r}")
        return self


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    rows: int
    questions: int


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=1, ge=1, le=10)
    learning_rate: float | None = Field(default=None, gt=0, le=1e-2)
    reward: dict[Literal["log", "brier", "spherical", "accuracy", "confident_miss"], float] | None = None  # rlcd only
    samples: int | None = Field(default=None, ge=2, le=64)
    sigma: float | None = Field(default=None, gt=0, le=2)
    beta: float | None = Field(default=None, ge=0, le=10)


class Method(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["supervised", "rlcd"] = "supervised"
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)


class FineTuningJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "selfjev-4b"
    training_file: str
    validation_file: str | None = None
    suffix: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    method: Method = Field(default_factory=Method)


class FineTuningJob(BaseModel):
    id: str
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    model: str
    fine_tuned_model: str | None = None
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: int
    finished_at: int | None = None
    training_file: str
    validation_file: str | None = None
    method: Method
    error: str | None = None


class JobEvent(BaseModel):
    object: Literal["fine_tuning.job.event"] = "fine_tuning.job.event"
    created_at: int
    level: Literal["info", "error"] = "info"
    message: str
    data: dict[str, Any] | None = None
