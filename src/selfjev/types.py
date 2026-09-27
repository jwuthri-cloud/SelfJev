"""The wire format of the decisions API (docs/api.md): Jev's request and answers, plus `multi`. Shared by the SDK
(selfjev.client) and the server (selfjev.server), so both validate exactly the same thing. Needs only pydantic.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 64
MAX_OPTIONS = 255


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1)
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
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
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
    criteria: dict[Literal["true", "false"], str] | None = None

    @field_validator("criteria")
    @classmethod
    def _both(cls, v):
        if v is not None and set(v) != {"true", "false"}:
            raise ValueError('noul criteria must have both "true" and "false"')
        return v


class Choice(_Question):
    """Exactly one option: key -> description (or None to use the key itself)."""

    type: Literal["choice"] = "choice"
    criteria: dict[str, str | None]

    @field_validator("criteria")
    @classmethod
    def _n(cls, v):
        return _options(v, 2)


class Score(_Question):
    """A position on an ordered scale of 2 to 10 described levels, lowest first."""

    type: Literal["score"] = "score"
    criteria: list[str]

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
    criteria: dict[str, str | None]

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
