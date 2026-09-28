"""selfjev: an open decisions model (selfjev-4b) with Jev's API.

The SDK is importable without torch: `from selfjev import SelfJev, Noul, Choice, Score, Multi`. Serving, training and
evaluation live in selfjev.engine, selfjev.server, selfjev.training and selfjev.evaluation (extras: serve, train).
"""

from importlib.metadata import version as _version

from .client import (
    APIConnectionError,
    APIError,
    AsyncSelfJev,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    OverloadedError,
    RateLimitError,
    SelfJev,
    SelfJevError,
)
from .types import Choice, DecisionRequest, DecisionResponse, FileObject, FineTuningJob, JobEvent, Multi, Noul, Score, TrainingRow

__version__ = _version("selfjev")  # one source: pyproject.toml
__all__ = [
    "APIConnectionError",
    "APIError",
    "AsyncSelfJev",
    "AuthenticationError",
    "Choice",
    "DecisionRequest",
    "DecisionResponse",
    "FileObject",
    "FineTuningJob",
    "InvalidRequestError",
    "JobEvent",
    "Multi",
    "NotFoundError",
    "Noul",
    "OverloadedError",
    "RateLimitError",
    "Score",
    "SelfJev",
    "SelfJevError",
    "TrainingRow",
]
