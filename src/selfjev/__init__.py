"""selfjev: an open decisions model (selfjev-4b) with Jev's API.

The SDK is importable without torch: `from selfjev import SelfJev, Noul, Choice, Score, Multi`. Serving, training and
evaluation live in selfjev.engine, selfjev.server, selfjev.training and selfjev.evaluation (extras: serve, train).
"""

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
from .types import Choice, DecisionRequest, DecisionResponse, Multi, Noul, Score

__version__ = "0.2.0"
__all__ = [
    "APIConnectionError",
    "APIError",
    "AsyncSelfJev",
    "AuthenticationError",
    "Choice",
    "DecisionRequest",
    "DecisionResponse",
    "InvalidRequestError",
    "Multi",
    "NotFoundError",
    "Noul",
    "OverloadedError",
    "RateLimitError",
    "Score",
    "SelfJev",
    "SelfJevError",
]
