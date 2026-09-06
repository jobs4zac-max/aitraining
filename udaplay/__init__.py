"""UdaPlay -- an AI gaming research agent.

Internal knowledge lives in a FAISS vector store built from ``data/games``.
When retrieval is judged insufficient, the agent falls back to keyless web
search (Wikipedia + DuckDuckGo).
"""

from .config import (
    assert_env,
    describe_env,
    get_embeddings,
    get_llm,
    load_env,
    smoke_test,
)
from .schemas import GameRecord, KeyDetails, RetrievalEvaluation, SourceOrigin, UdaPlayAnswer

__version__ = "0.1.0"

__all__ = [
    "GameRecord",
    "KeyDetails",
    "RetrievalEvaluation",
    "SourceOrigin",
    "UdaPlayAnswer",
    "assert_env",
    "describe_env",
    "get_embeddings",
    "get_llm",
    "load_env",
    "smoke_test",
]
