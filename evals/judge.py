"""The DeepEval judge model.

DeepEval defaults to calling ``api.openai.com``, which a ``voc-`` key cannot
reach. ``OpenAIModel`` accepts ``base_url`` directly, so the judge is built from
the same resolved configuration as the agent -- no custom
``DeepEvalBaseLLM`` subclass required.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from udaplay.config import get_api_key, get_base_url, get_chat_model, load_env  # noqa: E402


@lru_cache(maxsize=1)
def get_judge():
    """Judge LLM for every metric. Cached so metrics share one client.

    Note that judge and agent share a model (``gpt-4o-mini``). That is a real
    limitation -- a model grading its own output is more forgiving than an
    independent judge -- but the Vocareum proxy is the only endpoint available.
    Treat the scores as a regression signal, not an absolute quality measure.
    """
    from deepeval.models import OpenAIModel

    load_env()
    return OpenAIModel(
        model=get_chat_model(),
        api_key=get_api_key(),
        base_url=get_base_url(),
        temperature=0.0,
    )
