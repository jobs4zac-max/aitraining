"""Environment and client configuration for UdaPlay.

The same code has to run in three places:

  * locally, against a Vocareum ``voc-...`` key served by Vocareum's own
    OpenAI-compatible proxy;
  * inside hosted JupyterLab, where that key is the only one available;
  * against a stock ``sk-...`` OpenAI key.

The difference between them is entirely the base URL, so that is resolved once
here and every client is built through :func:`get_llm` / :func:`get_embeddings`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and index locations are overridable so the app can run where the
# application directory is not a good place to write -- Cloud Run, for one,
# backs the container filesystem with memory, so writes belong in /tmp.
#
# These are read from the process environment at import time, *before*
# `load_env()` has run, so they must be set as real environment variables
# (a Dockerfile `ENV`, or a Cloud Run variable) rather than in `.env`.
GAMES_DIR = Path(os.getenv("UDAPLAY_GAMES_DIR") or PROJECT_ROOT / "data" / "games")
INDEX_DIR = Path(os.getenv("UDAPLAY_INDEX_DIR") or PROJECT_ROOT / "faiss_index_udaplay")

_loaded = False


def load_env(force: bool = False) -> None:
    """Load keys from ``.env``, then ``config.env``.

    Both are supported because the project requirements name ``config.env``
    while local development uses ``.env``. ``load_dotenv`` does not override
    already-set variables, so loading ``.env`` first makes it the winner when a
    variable appears in both.

    Also normalises the base URL: an explicit ``OPENAI_BASE_URL`` always wins,
    otherwise a ``voc-`` prefixed key implies the Vocareum proxy.
    """
    global _loaded
    if _loaded and not force:
        return

    for name in (".env", "config.env"):
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            load_dotenv(candidate, override=force)

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "").strip()

    if not base_url and key.startswith("voc-"):
        base_url = VOCAREUM_BASE_URL

    if base_url:
        # The two libraries read different variable names: the `openai` SDK
        # reads OPENAI_BASE_URL, langchain-openai reads OPENAI_API_BASE. Set
        # both, and pass base_url= explicitly in the factories below -- a silent
        # fallback to api.openai.com with a voc- key surfaces as a bare 401.
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    _loaded = True


def get_base_url() -> str | None:
    """Resolved base URL, or ``None`` to let the client use stock OpenAI."""
    load_env()
    return (os.getenv("OPENAI_BASE_URL") or "").strip() or None


def get_api_key() -> str:
    load_env()
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def get_chat_model() -> str:
    load_env()
    return os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)


def get_embedding_model() -> str:
    load_env()
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def assert_env() -> None:
    """Fail fast and loudly if the environment is not usable.

    Mirrors the verification block in the project requirements.
    """
    load_env()
    assert os.getenv("OPENAI_API_KEY") is not None, (
        "OPENAI_API_KEY missing -- copy .env.example to .env and add your key"
    )
    key = get_api_key()
    assert key, "OPENAI_API_KEY is set but empty"

    # Catch the common case of copying .env.example without editing it. Left
    # unchecked this reaches the API and comes back as an opaque 401.
    assert key.rstrip(".") not in {"voc-", "sk-"}, (
        f"OPENAI_API_KEY is still the placeholder from .env.example ({mask(key)}) "
        "-- replace it with your real key"
    )


def mask(secret: str, keep: int = 6) -> str:
    """``voc-2981234abcd`` -> ``voc-29...abcd``. Safe to print in a notebook."""
    if not secret:
        return "<missing>"
    if len(secret) <= keep + 4:
        return f"{secret[:keep]}..."
    return f"{secret[:keep]}...{secret[-4:]}"


def describe_env() -> dict[str, str]:
    """Print and return the resolved configuration, with the key masked."""
    load_env()
    key = get_api_key()
    info = {
        "api_key": mask(key),
        "key_flavour": "vocareum" if key.startswith("voc-") else "openai",
        "base_url": get_base_url() or "https://api.openai.com/v1 (default)",
        "chat_model": get_chat_model(),
        "embedding_model": get_embedding_model(),
        "games_dir": str(GAMES_DIR),
        "index_dir": str(INDEX_DIR),
    }
    width = max(len(k) for k in info)
    print("UdaPlay environment")
    print("-" * (width + 40))
    for label, value in info.items():
        print(f"{label:<{width}}  {value}")
    return info


def get_llm(temperature: float = 0.0, **kwargs):
    """Chat model used for both the agent and the retrieval evaluator."""
    from langchain_openai import ChatOpenAI

    assert_env()
    return ChatOpenAI(
        model=get_chat_model(),
        temperature=temperature,
        api_key=get_api_key(),
        base_url=get_base_url(),
        **kwargs,
    )


def get_embeddings(**kwargs):
    """Embedding model used to build and query the FAISS index."""
    from langchain_openai import OpenAIEmbeddings

    assert_env()
    return OpenAIEmbeddings(
        model=get_embedding_model(),
        api_key=get_api_key(),
        base_url=get_base_url(),
        **kwargs,
    )


def smoke_test() -> bool:
    """One embedding call and one chat call through the resolved endpoint.

    Worth running first in a fresh environment: it distinguishes a bad key from
    a proxy that does not serve the requested model, which otherwise only shows
    up much later as a confusing failure mid-pipeline.
    """
    load_env()
    ok = True

    try:
        vector = get_embeddings().embed_query("Pokemon Red")
        print(f"[ok]   embeddings '{get_embedding_model()}' -> dim {len(vector)}")
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        ok = False
        print(f"[FAIL] embeddings '{get_embedding_model()}': {exc}")

    try:
        reply = get_llm().invoke("Reply with the single word: ready")
        print(f"[ok]   chat '{get_chat_model()}' -> {reply.content.strip()!r}")
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        ok = False
        print(f"[FAIL] chat '{get_chat_model()}': {exc}")

    return ok
