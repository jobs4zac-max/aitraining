"""The three LangChain tools the agent routes between.

Docstrings here are not documentation for humans -- they are the tool
descriptions the model reads to decide what to call and when, so they state the
workflow order explicitly.
"""

from __future__ import annotations

from langchain_core.tools import tool

from . import turn_state
from .config import get_llm
from .logging_setup import get_logger, timed
from .schemas import RetrievalEvaluation
from .search_providers import combined_search
from .vector_store import format_results, search

logger = get_logger("udaplay.tools")

WORKFLOW_VIOLATION = (
    "WORKFLOW ERROR: game_web_search cannot be called yet. The required order is "
    "retrieve_game, then evaluate_retrieval, and only then game_web_search if "
    "evaluate_retrieval returned is_sufficient=false.\n\n"
    "Do this now: call retrieve_game with the user's question, pass its output to "
    "evaluate_retrieval, and if the result is insufficient call game_web_search again."
)

EVALUATOR_SYSTEM_PROMPT = """You judge whether retrieved context is enough to answer a question about video games.

Be strict. Mark is_sufficient = false when any of these hold:
- The context does not mention the specific game, studio or fact being asked about.
- It names a related but different title (for example a prequel when the sequel was asked about).
- The question asks about the present, the future, or recent events, and the context is only historical records.
- It answers part of the question but omits something explicitly requested.

Mark is_sufficient = true only when the context on its own fully and directly answers the question.
Set confidence_score to how well the context answers the question: 0.0 is irrelevant, 1.0 is a complete answer.

One exception. If the question asks what the internal catalogue itself contains -- "which
racing games do you have", "what Nintendo titles are in the database", "list your
PlayStation games" -- judge it only against the records that were retrieved, and mark
is_sufficient = true when those records answer it. A web search cannot report the contents
of this database, so never route such a question to the web for lack of completeness."""


@tool
def retrieve_game(query: str, platform: str = "", publisher: str = "") -> str:
    """Search the internal UdaPlay game database for games matching a query.

    ALWAYS call this first for any game-related question. Returns the closest
    matching games with Title, Platform, Genre, Publisher, Release Date and
    Description.

    Args:
        query: What to look for, e.g. "Pokemon Red release platform".
        platform: Optional filter, e.g. "PlayStation" or "Game Boy".
        publisher: Optional filter, e.g. "Nintendo" or "Rockstar".
    """
    turn_state.record("retrieve_game")

    with timed() as elapsed:
        results = search(
            query,
            k=4,
            platform=platform.strip() or None,
            publisher=publisher.strip() or None,
        )

    logger.info(
        "retrieve_game query=%r platform=%r publisher=%r -> %d hits (%.0f ms) best=%s",
        query[:80],
        platform or None,
        publisher or None,
        len(results),
        elapsed.ms,
        f"{results[0][1]:.4f}" if results else "n/a",
    )
    return format_results(results)


@tool
def evaluate_retrieval(query: str, retrieved_docs: str) -> dict:
    """Judge whether internal search results are enough to answer the question.

    ALWAYS call this immediately after retrieve_game, passing its output
    verbatim as retrieved_docs. Returns confidence_score (0.0-1.0),
    is_sufficient (boolean) and reasoning.

    If is_sufficient is false, you MUST call game_web_search next.

    Args:
        query: The user's original question.
        retrieved_docs: The exact text returned by retrieve_game.
    """
    turn_state.record("evaluate_retrieval")
    evaluator = get_llm(temperature=0.0).with_structured_output(RetrievalEvaluation)

    with timed() as elapsed:
        verdict = evaluator.invoke(
            [
                ("system", EVALUATOR_SYSTEM_PROMPT),
                (
                    "human",
                    f"Question:\n{query}\n\nRetrieved context:\n{retrieved_docs}\n\n"
                    "Is this context sufficient to answer the question?",
                ),
            ]
        )

    logger.info(
        "evaluate_retrieval query=%r -> sufficient=%s confidence=%.2f (%.0f ms) reason=%r",
        query[:80],
        verdict.is_sufficient,
        verdict.confidence_score,
        elapsed.ms,
        verdict.reasoning[:120],
    )
    return verdict.model_dump()


@tool
def game_web_search(query: str) -> str:
    """Search the public web (Wikipedia and DuckDuckGo) for game information.

    Call this ONLY when evaluate_retrieval returned is_sufficient = false --
    that is, when the internal database lacks the answer, covers a different
    title, or is too old for a question about current or recent events.

    Returns snippets labelled [Wikipedia] or [DuckDuckGo] with source URLs.
    Cite those URLs in your answer.

    Args:
        query: A focused web search query, e.g. "God of War Ragnarok release date".
    """
    # The gate: refuse rather than serve results out of order. Returning
    # instructions (not raising) lets the agent correct itself and retry.
    if not turn_state.was_called("evaluate_retrieval"):
        logger.warning(
            "game_web_search blocked: evaluate_retrieval has not run this turn "
            "(tools so far: %s)",
            sorted(turn_state.called_tools()) or "none",
        )
        return WORKFLOW_VIOLATION

    with timed() as elapsed:
        result = combined_search(query)

    # "Unavailable" means a backend degraded rather than raised -- worth a
    # warning, because the answer that follows will be thinner than it looks.
    level = logger.warning if "Unavailable" in result else logger.info
    level(
        "game_web_search query=%r -> %d chars (%.0f ms)%s",
        query[:80],
        len(result),
        elapsed.ms,
        " [a backend was unavailable]" if "Unavailable" in result else "",
    )
    return result


ALL_TOOLS = [retrieve_game, evaluate_retrieval, game_web_search]
