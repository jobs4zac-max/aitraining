"""Input and output validation.

Two boundaries are guarded:

  * **Input** -- before a query reaches the LLM. Cheap, deterministic checks
    that stop junk from becoming a paid API call, and normalise whitespace so
    the same question does not produce different cache keys.
  * **Output** -- after the agent answers. The system prompt *mandates* a
    ``Source:`` line and Key Details; an LLM will occasionally drop them.
    Checking is how that becomes visible instead of silently degrading.

Output problems are returned as warnings rather than raised: a technically
malformed answer is still more useful to the user than an exception.
"""

from __future__ import annotations

import re
import unicodedata

MAX_QUERY_CHARS = 500
MIN_QUERY_CHARS = 3

# Strip C0/C1 control characters but keep newlines and tabs.
_CONTROL_CHARS = re.compile(r"[^\S\n\t]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class InvalidQuery(ValueError):
    """Raised when a query cannot usefully be sent to the agent."""


def validate_query(text: str | None) -> str:
    """Normalise and check a user query. Returns the cleaned query.

    Raises:
        InvalidQuery: with a message written for the person who typed it.
    """
    if text is None:
        raise InvalidQuery("No question provided.")

    # NFKC folds full-width and other compatibility forms so that a query
    # pasted from a PDF behaves like one that was typed.
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        raise InvalidQuery("Question is empty.")

    if len(cleaned) < MIN_QUERY_CHARS:
        raise InvalidQuery(
            f"Question is too short ({len(cleaned)} characters). "
            f"Please write at least {MIN_QUERY_CHARS}."
        )

    if len(cleaned) > MAX_QUERY_CHARS:
        raise InvalidQuery(
            f"Question is too long ({len(cleaned)} characters, "
            f"limit {MAX_QUERY_CHARS}). Please shorten it."
        )

    if not re.search(r"[A-Za-z0-9]", cleaned):
        raise InvalidQuery("Question contains no letters or digits.")

    return cleaned


def validate_answer(answer: str, used_web: bool) -> list[str]:
    """Check an answer against the mandated output format.

    Returns a list of human-readable warnings; empty means conformant.

    ``used_web`` comes from the execution trace, not from the text, so a
    mismatch between what ran and what was cited is detectable.
    """
    warnings: list[str] = []

    if not answer or not answer.strip():
        return ["Answer is empty."]

    lowered = answer.lower()

    if "source:" not in lowered:
        warnings.append("Answer omits the mandated 'Source:' citation line.")

    claims_internal = "internal game database" in lowered
    claims_web = "web search" in lowered

    if used_web and not claims_web:
        warnings.append(
            "Web search ran but the answer does not cite it -- the citation "
            "understates its sources."
        )
    if not used_web and claims_web:
        warnings.append(
            "Answer cites web search but no web tool ran -- the citation is wrong."
        )
    if not used_web and not claims_internal and "source:" in lowered:
        warnings.append("Answer cites neither the internal database nor web search.")

    if "key details" not in lowered:
        warnings.append("Answer omits the 'Key Details' block.")

    if used_web and "http" not in lowered:
        warnings.append("Answer used web search but lists no source URL.")

    return warnings
