"""Open-source web-search backends: Wikipedia and DuckDuckGo.

Both are keyless, which is why they were chosen over Tavily. Neither is
reliable in isolation, so they are used together:

  * Wikipedia is authoritative for settled facts (release dates, publishers)
    but knows nothing about "what is studio X working on right now".
  * DuckDuckGo covers recent news but is an unofficial endpoint that returns
    HTTP 429 under repeated calls.

Every function here returns a string and never raises. A demo notebook that
dies on a transient 429 is worse than one that reports the outage inline.
"""

from __future__ import annotations

import os
import time

import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia rejects requests without a descriptive User-Agent. This is the
# reason the `wikipedia` PyPI package no longer works.
USER_AGENT = "UdaPlay/0.1 (educational gaming research agent)"

REQUEST_TIMEOUT = 20


def _max_results() -> int:
    try:
        return int(os.getenv("SEARCH_MAX_RESULTS", "4"))
    except ValueError:
        return 4


def wikipedia_search(query: str, max_results: int | None = None) -> str:
    """Search Wikipedia and return intro extracts for the top pages.

    Uses the MediaWiki API directly: search for titles, then fetch each page's
    plain-text intro, which is where release dates and publishers live.
    """
    limit = max_results or _max_results()
    headers = {"User-Agent": USER_AGENT}

    try:
        found = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        found.raise_for_status()
        titles = [hit["title"] for hit in found.json()["query"]["search"]]

        if not titles:
            return "[Wikipedia] No matching articles."

        pages = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        pages.raise_for_status()

        blocks = []
        for page in pages.json()["query"]["pages"].values():
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            title = page["title"]
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            blocks.append(f"[Wikipedia] {title} ({url})\n{extract[:1200]}")

        return "\n\n".join(blocks) if blocks else "[Wikipedia] No usable extracts."

    except Exception as exc:  # noqa: BLE001 - degrade, never break the notebook
        return f"[Wikipedia] Unavailable ({type(exc).__name__}: {exc})."


def duckduckgo_search(
    query: str, max_results: int | None = None, retries: int = 2
) -> str:
    """Search DuckDuckGo via LangChain's community wrapper.

    Retries with backoff because the endpoint rate-limits aggressively; on
    exhaustion it reports the outage rather than raising.
    """
    limit = max_results or _max_results()

    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    except ImportError as exc:
        return f"[DuckDuckGo] Unavailable (missing dependency: {exc})."

    tool = DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=limit),
        output_format="list",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            results = tool.invoke(query)
            if not results:
                return "[DuckDuckGo] No results."
            blocks = [
                f"[DuckDuckGo] {item.get('title', 'Untitled')} ({item.get('link') or item.get('href', 'no url')})\n"
                f"{(item.get('snippet') or item.get('body') or '').strip()[:600]}"
                for item in results
            ]
            return "\n\n".join(blocks)
        except Exception as exc:  # noqa: BLE001 - includes rate-limit responses
            last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))

    return (
        f"[DuckDuckGo] Unavailable after {retries + 1} attempts "
        f"({type(last_error).__name__}: {last_error}). Likely rate limited."
    )


def combined_search(query: str, max_results: int | None = None) -> str:
    """Query both backends and merge, keeping the source label on each block."""
    sections = [
        wikipedia_search(query, max_results=max_results),
        duckduckgo_search(query, max_results=max_results),
    ]

    usable = [s for s in sections if "Unavailable" not in s and "No results" not in s]
    if not usable:
        return (
            "Web search returned nothing usable. Diagnostics:\n\n"
            + "\n\n".join(sections)
            + "\n\nIf both are unavailable, outbound network access is likely blocked."
        )

    return "\n\n".join(sections)
