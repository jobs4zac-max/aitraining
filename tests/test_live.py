"""Tests that touch the network or the OpenAI-compatible endpoint.

Deselected by default. Run them explicitly once a key is in place:

    uv run pytest -m live
"""

from __future__ import annotations

import pytest

from udaplay.search_providers import combined_search, duckduckgo_search, wikipedia_search

pytestmark = pytest.mark.live


class TestWebSearch:
    """These need network access but no API key."""

    def test_wikipedia_finds_a_known_release_date(self):
        result = wikipedia_search("God of War Ragnarok", max_results=1)
        assert "[Wikipedia]" in result
        assert "2022" in result
        assert "en.wikipedia.org" in result

    def test_duckduckgo_returns_labelled_snippets(self):
        result = duckduckgo_search("God of War Ragnarok release date", max_results=2)
        # A rate-limited run is an environment problem, not a code failure.
        if "Unavailable" in result:
            pytest.skip(f"DuckDuckGo unavailable: {result[:120]}")
        assert "[DuckDuckGo]" in result

    def test_combined_search_never_raises_on_nonsense(self):
        result = combined_search("zzzqqq not a real game title 12345")
        assert isinstance(result, str) and result


class TestEndpoint:
    """These need a valid OPENAI_API_KEY and a reachable base URL."""

    def test_smoke_test_passes(self):
        from udaplay.config import smoke_test

        assert smoke_test(), "endpoint check failed - see printed diagnostics"

    def test_embeddings_have_the_expected_dimension(self):
        from udaplay.config import get_embeddings

        vector = get_embeddings().embed_query("Pokemon Red")
        # text-embedding-3-small is 1536-dimensional; a different size means the
        # proxy substituted another model.
        assert len(vector) == 1536


class TestRetrieval:
    """Needs a key and a built index (`python -m udaplay index`)."""

    def test_internal_query_retrieves_the_right_game(self):
        from udaplay.vector_store import search

        results = search("When was Pokemon Red launched and on what platform?", k=3)
        names = [doc.metadata["name"].lower() for doc, _ in results]
        assert "pokemon red" in names

    def test_publisher_filter_restricts_results(self):
        from udaplay.vector_store import search

        results = search("open world game", k=3, publisher="Rockstar")
        assert results
        assert all("rockstar" in doc.metadata["publisher"].lower() for doc, _ in results)


class TestAgentRouting:
    """The rubric's routing behaviour. Needs a key, an index and network."""

    def test_internal_match_does_not_hit_the_web(self):
        from udaplay.agent import ask, reset_session, used_web_search

        reset_session("test-internal")
        result = ask(
            "When was Pokemon Red launched, and on what platform?",
            session_id="test-internal",
            show_trace=False,
        )
        assert not used_web_search(result)
        assert "1996" in result["output"]
        assert "Internal Game Database" in result["output"]

    def test_missing_game_triggers_the_web_fallback(self):
        from udaplay.agent import ask, reset_session, used_web_search

        reset_session("test-fallback")
        result = ask(
            "When was God of War Ragnarok released?",
            session_id="test-fallback",
            show_trace=False,
        )
        assert used_web_search(result), "expected fallback for a game absent from the DB"
        assert "2022" in result["output"]

    def test_history_resolves_a_pronoun_follow_up(self):
        from udaplay.agent import ask, build_stateful_agent, reset_session

        agent = build_stateful_agent(verbose=False)
        reset_session("test-history")
        ask(
            "Tell me about Gran Turismo.",
            session_id="test-history",
            agent=agent,
            show_trace=False,
        )
        follow_up = ask(
            "What platform was it on?",
            session_id="test-history",
            agent=agent,
            show_trace=False,
        )
        assert "PlayStation" in follow_up["output"]
