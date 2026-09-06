"""Tool-order enforcement tests. No network, no API key.

Regression cover for an observed failure: asked "What is Rockstar Games working
on right now?", the agent called `game_web_search` directly and skipped both
`retrieve_game` and `evaluate_retrieval`. The answer was correct, so a check on
the answer alone would not have caught it -- the mandated two-tier workflow had
simply not run.
"""

from __future__ import annotations

import pytest

from udaplay import turn_state
from udaplay.tools import WORKFLOW_VIOLATION, game_web_search


@pytest.fixture(autouse=True)
def fresh_turn():
    turn_state.begin_turn()
    yield
    turn_state.begin_turn()


def test_turn_starts_with_no_tools_recorded():
    assert turn_state.called_tools() == frozenset()


def test_recording_is_cumulative_within_a_turn():
    turn_state.record("retrieve_game")
    turn_state.record("evaluate_retrieval")
    assert turn_state.was_called("retrieve_game")
    assert turn_state.was_called("evaluate_retrieval")


def test_begin_turn_clears_previous_state():
    turn_state.record("evaluate_retrieval")
    turn_state.begin_turn()
    assert not turn_state.was_called("evaluate_retrieval")


def test_web_search_is_blocked_before_evaluation():
    """The gate must fire without any network call being attempted."""
    result = game_web_search.invoke({"query": "What is Rockstar working on right now?"})
    assert result == WORKFLOW_VIOLATION


def test_blocked_message_names_the_required_tools():
    """The refusal is read by the model, so it has to be actionable."""
    result = game_web_search.invoke({"query": "anything"})
    assert "retrieve_game" in result
    assert "evaluate_retrieval" in result


def test_retrieve_game_alone_does_not_open_the_gate():
    """Retrieval without evaluation is exactly the skip we are guarding."""
    turn_state.record("retrieve_game")
    assert game_web_search.invoke({"query": "anything"}) == WORKFLOW_VIOLATION


def test_gate_opens_once_evaluation_has_run(monkeypatch):
    monkeypatch.setattr(
        "udaplay.tools.combined_search", lambda query: "[Wikipedia] stub result"
    )
    turn_state.record("retrieve_game")
    turn_state.record("evaluate_retrieval")

    result = game_web_search.invoke({"query": "God of War Ragnarok release date"})
    assert result == "[Wikipedia] stub result"


def test_state_recorded_inside_a_tool_survives_the_invocation():
    """The regression that broke the first version of this gate.

    LangChain runs each tool inside `copy_context()`, so a ContextVar written
    inside a tool is discarded on return -- the gate then blocked every call
    and the agent looped to max_iterations. State must survive across separate
    `.invoke()` calls, so it is asserted from outside the tool.
    """
    from langchain_core.tools import tool

    @tool
    def marker(value: str) -> str:
        """Record a marker tool call."""
        turn_state.record("evaluate_retrieval")
        return "recorded"

    marker.invoke({"value": "x"})
    assert turn_state.was_called("evaluate_retrieval"), (
        "state written inside a tool did not survive the invocation"
    )


def test_gate_opens_when_evaluation_ran_via_the_real_tool(monkeypatch):
    """End-to-end gate check driving both tools through `.invoke()`."""
    from udaplay.schemas import RetrievalEvaluation
    from udaplay.tools import evaluate_retrieval

    class _StubEvaluator:
        def invoke(self, _messages):
            return RetrievalEvaluation(
                confidence_score=0.1, is_sufficient=False, reasoning="stub"
            )

    class _StubLLM:
        def with_structured_output(self, _schema):
            return _StubEvaluator()

    monkeypatch.setattr("udaplay.tools.get_llm", lambda **kwargs: _StubLLM())
    monkeypatch.setattr(
        "udaplay.tools.combined_search", lambda query: "[Wikipedia] stub result"
    )

    assert game_web_search.invoke({"query": "anything"}) == WORKFLOW_VIOLATION

    evaluate_retrieval.invoke({"query": "anything", "retrieved_docs": "no match"})

    assert game_web_search.invoke({"query": "anything"}) == "[Wikipedia] stub result"
