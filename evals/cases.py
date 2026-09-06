"""Turning an agent run into a DeepEval ``LLMTestCase``.

The interesting part is the mapping from LangChain's ``intermediate_steps`` to
DeepEval's ``tools_called``/``expected_tools``, which is what lets
``ToolCorrectnessMetric`` grade the *routing* rather than the prose. Routing is
the actual requirement -- an answer that happens to be right after skipping the
evaluation step has not satisfied the spec.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from udaplay.agent import ask, reset_session  # noqa: E402

RETRIEVE = "retrieve_game"
EVALUATE = "evaluate_retrieval"
WEB_SEARCH = "game_web_search"


@dataclass
class Scenario:
    """One evaluation scenario and the routing it is supposed to produce."""

    key: str
    question: str
    expected_tools: list[str]
    expect_web: bool
    rationale: str
    # Facts the answer must contain, checked literally -- cheap and
    # deterministic, unlike an LLM-judged correctness score.
    must_mention: list[str] = field(default_factory=list)


SCENARIOS: list[Scenario] = [
    Scenario(
        key="internal_match",
        question="When was Pokemon Red launched, and on what platform?",
        expected_tools=[RETRIEVE, EVALUATE],
        expect_web=False,
        rationale="Record exists; evaluation should pass and no web call should happen.",
        must_mention=["1996", "Game Boy"],
    ),
    Scenario(
        key="present_tense_fallback",
        question="What is Rockstar Games working on right now?",
        expected_tools=[RETRIEVE, EVALUATE, WEB_SEARCH],
        expect_web=True,
        rationale="Only historical Rockstar titles are indexed; a present-tense "
        "question cannot be answered from them.",
    ),
    Scenario(
        key="near_miss_fallback",
        question="When was God of War Ragnarok released?",
        expected_tools=[RETRIEVE, EVALUATE, WEB_SEARCH],
        expect_web=True,
        rationale="The index holds God of War (2018) but not Ragnarok, so the "
        "near-miss must be rejected rather than answered from.",
        must_mention=["2022"],
    ),
    Scenario(
        key="metadata_query",
        question="Which racing games do you have, and on which platforms?",
        expected_tools=[RETRIEVE, EVALUATE],
        expect_web=False,
        rationale="Gran Turismo and Super Mario Kart are both indexed, so this is "
        "answerable internally.",
        must_mention=["Gran Turismo"],
    ),
]


def run_scenario(scenario: Scenario, agent=None) -> dict:
    """Execute one scenario and return the raw agent result."""
    session = f"eval-{scenario.key}"
    reset_session(session)
    return ask(scenario.question, session_id=session, agent=agent, show_trace=False)


def to_test_case(scenario: Scenario, result: dict):
    """Build the DeepEval test case from an agent result.

    ``retrieval_context`` is set to the tool outputs the agent actually saw --
    both internal hits and web snippets. Faithfulness then measures whether the
    answer is grounded in what was retrieved, which is the property that
    matters for a RAG agent.
    """
    from deepeval.test_case import LLMTestCase, ToolCall

    record = result["run_record"]

    tools_called = [
        ToolCall(
            name=call.tool,
            input_parameters=call.tool_input
            if isinstance(call.tool_input, dict)
            else {"input": call.tool_input},
            output=call.output_preview,
        )
        for call in record.tool_calls
    ]

    retrieval_context = [
        call.output_preview for call in record.tool_calls if call.tool != EVALUATE
    ] or ["(no context retrieved)"]

    # Ground truth for the citation, derived from the trace rather than left for
    # the judge to infer. Asking gpt-4o-mini to work out "does this context
    # contain web snippets?" produced a confidently wrong verdict -- it reported
    # web snippets in a context holding only internal records, and scored a
    # correctly-cited answer 0.32. Anything computable should not be judged.
    required_source = (
        "Source: Web Search (Wikipedia/DuckDuckGo)"
        if record.used_web_search
        else "Source: Internal Game Database"
    )

    return LLMTestCase(
        name=scenario.key,
        input=scenario.question,
        actual_output=record.answer,
        expected_output=required_source,
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=[ToolCall(name=name) for name in scenario.expected_tools],
        completion_time=record.latency_ms / 1000,
    )


def prose_only(answer: str) -> str:
    """Strip the mandated structured blocks, leaving the prose answer.

    ``AnswerRelevancyMetric`` and ``FaithfulnessMetric`` judge against a generic
    notion of a good answer, which collides with a *required* output format:
    asked which racing games exist, a conforming answer scored 0.29 for
    relevancy because the judge read the mandated Key Details block -- publisher,
    release year -- as "irrelevant details", and faithfulness dropped because it
    read one combined block covering two games as conflating them.

    Both complaints are about the format, which is already checked exactly by
    ``validate_answer`` and judged by the citation metric. So relevance and
    groundedness are measured on the substantive prose instead.
    """
    for marker in ("Key Details", "Source:", "References:"):
        index = answer.find(marker)
        if index != -1:
            answer = answer[:index]
    return answer.strip() or "(no prose answer)"


def to_prose_test_case(scenario: Scenario, result: dict):
    """Same case, but with the structured blocks stripped from the output."""
    case = to_test_case(scenario, result)
    case.actual_output = prose_only(case.actual_output)
    return case
