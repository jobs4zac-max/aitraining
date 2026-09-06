"""DeepEval metric definitions.

Four metrics, chosen because each catches a failure the others miss:

  * **ToolCorrectness** -- deterministic, no LLM. Did the mandated tool
    sequence actually run? Catches a skipped evaluation step.
  * **Faithfulness** -- is the answer grounded in what was retrieved, or did
    the model fill gaps from its own weights? This is the metric that catches
    the near-miss failure mode: answering "Ragnarok" from the 2018 record.
  * **AnswerRelevancy** -- does the answer address the question asked?
  * **CitationCorrectness** (custom GEval) -- the project's own output
    contract: is the ``Source:`` line present and consistent with the tools
    that ran? No off-the-shelf metric covers this.
"""

from __future__ import annotations

from .judge import get_judge

# Thresholds are set for a same-model judge (see judge.py) and are deliberately
# not 0.9+: gpt-4o-mini grading gpt-4o-mini is noisy, and a flapping suite gets
# ignored. ToolCorrectness is the strict one because it involves no LLM.
TOOL_CORRECTNESS_THRESHOLD = 1.0
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7
CITATION_THRESHOLD = 0.7


def tool_correctness_metric():
    """Did the required tools run? Exact-match on the set, ordering ignored.

    Ordering is not enforced because the agent may legitimately re-retrieve
    after a web search. The *presence* of ``evaluate_retrieval`` is the
    requirement.
    """
    from deepeval.metrics import ToolCorrectnessMetric

    return ToolCorrectnessMetric(
        threshold=TOOL_CORRECTNESS_THRESHOLD,
        should_consider_ordering=False,
        include_reason=True,
    )


def faithfulness_metric():
    from deepeval.metrics import FaithfulnessMetric

    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=get_judge(),
        include_reason=True,
        async_mode=False,
    )


def answer_relevancy_metric():
    from deepeval.metrics import AnswerRelevancyMetric

    return AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=get_judge(),
        include_reason=True,
        async_mode=False,
    )


def citation_metric():
    """Custom metric for this project's citation and presentation contract.

    ``expected_output`` carries the *required* Source line, computed from the
    execution trace in ``cases.to_test_case``. The judge compares against that
    rather than inferring which sources were used from the context -- the
    earlier version asked it to infer, and it reported web snippets in a
    context containing only internal records, scoring a correctly-cited answer
    0.32. Attribution is fully determined by the trace, so it is supplied as
    ground truth; the judge is left only with presentation, which genuinely
    needs judgement.

    Note that attribution is *also* checked deterministically by
    ``validate_answer``, which is the authoritative check. This metric adds a
    view on how clearly the answer is presented.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name="CitationCorrectness",
        model=get_judge(),
        threshold=CITATION_THRESHOLD,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "'Expected Output' states the exact 'Source:' line this answer is "
            "required to carry. Treat it as ground truth about which sources were "
            "actually used -- do not try to infer the sources yourself.",
            "Check the actual output contains a 'Source:' line, and that it names the "
            "same source as 'Expected Output'. This is the most important criterion.",
            "Check the output includes a 'Key Details' block. Where the answer knows "
            "them, it should list platform, publisher and release date.",
            "Check the answer states its conclusion in prose before the structured "
            "blocks, rather than only listing fields.",
            "Do not judge whether the facts are correct, and do not penalise the "
            "answer for omitting a detail it does not know.",
        ],
        async_mode=False,
    )


#: Which test case each metric is scored against.
#:   "full"  -- the answer exactly as the user sees it
#:   "prose" -- the structured blocks stripped (see cases.prose_only)
METRIC_PLAN: list[tuple[str, str]] = [
    ("tool_correctness", "full"),
    ("faithfulness", "prose"),
    ("answer_relevancy", "prose"),
    ("citation", "full"),
]

_FACTORIES = {
    "tool_correctness": tool_correctness_metric,
    "faithfulness": faithfulness_metric,
    "answer_relevancy": answer_relevancy_metric,
    "citation": citation_metric,
}


def all_metrics() -> list:
    """Every metric, paired with the case kind it should be scored against."""
    return [(_FACTORIES[name](), case_kind) for name, case_kind in METRIC_PLAN]
