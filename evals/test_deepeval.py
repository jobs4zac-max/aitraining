"""The evaluation scenarios as pytest assertions.

    pytest evals -m live

Same scenarios as ``run_eval.py``, but failing rather than reporting, so this
is the form to put in CI. Marked ``live``: needs a key, an index and network.

The agent runs once per scenario and the result is shared across the assertions
for that scenario -- re-running it per metric would quadruple the cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.cases import (  # noqa: E402
    SCENARIOS,
    run_scenario,
    to_prose_test_case,
    to_test_case,
)

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def agent():
    from udaplay.agent import build_stateful_agent

    return build_stateful_agent(verbose=False)


@pytest.fixture(scope="module")
def runs(agent) -> dict:
    """Run every scenario once; share the results across all assertions."""
    return {scenario.key: run_scenario(scenario, agent=agent) for scenario in SCENARIOS}


def _scenario(key):
    return next(s for s in SCENARIOS if s.key == key)


@pytest.mark.parametrize("key", [s.key for s in SCENARIOS])
class TestRouting:
    """Deterministic checks -- no LLM judge, so these are the strict ones."""

    def test_web_fallback_matches_expectation(self, key, runs):
        scenario = _scenario(key)
        record = runs[key]["run_record"]
        assert record.used_web_search == scenario.expect_web, (
            f"{key}: expected web fallback={scenario.expect_web}, "
            f"got {record.used_web_search}. Tools: {record.tools_called}. "
            f"Rationale: {scenario.rationale}"
        )

    def test_evaluation_step_was_not_skipped(self, key, runs):
        """The two-tier workflow is void if evaluate_retrieval never runs."""
        record = runs[key]["run_record"]
        assert "evaluate_retrieval" in record.tools_called, (
            f"{key}: evaluate_retrieval was skipped. Tools: {record.tools_called}"
        )

    def test_answer_format_is_conformant(self, key, runs):
        record = runs[key]["run_record"]
        assert not record.warnings, f"{key}: {record.warnings}"

    def test_required_facts_are_present(self, key, runs):
        scenario = _scenario(key)
        if not scenario.must_mention:
            pytest.skip("no literal facts asserted for this scenario")
        answer = runs[key]["run_record"].answer.lower()
        missing = [f for f in scenario.must_mention if f.lower() not in answer]
        assert not missing, f"{key}: answer omits {missing}"


@pytest.mark.parametrize("key", [s.key for s in SCENARIOS])
class TestDeepEvalMetrics:
    """LLM-judged metrics. Softer thresholds -- see evals/metrics.py."""

    def _assert_metric(self, metric, key, runs, case_kind: str):
        builder = to_test_case if case_kind == "full" else to_prose_test_case
        metric.measure(builder(_scenario(key), runs[key]))
        assert metric.success, (
            f"{key}: {type(metric).__name__} scored {metric.score:.2f} "
            f"(threshold {metric.threshold:.2f}) -- {metric.reason}"
        )

    def test_tool_correctness(self, key, runs):
        from evals.metrics import tool_correctness_metric

        self._assert_metric(tool_correctness_metric(), key, runs, "full")

    def test_faithfulness(self, key, runs):
        from evals.metrics import faithfulness_metric

        self._assert_metric(faithfulness_metric(), key, runs, "prose")

    def test_answer_relevancy(self, key, runs):
        from evals.metrics import answer_relevancy_metric

        self._assert_metric(answer_relevancy_metric(), key, runs, "prose")

    def test_citation_correctness(self, key, runs):
        from evals.metrics import citation_metric

        self._assert_metric(citation_metric(), key, runs, "full")
