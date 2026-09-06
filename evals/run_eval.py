"""Run the DeepEval suite and print a scorecard.

    python -m udaplay eval
    # or
    python evals/run_eval.py --scenario near_miss_fallback

Costs money: every scenario runs the agent, then four metrics (three of them
LLM-judged) grade it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.cases import SCENARIOS, run_scenario, to_prose_test_case, to_test_case  # noqa: E402
from evals.metrics import all_metrics  # noqa: E402


def evaluate_scenario(scenario, agent=None) -> dict:
    """Run one scenario and score it against every metric."""
    print(f"\n{'=' * 78}\n{scenario.key}: {scenario.question}\n{'=' * 78}")
    print(f"Expectation: {scenario.rationale}")

    result = run_scenario(scenario, agent=agent)
    record = result["run_record"]
    cases = {
        "full": to_test_case(scenario, result),
        "prose": to_prose_test_case(scenario, result),
    }

    print(f"\nTools called : {' -> '.join(record.tools_called) or 'none'}")
    print(f"Web fallback : {record.used_web_search} (expected {scenario.expect_web})")
    print(f"Confidence   : {record.confidence_score}")
    print(f"Latency      : {record.latency_ms:.0f} ms")
    print(f"\nAnswer:\n{record.answer}\n")

    # Deterministic checks first: they cost nothing and are unambiguous.
    deterministic: dict[str, bool] = {
        "routing": record.used_web_search == scenario.expect_web,
        "format_valid": not record.warnings,
    }
    if scenario.must_mention:
        deterministic["facts_present"] = all(
            fact.lower() in record.answer.lower() for fact in scenario.must_mention
        )

    scores: dict[str, dict] = {}
    for metric, case_kind in all_metrics():
        name = getattr(metric, "__name__", type(metric).__name__)
        try:
            metric.measure(cases[case_kind])
            scores[name] = {
                "score": metric.score,
                "passed": bool(metric.success),
                "threshold": metric.threshold,
                "reason": (metric.reason or "")[:300],
            }
        except Exception as exc:  # noqa: BLE001 - one bad metric must not stop the rest
            scores[name] = {"score": None, "passed": False, "error": f"{type(exc).__name__}: {exc}"}

    print("Deterministic checks:")
    for name, passed in deterministic.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if record.warnings:
        for warning in record.warnings:
            print(f"         ! {warning}")

    print("\nDeepEval metrics:")
    for name, detail in scores.items():
        if detail.get("error"):
            print(f"  [ERROR] {name}: {detail['error']}")
            continue
        print(
            f"  [{'PASS' if detail['passed'] else 'FAIL'}] {name}: "
            f"{detail['score']:.2f} (threshold {detail['threshold']:.2f})"
        )
        if detail.get("reason"):
            print(f"         {detail['reason']}")

    return {
        "scenario": scenario.key,
        "question": scenario.question,
        "deterministic": deterministic,
        "metrics": scores,
        "record": record,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the UdaPlay DeepEval suite.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[s.key for s in SCENARIOS],
        help="run only these scenarios (repeatable); default is all",
    )
    args = parser.parse_args(argv)

    selected = [s for s in SCENARIOS if not args.scenario or s.key in args.scenario]

    from udaplay.agent import build_stateful_agent
    from udaplay.config import describe_env

    describe_env()
    agent = build_stateful_agent(verbose=False)

    outcomes = [evaluate_scenario(scenario, agent=agent) for scenario in selected]

    print(f"\n{'#' * 78}\n# SCORECARD\n{'#' * 78}\n")
    header = f"{'scenario':<24}{'routing':<10}{'format':<9}{'metrics passed':<16}"
    print(header)
    print("-" * len(header))

    all_passed = True
    for outcome in outcomes:
        metrics = outcome["metrics"]
        passed_count = sum(1 for detail in metrics.values() if detail.get("passed"))
        routing_ok = outcome["deterministic"]["routing"]
        format_ok = outcome["deterministic"]["format_valid"]
        all_passed &= routing_ok and passed_count == len(metrics)
        print(
            f"{outcome['scenario']:<24}"
            f"{'PASS' if routing_ok else 'FAIL':<10}"
            f"{'PASS' if format_ok else 'WARN':<9}"
            f"{passed_count}/{len(metrics)}"
        )

    print(
        "\nRouting is the hard requirement; the LLM-judged metrics are a regression "
        "signal (judge and agent share a model -- see evals/judge.py)."
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
