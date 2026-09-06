"""The evaluation run: the three required scenarios plus the bonus features.

This is the artefact that demonstrates the rubric. Each scenario asserts on the
*trace* rather than the prose, so a passing run proves the routing actually
happened instead of proving the model said the right words.
"""

from __future__ import annotations

from .agent import (
    ask,
    build_stateful_agent,
    reset_session,
    retrieval_confidence,
    to_structured_answer,
    used_web_search,
)

# (question, expect_web_fallback, why)
SCENARIOS: list[tuple[str, bool, str]] = [
    (
        "When was Pokemon Red launched, and on what platform?",
        False,
        "Internal DB match: the record exists, so evaluation should pass and no "
        "web call should happen.",
    ),
    (
        "What is Rockstar Games working on right now?",
        True,
        "Fallback trigger: the DB holds only historical Rockstar titles, which "
        "cannot answer a present-tense question.",
    ),
    (
        "When was God of War Ragnarok released?",
        True,
        "Release verification: the DB has God of War (2018) but not Ragnarok, so "
        "the near-miss must be rejected.",
    ),
]


def _banner(text: str) -> None:
    print()
    print("#" * 78)
    print(f"# {text}")
    print("#" * 78)


def run_scenarios(agent=None) -> list[dict]:
    """Run the three required scenarios and report routing correctness."""
    runner = agent if agent is not None else build_stateful_agent(verbose=False)
    outcomes = []

    for number, (question, expect_web, rationale) in enumerate(SCENARIOS, start=1):
        _banner(f"SCENARIO {number}: {rationale}")

        # A fresh session per scenario keeps them independent -- otherwise
        # scenario 3 could answer from scenario 2's history instead of routing.
        session = f"scenario-{number}"
        reset_session(session)

        result = ask(question, session_id=session, agent=runner, show_trace=True)

        went_web = used_web_search(result)
        confidence = retrieval_confidence(result)
        routed_correctly = went_web == expect_web

        print(f"Routing check   : expected web fallback={expect_web}, actual={went_web}")
        print(f"Confidence      : {confidence if confidence is not None else 'not evaluated'}")
        print(f"Result          : {'PASS' if routed_correctly else 'MISROUTED'}")

        outcomes.append(
            {
                "scenario": number,
                "question": question,
                "expected_web": expect_web,
                "used_web": went_web,
                "confidence": confidence,
                "passed": routed_correctly,
                "result": result,
            }
        )

    return outcomes


def run_multiturn_demo(agent=None) -> None:
    """Show that conversational state survives across turns.

    The follow-up contains no game name -- it can only be answered by reading
    the previous turn, so a coherent answer is itself the evidence.
    """
    _banner("STATE MANAGEMENT: multi-turn follow-up on one session")

    runner = agent if agent is not None else build_stateful_agent(verbose=False)
    session = "multiturn"
    reset_session(session)

    ask(
        "Tell me about the game God of War from 2018.",
        session_id=session,
        agent=runner,
    )
    ask(
        "Who published it, and what platform was it on?",
        session_id=session,
        agent=runner,
    )
    print("The second question names no game. Answering it required the history.")


def run_bonus_demos() -> None:
    """The three suggested enhancements."""
    from .vector_store import add_web_facts, format_results, search

    _banner("BONUS 1: metadata-filtered retrieval")
    print("Query 'open world adventure' restricted to publisher='Rockstar':\n")
    print(format_results(search("open world adventure", k=3, publisher="Rockstar")))

    _banner("BONUS 2: structured Pydantic output alongside the prose answer")
    question = "When was Gran Turismo released and who published it?"
    result = ask(question, session_id="bonus-structured", show_trace=False)
    structured = to_structured_answer(question, result)
    print(structured.model_dump_json(indent=2))
    print("\nRendered:\n")
    print(structured.render())

    _banner("BONUS 3: long-term memory -- writing a web fact back into FAISS")
    question = "When was God of War Ragnarok released?"
    result = ask(question, session_id="bonus-memory", show_trace=False)

    if used_web_search(result):
        add_web_facts(question, result["output"])
        print("\nThe same query now retrieves the learned note from the index:\n")
        print(format_results(search("God of War Ragnarok release date", k=2)))
    else:
        print("Web fallback did not fire, so there is no new fact to persist.")


def run_demo(skip_bonus: bool = False) -> int:
    """Full evaluation run. Returns a shell exit code: 0 when all routing passed."""
    from .config import describe_env
    from .vector_store import index_exists

    describe_env()

    if not index_exists():
        print("\nNo FAISS index found - building it now.")
        from .vector_store import build_index

        build_index()

    agent = build_stateful_agent(verbose=False)
    outcomes = run_scenarios(agent)
    run_multiturn_demo(agent)

    if not skip_bonus:
        run_bonus_demos()

    _banner("SUMMARY")
    for outcome in outcomes:
        status = "PASS" if outcome["passed"] else "MISROUTED"
        route = "web fallback" if outcome["used_web"] else "internal only"
        confidence = outcome["confidence"]
        confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
        print(
            f"  [{status}] Scenario {outcome['scenario']}: {route} "
            f"(confidence {confidence_text}) - {outcome['question']}"
        )

    passed = sum(1 for o in outcomes if o["passed"])
    print(f"\n{passed}/{len(outcomes)} scenarios routed as expected.")

    return 0 if passed == len(outcomes) else 1
