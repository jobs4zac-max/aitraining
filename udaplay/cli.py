"""Command-line entry point: ``python -m udaplay <command>``.

Commands map onto the pipeline stages so each can be run and inspected on its
own:

    check    verify the environment and the live endpoint
    index    build and persist the FAISS index from data/games
    search   run a raw similarity search (no LLM, no agent)
    ask      put one question through the full agent
    chat     interactive multi-turn session
    ui       launch the Streamlit chat interface
    demo     run the three evaluation scenarios end to end
    eval     run the DeepEval scorecard
    logs     show recent runs from the run log
"""

from __future__ import annotations

import argparse
import sys


def _cmd_check(args: argparse.Namespace) -> int:
    from .config import describe_env, smoke_test
    from .data_loader import load_game_records
    from .vector_store import index_exists

    describe_env()

    records, problems = load_game_records()
    print(f"\nCorpus: {len(records)} valid records, {len(problems)} problem(s)")
    for problem in problems:
        print(f"  - {problem}")

    print(f"FAISS index present: {index_exists()}")

    if args.offline:
        print("\nSkipping live endpoint check (--offline).")
        return 0

    print("\nLive endpoint check:")
    return 0 if smoke_test() else 1


def _cmd_index(args: argparse.Namespace) -> int:
    from .data_loader import load_documents
    from .vector_store import build_index, index_exists

    if index_exists() and not args.force:
        print("Index already exists. Re-run with --force to rebuild.")
        return 0

    documents = load_documents()
    build_index(documents)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from .vector_store import format_results, search

    results = search(
        args.query, k=args.k, platform=args.platform, publisher=args.publisher
    )
    print(format_results(results))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .agent import ask, to_structured_answer

    result = ask(args.query, session_id=args.session, show_trace=not args.quiet)

    if args.quiet:
        print(result["output"])

    if args.structured:
        print("\nStructured answer:")
        print(to_structured_answer(args.query, result).model_dump_json(indent=2))

    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .agent import build_stateful_agent, ask

    agent = build_stateful_agent(verbose=False)
    print("UdaPlay interactive session. Blank line or 'exit' to quit.\n")

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        ask(question, session_id=args.session, agent=agent, show_trace=not args.quiet)

    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    return run_demo(skip_bonus=args.skip_bonus)


def _cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit app.

    Streamlit has no supported in-process entry point, so this shells out to
    its CLI rather than importing it.
    """
    import subprocess

    from .config import PROJECT_ROOT

    app = PROJECT_ROOT / "app.py"
    if not app.exists():
        print(f"Error: {app} not found.", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        command += ["--server.headless", "true"]

    print(f"Starting UdaPlay UI on http://localhost:{args.port}")
    return subprocess.call(command)


def _cmd_eval(args: argparse.Namespace) -> int:
    try:
        from evals.run_eval import main as run_eval
    except ImportError as exc:
        print(
            f"Error: could not import the eval suite ({exc}).\n"
            "DeepEval is an optional dependency: uv sync --extra eval",
            file=sys.stderr,
        )
        return 1

    argv: list[str] = []
    for scenario in args.scenario or []:
        argv += ["--scenario", scenario]
    return run_eval(argv)


def _cmd_logs(args: argparse.Namespace) -> int:
    from .logging_setup import read_runs, summarise_runs

    runs = read_runs(limit=args.limit)
    if not runs:
        print("No runs logged yet.")
        return 0

    for record in runs:
        status = "ERR " if record.error else "ok  "
        confidence = (
            f"{record.confidence_score:.2f}" if record.confidence_score is not None else " n/a"
        )
        route = "web" if record.used_web_search else "int"
        print(
            f"{status}{record.timestamp}  {route}  conf={confidence}  "
            f"{record.latency_ms:>6.0f}ms  {record.query[:60]!r}"
        )
        if record.error:
            print(f"      error: {record.error}")
        for warning in record.warnings:
            print(f"      warn:  {warning}")

    summary = summarise_runs()
    print(
        f"\n{summary['runs']} run(s), {summary['errors']} error(s), "
        f"web fallback {summary['web_fallback_rate']:.0%}, "
        f"mean latency {summary['mean_latency_ms']:.0f} ms, "
        f"{summary['warnings']} format warning(s)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m udaplay",
        description="UdaPlay - AI gaming research agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="verify environment and endpoint")
    check.add_argument(
        "--offline", action="store_true", help="skip the live API smoke test"
    )
    check.set_defaults(func=_cmd_check)

    index = sub.add_parser("index", help="build and persist the FAISS index")
    index.add_argument("--force", action="store_true", help="rebuild if it exists")
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="raw similarity search, no agent")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=4, help="results to return")
    search.add_argument("--platform", default=None, help="filter by platform")
    search.add_argument("--publisher", default=None, help="filter by publisher")
    search.set_defaults(func=_cmd_search)

    ask_p = sub.add_parser("ask", help="one question through the full agent")
    ask_p.add_argument("query")
    ask_p.add_argument("--session", default="cli", help="session id for history")
    ask_p.add_argument("--quiet", action="store_true", help="answer only, no trace")
    ask_p.add_argument(
        "--structured", action="store_true", help="also emit the Pydantic answer as JSON"
    )
    ask_p.set_defaults(func=_cmd_ask)

    chat = sub.add_parser("chat", help="interactive multi-turn session")
    chat.add_argument("--session", default="interactive")
    chat.add_argument("--quiet", action="store_true", help="answers only, no traces")
    chat.set_defaults(func=_cmd_chat)

    demo = sub.add_parser("demo", help="run the three evaluation scenarios")
    demo.add_argument(
        "--skip-bonus", action="store_true", help="omit the bonus demonstrations"
    )
    demo.set_defaults(func=_cmd_demo)

    ui = sub.add_parser("ui", help="launch the Streamlit chat interface")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument(
        "--headless", action="store_true", help="do not open a browser (for remote hosts)"
    )
    ui.set_defaults(func=_cmd_ui)

    eval_p = sub.add_parser("eval", help="run the DeepEval scorecard")
    eval_p.add_argument(
        "--scenario", action="append", help="run only this scenario (repeatable)"
    )
    eval_p.set_defaults(func=_cmd_eval)

    logs = sub.add_parser("logs", help="show recent runs from the run log")
    logs.add_argument("-n", "--limit", type=int, default=20)
    logs.set_defaults(func=_cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        # Expected, actionable failures: missing key, missing index, empty
        # corpus. A traceback would bury the message.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
