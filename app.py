"""UdaPlay Streamlit frontend.

Run from the project root:

    streamlit run app.py
    # or
    python -m udaplay ui

Streamlit re-executes this whole file on every interaction, so anything
expensive lives behind ``@st.cache_resource`` and the conversation lives in
``st.session_state``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `udaplay` importable when Streamlit is launched from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from udaplay.config import describe_env, get_base_url, get_chat_model, mask  # noqa: E402
from udaplay.config import get_api_key  # noqa: E402
from udaplay.logging_setup import configure_logging, summarise_runs  # noqa: E402
from udaplay.validation import MAX_QUERY_CHARS, InvalidQuery, validate_query  # noqa: E402

st.set_page_config(
    page_title="UdaPlay - Gaming Research Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = [
    "When was Pokemon Red launched, and on what platform?",
    "What is Rockstar Games working on right now?",
    "When was God of War Ragnarok released?",
    "Which racing games do you have for PlayStation?",
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _bootstrap():
    """Configure logging and build the agent once per server process."""
    configure_logging(console=False)  # the console sink would go to the server log

    from udaplay.agent import build_stateful_agent
    from udaplay.vector_store import get_store

    get_store()  # builds the index on first run if it is missing
    return build_stateful_agent(verbose=False)


@st.cache_data(show_spinner=False)
def _corpus_stats() -> dict:
    from udaplay.data_loader import load_game_records

    records, problems = load_game_records()
    return {
        "games": len(records),
        "problems": len(problems),
        "publishers": sorted({r.publisher for r in records}),
        "platforms": sorted({r.platform for r in records}),
        "year_range": (
            min(r.year_of_release for r in records),
            max(r.year_of_release for r in records),
        )
        if records
        else (0, 0),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Configuration")

        key = get_api_key()
        if not key or key.rstrip(".") in {"voc-", "sk-"}:
            st.error("No API key set. Add `OPENAI_API_KEY` to `.env`.")
        else:
            st.success(f"Key: `{mask(key)}`")

        st.caption(f"**Model:** {get_chat_model()}")
        st.caption(f"**Endpoint:** {get_base_url() or 'api.openai.com (default)'}")

        st.divider()
        st.header("Knowledge base")
        try:
            stats = _corpus_stats()
            left, right = st.columns(2)
            left.metric("Games", stats["games"])
            right.metric("Publishers", len(stats["publishers"]))
            st.caption(
                f"Release years {stats['year_range'][0]}–{stats['year_range'][1]}. "
                "Anything more recent needs the web fallback."
            )
            if stats["problems"]:
                st.warning(f"{stats['problems']} record(s) failed validation.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the corpus: {exc}")

        st.divider()
        st.header("Session stats")
        summary = summarise_runs()
        if summary.get("runs"):
            left, right = st.columns(2)
            left.metric("Queries logged", summary["runs"])
            right.metric("Web fallback", f"{summary['web_fallback_rate']:.0%}")
            if summary.get("mean_confidence") is not None:
                st.caption(f"Mean retrieval confidence: {summary['mean_confidence']:.2f}")
            st.caption(f"Mean latency: {summary['mean_latency_ms']:.0f} ms")
            if summary.get("errors"):
                st.warning(f"{summary['errors']} failed run(s) - see `logs/udaplay.log`.")
        else:
            st.caption("No queries logged yet.")

        st.divider()
        if st.button("Clear conversation", width="stretch"):
            from udaplay.agent import reset_session

            reset_session(st.session_state.get("session_id", "streamlit"))
            st.session_state.messages = []
            st.rerun()

        with st.expander("Environment detail"):
            st.code("\n".join(f"{k}: {v}" for k, v in describe_env().items()))


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def render_trace(record) -> None:
    """Show which tools ran, in order, with the routing decision made explicit."""
    route = "Web fallback" if record.used_web_search else "Internal database only"
    confidence = (
        f"{record.confidence_score:.2f}" if record.confidence_score is not None else "n/a"
    )

    with st.expander(
        f"Reasoning trace — {len(record.tool_calls)} tool calls · "
        f"{route} · confidence {confidence} · {record.latency_ms:.0f} ms"
    ):
        if not record.tool_calls:
            st.caption("The model answered without calling any tool.")

        for step, call in enumerate(record.tool_calls, start=1):
            st.markdown(f"**{step}. `{call.tool}`**")
            st.caption(f"Input: `{call.tool_input}`")
            preview = call.output_preview
            if call.output_chars > len(preview):
                preview += f"\n… ({call.output_chars:,} chars total)"
            st.code(preview, language="text")

        if record.warnings:
            st.warning("Format warnings:\n" + "\n".join(f"- {w}" for w in record.warnings))


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🎮" if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            if message.get("record") is not None:
                render_trace(message["record"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🎮 UdaPlay")
    st.subheader("AI Gaming Research Agent")
    st.markdown(
        "Ask about video games. UdaPlay searches its **internal database** first, "
        "judges whether the result actually answers your question, and falls back to "
        "**Wikipedia and DuckDuckGo** when it does not — then tells you which source "
        "it used."
    )

    render_sidebar()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", "streamlit")
    st.session_state.setdefault("pending", None)

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        columns = st.columns(len(SAMPLE_QUESTIONS))
        for column, question in zip(columns, SAMPLE_QUESTIONS):
            if column.button(question, width="stretch"):
                st.session_state.pending = question
                st.rerun()

    st.divider()
    render_history()

    typed = st.chat_input(f"Ask about a game… (max {MAX_QUERY_CHARS} characters)")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if not question:
        return

    try:
        question = validate_query(question)
    except InvalidQuery as exc:
        st.error(str(exc))
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎮"):
        placeholder = st.empty()
        placeholder.markdown("_Searching the internal database…_")

        try:
            agent = _bootstrap()
            from udaplay.agent import ask

            result = ask(
                question,
                session_id=st.session_state.session_id,
                agent=agent,
                show_trace=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface it in the UI, log the rest
            placeholder.empty()
            st.error(f"{type(exc).__name__}: {exc}")
            st.caption("Full traceback is in `logs/udaplay.log`.")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Request failed: {exc}", "record": None}
            )
            return

        answer = result["output"]
        record = result["run_record"]

        placeholder.markdown(answer)
        render_trace(record)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "record": record}
    )


main()
