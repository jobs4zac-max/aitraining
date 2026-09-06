"""The UdaPlay agent: tool-calling executor with per-session conversation state."""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from . import turn_state
from .config import get_llm
from .logging_setup import RunRecord, ToolCallRecord, get_logger, log_run, timed
from .schemas import KeyDetails, SourceOrigin, UdaPlayAnswer
from .tools import ALL_TOOLS
from .validation import validate_answer, validate_query

SYSTEM_PROMPT = """You are UdaPlay, a video game research analyst serving executives, analysts and gamers.

## Mandatory workflow
For every game-related question, in this order:
1. Call `retrieve_game` to search the internal database.
2. Call `evaluate_retrieval` with the question and retrieve_game's exact output.
3. If `is_sufficient` is true, answer from the retrieved context alone.
4. If `is_sufficient` is false, call `game_web_search`, then answer from what it returns.

Never skip step 2. Never call `game_web_search` before step 2 has returned false.

This holds even when a question is obviously about the present or the future and you are
confident the internal database cannot help. Steps 1 and 2 are still required: the
evaluation result is part of the deliverable, not just a routing hint. `game_web_search`
is blocked until `evaluate_retrieval` has run and will return a WORKFLOW ERROR telling you
what to do; if you see that, make the missing calls and try again.

## Answer format
Write a direct prose answer first, then:

Key Details:
  - Title: ...
  - Platform: ...
  - Publisher: ...
  - Release Date: ...

Source: Internal Game Database
  -- or --
Source: Web Search (Wikipedia/DuckDuckGo)

Rules for the Source line:
- Cite `Internal Game Database` only when you answered without calling game_web_search.
- Cite `Web Search (Wikipedia/DuckDuckGo)` when any part of the answer came from the web, and list the source URLs beneath it.
- If you used both, say `Internal Game Database + Web Search` and make clear which fact came from where.

Omit any Key Detail you genuinely do not know rather than guessing. If neither source answers the question, say so plainly.

Earlier turns in this conversation are context: resolve pronouns and follow-ups such as "what about its sequel?" against them."""

# Session id -> history. In-process only, which is the right scope for a
# notebook demo; a real deployment would back this with Redis or a database.
_session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def reset_session(session_id: str | None = None) -> None:
    """Clear one session, or all of them when no id is given."""
    if session_id is None:
        _session_store.clear()
    else:
        _session_store.pop(session_id, None)


def build_agent(verbose: bool = True) -> AgentExecutor:
    """The stateless executor: tools plus the workflow prompt."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(get_llm(), ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        return_intermediate_steps=True,
        # The workflow needs 3 tool calls plus a final answer; the cap stops a
        # confused model from looping on game_web_search indefinitely.
        max_iterations=8,
        handle_parsing_errors=True,
    )


def build_stateful_agent(verbose: bool = True) -> RunnableWithMessageHistory:
    """Wrap the executor so history is loaded and saved per ``session_id``.

    ``RunnableWithMessageHistory`` is the current equivalent of the deprecated
    ``ConversationBufferMemory``.
    """
    return RunnableWithMessageHistory(
        build_agent(verbose=verbose),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )


def ask(
    question: str,
    session_id: str = "demo",
    agent: RunnableWithMessageHistory | None = None,
    show_trace: bool = True,
) -> dict:
    """Ask a question, validating input, logging the run, checking the answer.

    Returns the executor result, so ``intermediate_steps`` stays available for
    inspection, plus a ``run_record`` key holding the observability record.

    Raises:
        InvalidQuery: if the question fails input validation. This happens
            before any API call, so a malformed query costs nothing.
    """
    logger = get_logger()
    runnable = agent if agent is not None else build_stateful_agent(verbose=False)

    # Validate first: a rejected query must not reach the LLM.
    question = validate_query(question)

    record = RunRecord(session_id=session_id, query=question)
    logger.info("run start session=%s query=%r", session_id, question[:120])

    # Reset the workflow gate: tool-order enforcement is scoped to one turn.
    turn_state.begin_turn()

    try:
        with timed() as elapsed:
            result = runnable.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
    except Exception as exc:
        record.latency_ms = elapsed.ms
        record.error = f"{type(exc).__name__}: {exc}"
        log_run(record)
        raise

    record.latency_ms = elapsed.ms
    record.answer = result.get("output", "")

    steps = result.get("intermediate_steps", [])
    record.tool_calls = [
        ToolCallRecord(
            tool=action.tool,
            tool_input=action.tool_input
            if isinstance(action.tool_input, dict)
            else str(action.tool_input),
            output_preview=str(observation)[:400],
            output_chars=len(str(observation)),
        )
        for action, observation in steps
    ]
    record.tools_called = [call.tool for call in record.tool_calls]
    record.used_web_search = used_web_search(result)
    record.confidence_score = retrieval_confidence(result)
    record.warnings = validate_answer(record.answer, record.used_web_search)

    log_run(record)
    result["run_record"] = record

    if show_trace:
        print("=" * 78)
        print(f"QUERY: {question}")
        print("=" * 78)
        print(f"\nReasoning trace ({len(result.get('intermediate_steps', []))} tool calls):")
        for step, (action, observation) in enumerate(result.get("intermediate_steps", []), 1):
            preview = str(observation).replace("\n", " ")[:220]
            print(f"\n  {step}. {action.tool}({_short(action.tool_input)})")
            print(f"     -> {preview}{'...' if len(str(observation)) > 220 else ''}")
        print("\n" + "-" * 78)
        print("ANSWER:\n")
        print(result["output"])
        if record.warnings:
            print("\nFormat warnings:")
            for warning in record.warnings:
                print(f"  ! {warning}")
        print(f"\n({record.latency_ms:.0f} ms)")
        print("-" * 78 + "\n")

    return result


def _short(tool_input, limit: int = 90) -> str:
    text = str(tool_input)
    return text if len(text) <= limit else text[:limit] + "..."


def used_web_search(result: dict) -> bool:
    """Whether the web fallback fired -- used to verify routing in notebook 02."""
    return any(
        action.tool == "game_web_search"
        for action, _ in result.get("intermediate_steps", [])
    )


def retrieval_confidence(result: dict) -> float | None:
    """Pull confidence_score out of the evaluate_retrieval step, if it ran."""
    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "evaluate_retrieval" and isinstance(observation, dict):
            return observation.get("confidence_score")
    return None


def to_structured_answer(question: str, result: dict) -> UdaPlayAnswer:
    """Second LLM pass turning the prose answer into ``UdaPlayAnswer``.

    Source origin is derived from the trace rather than asked for, so the
    citation reflects which tools actually ran instead of what the model claims.
    """
    web = used_web_search(result)
    internal = any(
        action.tool == "retrieve_game" for action, _ in result.get("intermediate_steps", [])
    )

    if web and internal:
        origin = SourceOrigin.BOTH
    elif web:
        origin = SourceOrigin.WEB
    elif internal:
        origin = SourceOrigin.INTERNAL
    else:
        origin = SourceOrigin.NONE

    class _Extraction(KeyDetails):
        pass

    extractor = get_llm().with_structured_output(_Extraction)
    details = extractor.invoke(
        [
            (
                "system",
                "Extract the game's title, platform, publisher, release date and genre "
                "from the answer below. Leave a field null if the answer does not state it. "
                "Do not infer anything not present in the text.",
            ),
            ("human", result["output"]),
        ]
    )

    citations = sorted(
        {
            word.strip(").,'\"")
            for observation in (str(o) for _, o in result.get("intermediate_steps", []))
            for word in observation.split()
            if word.startswith("http")
        }
    )

    return UdaPlayAnswer(
        query=question,
        answer=result["output"],
        key_details=KeyDetails(**details.model_dump()),
        source_origin=origin,
        citations=citations,
        confidence_score=retrieval_confidence(result),
    )
