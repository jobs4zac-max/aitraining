"""Per-turn record of which tools have run, used to enforce the workflow.

The requirements mandate an order: retrieve, then evaluate, and only then fall
back to web search. Stating that in the system prompt is not enough -- observed
in practice, gpt-4o-mini skips straight to ``game_web_search`` for an obviously
present-tense question like "what is Rockstar working on right now?", because
going straight to the web *looks* correct to the model. The answer is fine; the
mandated evaluation step never ran.

So the ordering is enforced here instead, where it cannot be talked out of.
``game_web_search`` consults this module and refuses to run early, returning
instructions rather than results; the agent then makes the missing calls and
retries. Self-correcting, and deterministic.

Implementation note -- why a plain global and not a ``ContextVar``:
LangChain invokes each tool inside ``contextvars.copy_context()``, so a
``ContextVar.set()`` performed inside a tool is discarded the moment that tool
returns. A ContextVar therefore cannot carry state *between* tool calls, which
is the entire requirement here. (This was not a guess: an ordering gate built
on a ContextVar blocked every call, because no write ever survived, and the
agent looped until it hit ``max_iterations``.)

The cost of a plain global is that state is process-wide rather than
per-session. Two turns running concurrently in one process could see each
other's tools. That only ever makes the gate *more permissive* -- it can cause
a missed enforcement, never a wrongly blocked call -- so it degrades toward the
prompt-only behaviour rather than toward a broken agent.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_tools_called: set[str] = set()


def begin_turn() -> None:
    """Clear the record. Called once per user turn, before the agent runs."""
    with _lock:
        _tools_called.clear()


def record(tool_name: str) -> None:
    """Note that a tool ran during this turn."""
    with _lock:
        _tools_called.add(tool_name)


def was_called(tool_name: str) -> bool:
    with _lock:
        return tool_name in _tools_called


def called_tools() -> frozenset[str]:
    with _lock:
        return frozenset(_tools_called)
