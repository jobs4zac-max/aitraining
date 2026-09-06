"""Logging and run-record tests. No network, no API key."""

from __future__ import annotations

import json
import logging

import pytest

from udaplay.logging_setup import (
    RunRecord,
    ToolCallRecord,
    configure_logging,
    log_run,
    read_runs,
    summarise_runs,
    timed,
)


def _record(**overrides) -> RunRecord:
    defaults = dict(
        session_id="test",
        query="When was Pokemon Red launched?",
        answer="1996, Game Boy. Source: Internal Game Database",
        tools_called=["retrieve_game", "evaluate_retrieval"],
        confidence_score=0.9,
        used_web_search=False,
        latency_ms=1234.5,
    )
    return RunRecord(**{**defaults, **overrides})


def test_run_record_defaults_a_timestamp():
    record = RunRecord(session_id="s", query="q")
    assert record.timestamp
    assert record.ok is True


def test_run_record_with_an_error_is_not_ok():
    assert RunRecord(session_id="s", query="q", error="boom").ok is False


def test_log_run_appends_one_json_line_per_run(tmp_path):
    log_run(_record(), log_dir=tmp_path)
    log_run(_record(query="second question"), log_dir=tmp_path)

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["query"] == "second question"


def test_read_runs_round_trips_a_record(tmp_path):
    original = _record(
        tool_calls=[
            ToolCallRecord(
                tool="retrieve_game",
                tool_input={"query": "pokemon"},
                output_preview="[Result 1] ...",
                output_chars=400,
            )
        ]
    )
    log_run(original, log_dir=tmp_path)

    restored = read_runs(log_dir=tmp_path)
    assert len(restored) == 1
    assert restored[0].query == original.query
    assert restored[0].tool_calls[0].tool == "retrieve_game"


def test_read_runs_skips_corrupt_lines(tmp_path):
    """A truncated write must not make the whole log unreadable."""
    log_run(_record(), log_dir=tmp_path)
    with (tmp_path / "runs.jsonl").open("a") as handle:
        handle.write('{"partial": \n')
    log_run(_record(query="after the corruption"), log_dir=tmp_path)

    restored = read_runs(log_dir=tmp_path)
    assert [r.query for r in restored] == [
        "When was Pokemon Red launched?",
        "after the corruption",
    ]


def test_read_runs_honours_the_limit(tmp_path):
    for i in range(5):
        log_run(_record(query=f"question {i}"), log_dir=tmp_path)
    assert [r.query for r in read_runs(limit=2, log_dir=tmp_path)] == [
        "question 3",
        "question 4",
    ]


def test_read_runs_on_a_missing_log_is_empty(tmp_path):
    assert read_runs(log_dir=tmp_path / "nope") == []


def test_log_run_does_not_raise_when_the_directory_is_unwritable(tmp_path):
    """Losing an observability record must not fail the request it observed."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    log_run(_record(), log_dir=blocker)  # must not raise


def test_summarise_runs_on_an_empty_log():
    assert summarise_runs([]) == {"runs": 0}


def test_summarise_runs_aggregates():
    runs = [
        _record(used_web_search=False, confidence_score=0.9, latency_ms=1000),
        _record(used_web_search=True, confidence_score=0.3, latency_ms=3000),
        _record(used_web_search=True, confidence_score=0.1, latency_ms=2000, error="boom"),
    ]
    summary = summarise_runs(runs)

    assert summary["runs"] == 3
    assert summary["errors"] == 1
    assert summary["web_fallback_rate"] == pytest.approx(2 / 3)
    assert summary["mean_confidence"] == pytest.approx(0.4333, abs=1e-3)
    assert summary["mean_latency_ms"] == pytest.approx(2000)


def test_configure_logging_is_idempotent(tmp_path):
    """Streamlit reruns the script on every interaction; a non-idempotent
    setup would attach a duplicate handler each time and multiply log lines."""
    import udaplay.logging_setup as module

    module._configured = False
    logging.getLogger("udaplay").handlers.clear()

    first = configure_logging(log_dir=tmp_path)
    count = len(first.handlers)
    for _ in range(3):
        configure_logging(log_dir=tmp_path)

    assert len(logging.getLogger("udaplay").handlers) == count


def test_timed_measures_elapsed_milliseconds():
    with timed() as elapsed:
        sum(range(200_000))
    assert elapsed.ms > 0
