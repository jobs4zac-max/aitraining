"""Logging and per-run observability.

Two sinks, because they answer different questions:

  * ``logs/udaplay.log`` -- human-readable, rotating. "What happened, in order?"
  * ``logs/runs.jsonl``  -- one JSON object per agent run. "How did the agent
    route across the last 200 queries, and how often did the fallback fire?"

The JSONL sink is what makes routing behaviour measurable after the fact
without re-running the agent, and it is what the DeepEval suite reads.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import PROJECT_ROOT

# Overridable for the same reason as INDEX_DIR -- see udaplay/config.py. Set as
# a real environment variable, not via .env.
LOG_DIR = Path(os.getenv("UDAPLAY_LOG_DIR") or PROJECT_ROOT / "logs")
TEXT_LOG = LOG_DIR / "udaplay.log"
RUN_LOG = LOG_DIR / "runs.jsonl"

_configured = False


class ToolCallRecord(BaseModel):
    """One tool invocation within a run."""

    tool: str
    tool_input: dict[str, Any] | str
    output_preview: str
    output_chars: int
    duration_ms: float | None = None


class RunRecord(BaseModel):
    """Everything worth knowing about a single agent run."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    session_id: str
    query: str
    answer: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    used_web_search: bool = False
    source_origin: str | None = None
    latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def configure_logging(
    level: int | str = logging.INFO,
    log_dir: Path | str | None = None,
    console: bool = True,
) -> logging.Logger:
    """Set up the ``udaplay`` logger. Idempotent -- safe to call repeatedly.

    Streamlit reruns the whole script on every interaction, so a
    non-idempotent setup here would attach a duplicate handler per keystroke
    and multiply every log line.
    """
    global _configured

    logger = logging.getLogger("udaplay")

    if _configured:
        return logger

    directory = Path(log_dir) if log_dir else LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    logger.setLevel(level)
    logger.propagate = False  # don't duplicate into the root logger

    file_handler = logging.handlers.RotatingFileHandler(
        directory / TEXT_LOG.name, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        # Terser on the console -- the timestamp is noise when watching live.
        console_handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        console_handler.setLevel(max(logging.INFO, logging.getLevelName(level) if isinstance(level, str) else level))
        logger.addHandler(console_handler)

    _configured = True
    logger.debug("Logging configured -> %s", directory)
    return logger


def get_logger(name: str = "udaplay") -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_run(record: RunRecord, log_dir: Path | str | None = None) -> None:
    """Append a run to the JSONL log.

    Never raises: losing an observability record must not fail the request it
    was observing.
    """
    directory = Path(log_dir) if log_dir else LOG_DIR
    logger = get_logger()

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / RUN_LOG.name).open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
    except OSError as exc:
        logger.warning("Could not write run log: %s", exc)

    if record.error:
        logger.error(
            "run failed session=%s query=%r error=%s",
            record.session_id,
            record.query[:80],
            record.error,
        )
    else:
        logger.info(
            "run ok session=%s tools=%s confidence=%s web=%s latency=%.0fms",
            record.session_id,
            "->".join(record.tools_called) or "none",
            f"{record.confidence_score:.2f}" if record.confidence_score is not None else "n/a",
            record.used_web_search,
            record.latency_ms,
        )
    for warning in record.warnings:
        logger.warning("run warning session=%s: %s", record.session_id, warning)


def read_runs(limit: int | None = None, log_dir: Path | str | None = None) -> list[RunRecord]:
    """Read back the run log, newest last. Malformed lines are skipped."""
    directory = Path(log_dir) if log_dir else LOG_DIR
    path = directory / RUN_LOG.name

    if not path.exists():
        return []

    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(RunRecord.model_validate_json(line))
        except Exception:  # noqa: BLE001 - a corrupt line should not hide the rest
            continue

    return records[-limit:] if limit else records


def summarise_runs(records: list[RunRecord] | None = None) -> dict[str, Any]:
    """Aggregate stats over the run log -- powers the Streamlit sidebar."""
    runs = records if records is not None else read_runs()

    if not runs:
        return {"runs": 0}

    successful = [r for r in runs if r.ok]
    confidences = [r.confidence_score for r in runs if r.confidence_score is not None]
    fallbacks = sum(1 for r in runs if r.used_web_search)

    return {
        "runs": len(runs),
        "errors": len(runs) - len(successful),
        "web_fallback_rate": fallbacks / len(runs),
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "mean_latency_ms": sum(r.latency_ms for r in runs) / len(runs),
        "warnings": sum(len(r.warnings) for r in runs),
    }


class timed:
    """Context manager measuring elapsed milliseconds.

    ``with timed() as t: ...`` then read ``t.ms``.
    """

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
