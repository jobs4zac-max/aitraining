"""Load and validate the raw game JSON, then convert it to LangChain Documents."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from pydantic import ValidationError

from .config import GAMES_DIR
from .schemas import GameRecord


def load_game_records(
    games_dir: Path | str | None = None, strict: bool = False
) -> tuple[list[GameRecord], list[str]]:
    """Read every ``*.json`` under ``games_dir`` and validate it.

    A file may hold either a single game object or a list of them, so both the
    one-file-per-game layout used here and a single combined dump will load.

    Returns ``(records, problems)``. Problems are collected rather than raised
    so one malformed file cannot stop an index rebuild -- pass ``strict=True``
    to raise on the first failure instead.
    """
    directory = Path(games_dir) if games_dir else GAMES_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"No games directory at {directory}. Expected JSON files with keys: "
            "Name, Platform, Genre, Publisher, YearOfRelease, Description."
        )

    records: list[GameRecord] = []
    problems: list[str] = []

    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            message = f"{path.name}: invalid JSON ({exc})"
            if strict:
                raise ValueError(message) from exc
            problems.append(message)
            continue

        for entry in raw if isinstance(raw, list) else [raw]:
            try:
                records.append(GameRecord.model_validate(entry))
            except ValidationError as exc:
                first = exc.errors()[0]
                message = (
                    f"{path.name}: {'.'.join(str(p) for p in first['loc'])} "
                    f"-- {first['msg']}"
                )
                if strict:
                    raise ValueError(message) from exc
                problems.append(message)

    return records, problems


def records_to_documents(records: list[GameRecord]) -> list[Document]:
    """Wrap validated records as Documents ready for embedding."""
    return [
        Document(page_content=record.to_text(), metadata=record.to_metadata())
        for record in records
    ]


def load_documents(
    games_dir: Path | str | None = None, verbose: bool = True
) -> list[Document]:
    """The convenience path used by the notebooks: JSON on disk -> Documents."""
    records, problems = load_game_records(games_dir)

    if verbose:
        print(f"Loaded and validated {len(records)} game records.")
        if problems:
            print(f"Skipped {len(problems)} malformed entr{'y' if len(problems) == 1 else 'ies'}:")
            for problem in problems:
                print(f"  - {problem}")

    if not records:
        raise ValueError(
            f"No valid game records found in {games_dir or GAMES_DIR}. "
            "Cannot build an index from an empty corpus."
        )

    return records_to_documents(records)
