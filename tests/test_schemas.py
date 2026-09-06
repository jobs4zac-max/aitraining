"""Validation-boundary tests. No network, no API key."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from udaplay.schemas import (
    GameRecord,
    KeyDetails,
    RetrievalEvaluation,
    SourceOrigin,
    UdaPlayAnswer,
)

VALID = {
    "Name": "Pokemon Red",
    "Platform": "Game Boy",
    "Genre": "Role-playing",
    "Publisher": "Nintendo",
    "YearOfRelease": 1996,
    "Description": "Catch and train creatures across Kanto.",
}


def test_loads_dataset_pascal_case_keys():
    record = GameRecord.model_validate(VALID)
    assert record.name == "Pokemon Red"
    assert record.year_of_release == 1996


def test_loads_snake_case_keys_too():
    """populate_by_name keeps round-tripping from Python dicts working."""
    record = GameRecord(
        name="Halo",
        platform="Xbox",
        genre="FPS",
        publisher="Microsoft",
        year_of_release=2001,
        description="Master Chief versus the Covenant.",
    )
    assert record.name == "Halo"


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "Publisher"}
    with pytest.raises(ValidationError):
        GameRecord.model_validate(payload)


def test_blank_field_is_rejected():
    with pytest.raises(ValidationError):
        GameRecord.model_validate({**VALID, "Name": "   "})


@pytest.mark.parametrize("year", [1800, 2200, "not-a-year"])
def test_implausible_year_is_rejected(year):
    with pytest.raises(ValidationError):
        GameRecord.model_validate({**VALID, "YearOfRelease": year})


def test_embedded_text_mentions_the_searchable_facts():
    text = GameRecord.model_validate(VALID).to_text()
    for expected in ("Pokemon Red", "Game Boy", "Nintendo", "1996"):
        assert expected in text


def test_metadata_carries_the_filterable_fields():
    metadata = GameRecord.model_validate(VALID).to_metadata()
    assert set(metadata) == {"name", "platform", "genre", "publisher", "year_of_release"}
    assert isinstance(metadata["year_of_release"], int)


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_confidence_score_is_bounded(score):
    with pytest.raises(ValidationError):
        RetrievalEvaluation(confidence_score=score, is_sufficient=True, reasoning="x")


def test_rendered_answer_states_its_source():
    answer = UdaPlayAnswer(
        query="When was Pokemon Red launched?",
        answer="Pokemon Red launched in 1996 on the Game Boy.",
        key_details=KeyDetails(title="Pokemon Red", platform="Game Boy"),
        source_origin=SourceOrigin.INTERNAL,
        confidence_score=0.93,
    )
    rendered = answer.render()
    assert "Source: Internal Game Database" in rendered
    assert "Platform: Game Boy" in rendered
    # Unset details are omitted rather than rendered as "None".
    assert "None" not in rendered
