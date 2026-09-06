"""Pydantic v2 models: the validation boundary for data in and answers out."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GameRecord(BaseModel):
    """One video game as stored in ``data/games/*.json``.

    Field aliases accept the dataset's PascalCase keys while the Python side
    stays snake_case. ``populate_by_name`` means either spelling loads, and the
    release-year validation alias covers both ``YearOfRelease`` (the shape used
    by the shipped dataset) and ``ReleaseYear``.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    name: str = Field(alias="Name")
    platform: str = Field(alias="Platform")
    genre: str = Field(alias="Genre")
    publisher: str = Field(alias="Publisher")
    description: str = Field(alias="Description")
    year_of_release: int = Field(
        validation_alias="YearOfRelease",
        serialization_alias="YearOfRelease",
        ge=1950,
        le=2100,
    )

    @field_validator("name", "platform", "genre", "publisher", "description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    def to_text(self) -> str:
        """Natural-language rendering -- this is what actually gets embedded.

        Prose embeds better than a key/value dump, and repeating the title and
        platform inside the sentence means a query naming either one has
        something to match on.
        """
        return (
            f"{self.name} is a {self.genre} game released in "
            f"{self.year_of_release} for the {self.platform}, "
            f"published by {self.publisher}. {self.description}"
        )

    def to_metadata(self) -> dict[str, str | int]:
        """Flat metadata for FAISS -- drives the publisher/platform filters."""
        return {
            "name": self.name,
            "platform": self.platform,
            "genre": self.genre,
            "publisher": self.publisher,
            "year_of_release": self.year_of_release,
        }


class RetrievalEvaluation(BaseModel):
    """Verdict from ``evaluate_retrieval`` -- the fallback gate."""

    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the retrieved context answers the query, 0.0 to 1.0.",
    )
    is_sufficient: bool = Field(
        description=(
            "True only if the retrieved context alone fully answers the query. "
            "False if it is off-topic, partial, or too stale to be trusted."
        )
    )
    reasoning: str = Field(
        description="One or two sentences justifying the score and the verdict."
    )


class SourceOrigin(str, Enum):
    INTERNAL = "Internal Game Database"
    WEB = "Web Search (Wikipedia/DuckDuckGo)"
    BOTH = "Internal Game Database + Web Search"
    NONE = "No source found"


class KeyDetails(BaseModel):
    """The fields the requirements insist every answer surfaces."""

    title: str | None = None
    platform: str | None = None
    publisher: str | None = None
    release_date: str | None = None
    genre: str | None = None


class UdaPlayAnswer(BaseModel):
    """Structured mirror of the agent's natural-language reply."""

    query: str
    answer: str
    key_details: KeyDetails = Field(default_factory=KeyDetails)
    source_origin: SourceOrigin = SourceOrigin.NONE
    citations: list[str] = Field(default_factory=list)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    def render(self) -> str:
        """Human-readable form matching the required output format."""
        lines = [self.answer.strip(), "", "Key Details:"]
        details = self.key_details.model_dump()
        shown = {k: v for k, v in details.items() if v}
        if shown:
            lines += [f"  - {k.replace('_', ' ').title()}: {v}" for k, v in shown.items()]
        else:
            lines.append("  - (none extracted)")
        lines += ["", f"Source: {self.source_origin.value}"]
        if self.confidence_score is not None:
            lines.append(f"Retrieval confidence: {self.confidence_score:.2f}")
        if self.citations:
            lines.append("References:")
            lines += [f"  - {c}" for c in self.citations]
        return "\n".join(lines)
