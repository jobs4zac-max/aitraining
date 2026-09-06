"""Input and output validation tests. No network, no API key."""

from __future__ import annotations

import pytest

from udaplay.validation import (
    MAX_QUERY_CHARS,
    InvalidQuery,
    validate_answer,
    validate_query,
)

# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  When was Pokemon Red launched?  ", "When was Pokemon Red launched?"),
        ("What\n\nplatform\twas  it on?", "What platform was it on?"),
        ("Ｐｏｋｅｍｏｎ Red", "Pokemon Red"),  # NFKC folds full-width forms
    ],
)
def test_queries_are_normalised(raw, expected):
    assert validate_query(raw) == expected


def test_control_characters_are_stripped():
    assert validate_query("Pokemon\x00 Red\x07 release") == "Pokemon Red release"


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_empty_queries_are_rejected(raw):
    with pytest.raises(InvalidQuery):
        validate_query(raw)


def test_too_short_query_is_rejected():
    with pytest.raises(InvalidQuery, match="too short"):
        validate_query("hi")


def test_too_long_query_is_rejected():
    with pytest.raises(InvalidQuery, match="too long"):
        validate_query("a" * (MAX_QUERY_CHARS + 1))


def test_query_at_the_limit_is_accepted():
    assert len(validate_query("a" * MAX_QUERY_CHARS)) == MAX_QUERY_CHARS


def test_query_without_letters_or_digits_is_rejected():
    with pytest.raises(InvalidQuery, match="no letters or digits"):
        validate_query("!!!???...")


# --------------------------------------------------------------------------
# Output validation
# --------------------------------------------------------------------------

INTERNAL_ANSWER = """Pokemon Red launched in 1996 for the Game Boy.

Key Details:
  - Title: Pokemon Red
  - Platform: Game Boy
  - Publisher: Nintendo
  - Release Date: 1996

Source: Internal Game Database"""

WEB_ANSWER = """God of War Ragnarok released on 9 November 2022.

Key Details:
  - Platform: PlayStation 4, PlayStation 5
  - Release Date: 9 November 2022

Source: Web Search (Wikipedia/DuckDuckGo)
References:
  - https://en.wikipedia.org/wiki/God_of_War_Ragnarok"""


def test_conformant_internal_answer_has_no_warnings():
    assert validate_answer(INTERNAL_ANSWER, used_web=False) == []


def test_conformant_web_answer_has_no_warnings():
    assert validate_answer(WEB_ANSWER, used_web=True) == []


def test_empty_answer_is_flagged():
    assert validate_answer("", used_web=False) == ["Answer is empty."]


def test_missing_source_line_is_flagged():
    warnings = validate_answer("Pokemon Red came out in 1996.", used_web=False)
    assert any("Source:" in w for w in warnings)


def test_citing_the_web_when_no_web_tool_ran_is_flagged():
    """The important one: a wrong citation is worse than a missing one."""
    warnings = validate_answer(WEB_ANSWER, used_web=False)
    assert any("no web tool ran" in w for w in warnings)


def test_omitting_the_web_citation_after_a_web_call_is_flagged():
    warnings = validate_answer(INTERNAL_ANSWER, used_web=True)
    assert any("does not cite it" in w for w in warnings)


def test_missing_key_details_block_is_flagged():
    answer = "Pokemon Red launched in 1996.\n\nSource: Internal Game Database"
    warnings = validate_answer(answer, used_web=False)
    assert any("Key Details" in w for w in warnings)


def test_web_answer_without_a_url_is_flagged():
    answer = WEB_ANSWER.split("References:")[0]
    warnings = validate_answer(answer, used_web=True)
    assert any("no source URL" in w for w in warnings)
