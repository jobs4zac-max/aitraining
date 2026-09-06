"""Corpus, config-resolution and formatting tests. No network, no API key."""

from __future__ import annotations

import json

import pytest

from udaplay import config
from udaplay.data_loader import load_documents, load_game_records, records_to_documents
from udaplay.schemas import GameRecord
from udaplay.vector_store import format_results


# --------------------------------------------------------------------------
# The shipped corpus
# --------------------------------------------------------------------------


def test_shipped_corpus_is_valid():
    records, problems = load_game_records()
    assert problems == [], f"corpus has invalid entries: {problems}"
    assert len(records) >= 15


def test_corpus_omits_the_fallback_targets():
    """The evaluation scenarios depend on these absences.

    If someone adds God of War Ragnarok to the corpus, scenario 3 stops testing
    the fallback path and starts testing plain retrieval. Fail loudly instead.
    """
    names = {r.name.lower() for r in load_game_records()[0]}
    assert not any("ragnar" in name for name in names)
    assert max(r.year_of_release for r in load_game_records()[0]) <= 2022


def test_corpus_includes_the_internal_match_target():
    names = {r.name.lower() for r in load_game_records()[0]}
    assert "pokemon red" in names
    assert "god of war" in names  # the near-miss scenario 3 must reject


def test_documents_carry_content_and_metadata():
    documents = load_documents(verbose=False)
    assert len(documents) >= 15
    for document in documents:
        assert document.page_content.strip()
        assert document.metadata["name"]
        assert document.metadata["platform"]


def test_missing_directory_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No games directory"):
        load_game_records(tmp_path / "nope")


def test_malformed_entries_are_collected_not_raised(tmp_path):
    (tmp_path / "good.json").write_text(
        json.dumps(
            {
                "Name": "Tetris",
                "Platform": "Game Boy",
                "Genre": "Puzzle",
                "Publisher": "Nintendo",
                "YearOfRelease": 1989,
                "Description": "Falling blocks.",
            }
        )
    )
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "incomplete.json").write_text(json.dumps({"Name": "Mystery"}))

    records, problems = load_game_records(tmp_path)
    assert len(records) == 1
    assert len(problems) == 2


def test_strict_mode_raises_on_the_first_problem(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(ValueError):
        load_game_records(tmp_path, strict=True)


def test_a_single_file_may_hold_a_list_of_games(tmp_path):
    entries = [
        {
            "Name": f"Game {i}",
            "Platform": "PC",
            "Genre": "Puzzle",
            "Publisher": "Someone",
            "YearOfRelease": 2000 + i,
            "Description": "A game.",
        }
        for i in range(3)
    ]
    (tmp_path / "all.json").write_text(json.dumps(entries))
    records, problems = load_game_records(tmp_path)
    assert len(records) == 3
    assert problems == []


# --------------------------------------------------------------------------
# Base-URL resolution -- the Vocareum-specific behaviour
# --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate config resolution from the developer's real .env.

    Clearing the variables is not enough: `load_env(force=True)` re-reads the
    dotenv files with `override=True`, which would put the real key straight
    back. Pointing PROJECT_ROOT at an empty directory means there is no dotenv
    file to find, so only what a test sets is in play.
    """
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "_loaded", True)
    yield monkeypatch
    config._loaded = False


def test_voc_key_selects_the_vocareum_proxy(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "voc-2981234567890")
    config._loaded = False
    config.load_env(force=True)
    assert config.get_base_url() == config.VOCAREUM_BASE_URL


def test_standard_openai_key_leaves_the_base_url_unset(clean_env):
    """An sk- key must fall through to api.openai.com, not the proxy."""
    clean_env.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
    config._loaded = False
    config.load_env(force=True)
    assert config.get_base_url() is None


def test_voc_key_resolution_sets_both_variable_names(clean_env):
    """langchain-openai reads OPENAI_API_BASE, the openai SDK reads
    OPENAI_BASE_URL. Both must be set or one library silently uses
    api.openai.com and fails with an opaque 401."""
    import os

    clean_env.setenv("OPENAI_API_KEY", "voc-2981234567890")
    config._loaded = False
    config.load_env(force=True)

    assert os.environ["OPENAI_BASE_URL"] == config.VOCAREUM_BASE_URL
    assert os.environ["OPENAI_API_BASE"] == config.VOCAREUM_BASE_URL


def test_explicit_base_url_wins_over_the_voc_default(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "voc-2981234567890")
    clean_env.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    config._loaded = False
    config.load_env(force=True)
    assert config.get_base_url() == "https://example.test/v1"


def test_missing_key_fails_with_an_actionable_message(clean_env):
    config._loaded = True
    with pytest.raises(AssertionError, match="OPENAI_API_KEY"):
        config.assert_env()


@pytest.mark.parametrize("placeholder", ["voc-...", "sk-...", "voc-"])
def test_unedited_placeholder_is_rejected(clean_env, placeholder):
    """Copying .env.example without editing it must fail locally, not as a 401."""
    clean_env.setenv("OPENAI_API_KEY", placeholder)
    config._loaded = True
    with pytest.raises(AssertionError, match="placeholder"):
        config.assert_env()


@pytest.mark.parametrize(
    "secret, expected_visible",
    [("voc-2981234567890abcd", "voc-29"), ("", "<missing>"), ("short", "short")],
)
def test_mask_never_reveals_the_whole_key(secret, expected_visible):
    masked = config.mask(secret)
    assert expected_visible in masked
    if len(secret) > 10:
        assert secret not in masked


# --------------------------------------------------------------------------
# Result formatting
# --------------------------------------------------------------------------


def test_format_results_labels_every_required_field():
    record = GameRecord.model_validate(
        {
            "Name": "Gran Turismo",
            "Platform": "PlayStation 1",
            "Genre": "Racing",
            "Publisher": "Sony Computer Entertainment",
            "YearOfRelease": 1997,
            "Description": "A racing simulator.",
        }
    )
    document = records_to_documents([record])[0]
    rendered = format_results([(document, 0.1234)])

    for label in ("Title:", "Platform:", "Genre:", "Publisher:", "Release Date:"):
        assert label in rendered
    assert "Gran Turismo" in rendered


def test_format_results_handles_no_hits():
    assert "No matching games" in format_results([])
