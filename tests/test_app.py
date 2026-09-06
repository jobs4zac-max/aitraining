"""Streamlit frontend tests via `AppTest`, Streamlit's headless harness.

Curling the server only proves the shell HTML was served -- the script itself
runs over a websocket, so a Python error in `app.py` would not show up. AppTest
executes the script in-process and surfaces exceptions, which is what makes
this a real check.

No key and no network: the app only builds the agent once a question is
submitted, so rendering is exercised without touching the API.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from udaplay.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")
TIMEOUT = 60


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    return instance


def test_app_runs_without_raising(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_headline_and_subheading_are_present(app):
    assert any("UdaPlay" in t.value for t in app.title)
    assert any("Gaming Research Agent" in s.value for s in app.subheader)


def test_chat_input_is_rendered(app):
    assert len(app.chat_input) == 1


def test_sample_question_buttons_are_offered(app):
    """On a fresh session the samples are the only way in besides typing."""
    labels = [b.label for b in app.button]
    assert any("Pokemon Red" in label for label in labels)
    assert any("Rockstar" in label for label in labels)


def test_sidebar_reports_the_corpus_size(app):
    metrics = {m.label: m.value for m in app.sidebar.metric}
    assert metrics.get("Games") == "15"


def test_sidebar_reports_the_resolved_endpoint(app):
    captions = " ".join(c.value for c in app.sidebar.caption)
    assert "gpt-4o-mini" in captions
    assert "Endpoint:" in captions


def test_placeholder_key_is_reported_as_an_error(app):
    """.env currently holds the template placeholder, so the UI must say so
    rather than letting the user discover it as a 401 mid-chat."""
    from udaplay.config import get_api_key

    if get_api_key().rstrip(".") not in {"voc-", "sk-"}:
        pytest.skip("a real key is configured")
    assert app.sidebar.error, "expected a visible warning about the missing key"


def test_too_short_query_is_rejected_in_the_ui():
    """Input validation must fire before the agent is ever constructed."""
    instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    instance.chat_input[0].set_value("hi").run()

    assert not instance.exception, [str(e) for e in instance.exception]
    assert any("too short" in e.value for e in instance.error)


def test_blank_query_does_not_invoke_the_agent():
    instance = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    instance.chat_input[0].set_value("   ").run()

    assert not instance.exception, [str(e) for e in instance.exception]
