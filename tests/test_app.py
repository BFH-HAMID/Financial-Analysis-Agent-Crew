"""Smoke and integration tests for the Streamlit front end.

``streamlit.testing.v1.AppTest`` runs the real app script in a headless runtime,
so these tests catch import errors, widget wiring mistakes and rendering crashes
that unit tests cannot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app():
    """A fresh AppTest instance with a generous timeout (offline demo run)."""
    instance = AppTest.from_file(APP_PATH, default_timeout=120)
    instance.run()
    return instance


def test_app_starts_without_exception(app):
    assert not app.exception
    assert (
        any("Financial Analysis Agent Crew" in str(title.value) for title in app.title)
        or app.markdown
    )


def test_sidebar_configuration_is_rendered(app):
    labels = " ".join(str(toggle.label) for toggle in app.toggle)
    assert "Use live Gemini agents" in labels or "live" in labels.lower()
    assert len(app.sidebar) > 0


def test_form_inputs_exist(app):
    assert any("ticker" in str(box.label).lower() for box in app.text_input)
    assert len(app.date_input) >= 2
    assert any("Analyze" in str(button.label) for button in app.button)


def test_empty_state_is_rendered(app):
    body = " ".join(str(block.value) for block in app.markdown)
    assert "How it works" in body


def test_invalid_ticker_shows_error(app):
    # Widget value and submit must be set before a single rerun: form widgets are
    # re-sent together on submit, so an intermediate .run() would reset them.
    app.text_input[0].set_value("BAD TICKER!!")
    app.button[0].click()
    app.run()
    assert not app.exception
    assert app.error
    assert "valid ticker" in str(app.error[0].value).lower()


def test_full_analysis_flow_renders_results():
    """Click Analyze and verify metrics, tabs and the download button appear."""
    instance = AppTest.from_file(APP_PATH, default_timeout=240).run()
    instance.text_input[0].set_value("AAPL")
    instance.button[0].click()
    instance.run(timeout=240)

    assert not instance.exception, instance.exception
    assert not instance.error
    result = instance.session_state["result"]
    assert result is not None
    assert result.ticker == "AAPL"
    assert instance.session_state["pdf_bytes"]
    assert "Results" in " ".join(str(block.value) for block in instance.markdown)

    tabs = {tab.label for tab in instance.tabs}
    assert any("Report" in label for label in tabs)
    assert any("Agent Conversation Log" in label for label in tabs)

    download_labels = [str(button.label) for button in instance.get("download_button")]
    assert any("PDF" in label for label in download_labels)
