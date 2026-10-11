"""Smoke test: the Streamlit app runs every question on the sample data without errors."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_ok(at):
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_question_and_sample_runs():
    at = run_ok(AppTest.from_file(APP, default_timeout=120))
    for option in at.selectbox[0].options:                       # compare: each sample dataset
        at.selectbox[0].set_value(option)
        run_ok(at)
        assert at.info or at.success or at.warning               # an answer box is shown
    at.radio[1].set_value("Type in summary numbers")
    run_ok(at)
    at.radio[0].set_value("monitor")
    run_ok(at)
    at.selectbox[0].set_value(at.selectbox[0].options[1])
    run_ok(at)
    at.radio[0].set_value("plan")
    run_ok(at)
    assert "users per group" in at.success[0].value
