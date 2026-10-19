"""Review learned (prototype, 2026-10-02): what survives of the "What has
the model learned?" section after the v3 Review. The Review draws from the
measurements alone, so the ordering accuracy, verdict, clear-effect,
forecast and Called-it tests went with the code they exercised; the guard
that no text on the Review reads a value as 100 stays."""
import re

import pytest
from streamlit.testing.v1 import AppTest

import wording
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.review_screens import _install, _leaves, _text


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-learned-ui"))
    _install(D.LINE_A)
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = D.LINE_A
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    yield at
    mp.undo()


def test_no_prediction_reads_100_on_the_review(screen):
    for el, _, _ in _leaves(screen.tabs[3]):
        if el.type in ("markdown", "caption"):
            assert not re.search(r"\(\d+–100\)|\b100 \(", _text(el)), _text(el)
