"""0.8.0 review fixes, the final wave: the whole-branch review's items
(final-review.md I-1..I-3, M-1..M-5) and the parked minors."""
import pytest

import insights
import ui_insights
import wording
from tests.test_complete_results import _patty, _tell


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  I-1 the header's latest Round is the latest scored Round
# ------------------------------------------------------------------ #

def test_the_header_names_the_latest_scored_round_while_a_later_one_waits():
    """Rounds 1-2 scored, Round 3 all waiting: "up to Round 2", the Round
    "Is it still improving?" calls the last one; the waiting sentence
    still says what is pending."""
    opt = _patty(rows=6)
    for no in (7, 8, 9):
        _tell(opt, no, 3, drop=("Cook loss",))
    assert [r for r, _ in insights.scored_rounds(opt)] == [1, 2]
    assert ui_insights.header_text(opt, None).startswith(
        "Based on 6 scored formulations, up to Round 2. "
        "3 more are waiting for Cook loss. ")


def test_the_header_leaves_out_a_round_moved_to_not_scored():
    opt = _patty(rows=6)
    for no in (7, 8, 9):
        _tell(opt, no, 3, drop=("Cook loss",))
        opt.set_aside_formulation(no)
    assert ui_insights.header_text(opt, None).startswith(
        "Based on 6 scored formulations, up to Round 2. Likely range")


# ------------------------------------------------------------------ #
#  M-5 every formulation waiting: no "0 ... more", no empty question
# ------------------------------------------------------------------ #

def test_with_none_scored_the_header_says_what_is_waiting_without_more():
    assert wording.insights_header(0, None, 3, "Cook loss").startswith(
        "3 formulations are waiting for Cook loss. Likely range")
    assert wording.insights_header(0, None, 1, "Cook loss").startswith(
        "1 formulation is waiting for Cook loss. Likely range")
    # Some scored: the waiting ones are "more" than those.
    assert wording.insights_header(2, 1, 1, "Cook loss").startswith(
        "Based on 2 scored formulations, up to Round 1. 1 more is waiting for Cook loss. ")


def test_header_text_with_every_formulation_waiting():
    opt = _patty(rows=0)
    for no in (1, 2, 3):
        _tell(opt, no, 1, drop=("Cook loss",))
    assert ui_insights.header_text(opt, None).startswith(
        "3 formulations are waiting for Cook loss. ")
