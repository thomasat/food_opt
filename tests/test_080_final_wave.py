"""0.8.0 final whole-branch fix wave (final review I-2, M-1, M-2; Plan B
parked items): what the "How" table says about the Overall score's
widening, "What to do next" while a Round is open, one factor for the
predicted best and Make a round's column."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  Final review I-2 (ruling M5): the Overall-score row shows both
#  widenings, as the ranges on screen carry both
# ------------------------------------------------------------------ #

def test_the_overall_score_row_shows_both_widenings(monkeypatch):
    opt = _burger(rounds=5, noise=0.6)
    monkeypatch.setattr(insights, "column_factor", lambda o, c: 2.5)
    score = SimpleNamespace(name=insights.SCORE, widen=1.3)
    firm = SimpleNamespace(name="Firmness", widen=1.3)
    assert ui_insights.widen_cell(opt, score, []) == "×1.3 · ×2.5 at the top"
    assert ui_insights.widen_cell(opt, firm, []) == "×1.3"
    monkeypatch.setattr(insights, "column_factor", lambda o, c: 1.0)
    assert ui_insights.widen_cell(opt, score, []) == "×1.3"          # nothing on top
    assert ui_insights.widen_cell(opt, SimpleNamespace(name=insights.SCORE, widen=None),
                                  []) == ""


def test_the_how_table_carries_the_factor_the_screens_use():
    opt = _burger(rounds=5, noise=0.6)
    checks = insights.check_rounds(opt)
    top = insights.column_factor(opt, checks)
    assert top > 1.0
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    [table] = [d for d in at.tabs[3].dataframe
               if wording.DETAILS_COLUMNS_2[5] in d.value.columns]
    cells = dict(zip(table.value[wording.DETAILS_COLUMNS_2[0]],
                     table.value[wording.DETAILS_COLUMNS_2[5]]))
    [score_cell] = [v for k, v in cells.items() if k == wording.OVERALL_SCORE_COLUMN]
    assert score_cell.endswith(f"· {wording.widen_text(top)} at the top")
    captions = [c.value for c in at.tabs[3].caption]
    assert wording.HOW_WIDENED_SCORE in captions


def test_the_widening_words_pass_the_guard():
    for text in (wording.score_widen_text(1.3, 2.5), wording.HOW_WIDENED_SCORE):
        assert not any(p.search(text) for p in _BANNED), text


# ------------------------------------------------------------------ #
#  Final review M-2: mid-Round, answer 2's predicted best and Make a
#  round's column carry one factor
# ------------------------------------------------------------------ #

def test_the_predicted_best_is_widened_as_the_column_is_mid_round():
    opt = _burger(rounds=5, noise=0.6)
    checks = insights.check_rounds(opt)
    fitted = {insights.SCORE: insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))}
    opt.pending_batch_no = 5                  # Round 5 open, its rows recorded
    best = insights.predicted_best(opt, fitted, checks)
    assert best.factor == insights.column_factor(opt, checks)
    assert best.factor != insights.score_widen(checks)
