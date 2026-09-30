"""0.8.0 answer-key read-through fixes (docs/superpowers/research/
2026-09-29-speed-ux/answer-key-refresh.md §7): advice that must hold on the
numbers shown, one bias rule for Insights and Make a round, Repeats left out
said as left out, the width caution beside "still rising", and Results' and
Set up's words."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _clean(*texts):
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t


# ------------------------------------------------------------------ #
#  1. "Higher" must hold on the numbers shown (Timing check: the
#     predicted best 92 below Best so far's 92.50)
# ------------------------------------------------------------------ #

def _pred(mean):
    return insights.Prediction(mean, mean - 5, mean + 5, mean, 2.5)


def _best(opt, mean, repeat):
    return insights.PredictedBest(dict(opt.recipe_history[0]), _pred(mean), None,
                                  opt.best_index(), 1.0, _pred(repeat))


def _top(opt, score):
    top = opt.best_index()
    opt.Y_history[top] = score
    return opt


def test_the_predicted_best_is_not_higher_when_it_reads_below_best_so_far():
    """Timing check: predicted best 92.15 (shown 92), Best so far's
    predicted Repeat 89.99, Best so far measured 92.50."""
    opt = _top(_patty(rows=6), 92.50)
    s = insights.best_standing(opt, _best(opt, 92.15, 89.99), 1.0)
    assert not s.higher
    assert not ui_insights.standing_text(opt, s, insights.Yardstick(1.0, None, 0, False)
                                         ).startswith("The predicted best is 92 (likely")


def test_higher_needs_a_shown_lead_beyond_the_yardstick_too():
    opt = _top(_patty(rows=6), 92.50)
    assert insights.best_standing(opt, _best(opt, 94.2, 89.99), 1.0).higher     # 94 vs 92.50
    assert not insights.best_standing(opt, _best(opt, 93.4, 89.99), 1.0).higher  # 93 vs 92.50


def test_untested_without_an_open_round_says_the_next_round():
    opt = _patty(rows=6)
    step = SimpleNamespace(kind="untested", gap=None, disagreement=None, binding=None)
    text, _ = ui_insights.next_step_text(opt, step)
    assert text.endswith("you can include it in the next Round.")
    assert "add it to the Round" not in text
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text, _ = ui_insights.next_step_text(opt, step)
    assert text.endswith("you can also add it to the Round yourself.")
    _clean(text, wording.next_untested(False, open_round=False))
