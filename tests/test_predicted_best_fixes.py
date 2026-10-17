"""0.8.0 review fixes, Task 7: "Predicted to score highest" is widened by
the project's own Overall-score record and says when Best so far's score
includes luck; Make a round's predicted column sits after Formulation and
its help carries the record; one decimal rule per measurement on Insights
and Make a round; the two lines about how predictions are made (spec
2026-09-29 §3.3, §3.6, §6, §9 I6, I12, I13)."""
import math
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_insights import _burger
from tests.test_insights_ui import _open_how, _open_insights, insights_scored  # noqa: F401

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _score_check(z, sd=2.0, full=True):
    return insights.Check(1, 3, insights.SCORE, 80.0, 76.0, 84.0, 80.0 + z * sd,
                          abs(z) <= insights.Z95, full, 80.0, sd)


# ------------------------------------------------------------------ #
#  §3.6 the widening
# ------------------------------------------------------------------ #

def test_score_widen_is_the_rms_of_standardised_score_misses_never_below_one():
    assert insights.score_widen([_score_check(2.0)] * 4) == 1.0          # too few
    five = [_score_check(z) for z in (2.0, -2.0, 1.0, 1.0, 0.0)]
    assert insights.score_widen(five) == pytest.approx(math.sqrt(10 / 5))
    assert insights.score_widen([_score_check(0.1)] * 5) == 1.0          # never narrower
    partial = [_score_check(3.0, full=False)] * 5
    assert insights.score_widen(partial) == 1.0


def test_widened_keeps_the_mean_and_clips_to_the_range():
    f = insights.Fitted(insights.SCORE, None, None, 9, 0.0, 100.0)
    p = insights.Prediction(90.0, 86.08, 93.92, 90.0, 2.0)
    w = insights.widened(f, p, 2.0)
    assert w.mean == 90.0 and w.sd == 4.0
    assert (w.low, w.high) == pytest.approx((90 - insights.Z95 * 4, 90 + insights.Z95 * 4))
    top = insights.widened(f, insights.Prediction(99.0, 95.0, 100.0, 99.0, 2.0), 3.0)
    assert top.high == 100.0
    assert insights.widened(f, p, 1.0) is p


def test_the_predicted_best_and_its_repeat_are_widened_by_the_record():
    opt = _burger(rounds=5, noise=0.4)
    fitted = insights.fit_all(opt)
    checks = insights.check_rounds(opt)
    plain = insights.predicted_best(opt, fitted)
    wide = insights.predicted_best(opt, fitted, checks)
    factor = insights.score_widen(checks)
    assert wide.factor == factor and plain.factor == 1.0
    assert wide.score.mean == pytest.approx(plain.score.mean)
    assert wide.score.sd == pytest.approx(plain.score.sd * factor)
    assert wide.repeat is not None and wide.repeat.sd == pytest.approx(plain.repeat.sd * factor)


def test_best_luck_wording():
    assert wording.best_luck(30, "84", "79", "89") == (
        "A Repeat of Formulation 30 is predicted at 84 (95 % interval 79–89). Best so far's "
        "measured score is partly chance, so its Repeat is predicted lower.")


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_luck_line_shows_only_below_best_so_far(insights_scored, monkeypatch):
    """Spec 2026-09-30 §2.3: only when Best so far's predicted Repeat is
    below its measured score by more than the yardstick (1 point here)."""
    top = insights_scored.best_index()
    measured = float(FoodOptimizer("burger").Y_history[top])

    def fake(repeat):
        def predicted_best(opt, fitted, checks=()):
            return insights.PredictedBest(
                dict(opt.recipe_history[0]), insights.Prediction(80.0, 75.0, 85.0),
                None, opt.best_index(), 1.0,
                insights.Prediction(repeat, repeat - 5, repeat + 5))
        return predicted_best
    repeat = measured - 3
    monkeypatch.setattr(insights, "predicted_best", fake(repeat))
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    md = [m.value for m in at.tabs[3].markdown]
    luck = wording.best_luck(insights_scored.best_formulation_no(), f"{repeat:.0f}",
                             f"{repeat - 5:.0f}", f"{repeat + 5:.0f}", f"{measured:.2f}")
    assert luck in md
    monkeypatch.setattr(insights, "predicted_best", fake(measured - 0.5))
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not [m.value for m in at.tabs[3].markdown if "is partly chance" in m.value]


# ------------------------------------------------------------------ #
#  §9 I6 (v3: Make a round shows no record; the help's own words stay)
# ------------------------------------------------------------------ #

def test_the_help_says_fall_inside():
    # Ledger, statistician cold read I2 (fix round B): no "19 in 20" claim; the
    # plain reading gives the 95 % aim and the coverage measured on simulated projects.
    # Ftue fix A's re-run (I9 folds): 90 % of Overall scores (was 89 %).
    assert "About 19 in 20" not in wording.PREDICTED_SCORE_HELP
    assert ("Aims for 95 %; measured about 91 % (measurements) / 90 % (Overall score) on "
            "simulated projects.") in wording.PREDICTED_SCORE_HELP


# ------------------------------------------------------------------ #
#  §9 I12 decimals, §9 I13 the two lines
# ------------------------------------------------------------------ #

def test_a_measurement_prediction_shares_one_decimal_count():
    p = insights.Prediction(5.38, 3.5, 7.27)
    assert ui_insights.value_text(p, 1) == "5.4 (3.5–7.3)"
    assert ui_insights.value_text(insights.Prediction(12.66, 8.71, 18.4), 1) == "12.7 (8.7–18.4)"
    assert ui_insights.value_text(p) == "5.38 (3.5–7.27)"          # no rule given: as before


def test_how_these_insights_are_made_says_the_process_is_assumed_unchanged(insights_scored):
    at = _open_how(_open_insights(AppTest.from_file(APP_PATH, default_timeout=180)))
    [how] = [e for e in at.tabs[3].expander if e.label == wording.FOLD_METHOD]
    # One-glance Review (owner, 2026-10-14): the fold is one line per term
    # and links the README; the caveat moved to its "How the Review works",
    # Limits, word for word.
    assert wording.METHOD_README in [c.value for c in how.caption]
    section = open(README, encoding="utf-8").read().split("## How the Review works", 1)[1]
    assert ("Predictions assume your ingredients, suppliers and equipment behave as "
            "before.") in section.split("\n## ", 1)[0]


def test_the_new_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.best_luck(30, "84", "79", "89"), wording.PREDICTED_SCORE_HELP,
                 wording.PREDICTED_MEASUREMENTS_CAPTION):
        assert not any(p.search(text) for p in _BANNED), text
