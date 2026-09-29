"""0.8.0 review fixes, Task 5: the Insights header and "Did the predictions
come true?" — scored formulations counted apart from waiting ones, a
past-tense verdict about the likely range, a wide-range warning, the
Overall score's own record always in view, and the typical miss compared
with the spread between Repeats instead of R² (spec 2026-09-29 §3.1-3.5,
§9 B1, I7, I8, I9)."""
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_insights import _burger
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _check(name="Firmness", rnd=3, measured=5.0, predicted=5.0, low=4.0, high=6.0,
           inside=True, full=True, no=1):
    return insights.Check(no, rnd, name, predicted, low, high, measured, inside, full)


# ------------------------------------------------------------------ #
#  §3.1 / §3.5 the header
# ------------------------------------------------------------------ #

def test_the_header_counts_scored_and_waiting_apart():
    assert wording.insights_header(18, 6, 2, "Cook loss", early=False) == (
        "Based on 18 scored formulations, up to Round 6. 2 more are waiting for Cook loss. "
        "Likely range: where about 19 in 20 results should land.")
    assert wording.insights_header(3, 1, 1, "Juiciness").startswith(
        "Based on 3 scored formulations, up to Round 1. 1 more is waiting for Juiciness. ")


def test_until_the_first_verdict_the_range_sentence_says_early_on():
    assert wording.insights_header(3, 1).endswith(
        "Likely range: where about 19 in 20 results should land once there are enough "
        "formulations; early on it is closer to 17 in 20.")


def test_header_text_leaves_out_waiting_and_set_aside_rows():
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    _tell(opt, 8, 3, drop=("Cook loss",))
    opt.set_aside_formulation(8)
    text = ui_insights.header_text(opt, None)
    assert text.startswith("Based on 6 scored formulations, up to Round 3. "
                           "1 more is waiting for Cook loss. ")


# ------------------------------------------------------------------ #
#  §3.2 / §9 B1, I8 the verdict and the wide-range line
# ------------------------------------------------------------------ #

def test_the_verdict_is_about_the_likely_range_in_the_past_tense():
    base = "In the last 3 rounds checked (Rounds 4–6), 24 of 26 measured values landed inside their likely range"
    assert wording.predictions_count(3, [4, 5, 6], 24, 26, "holding") == (
        base + ": the likely range proved right.")
    assert wording.predictions_count(3, [4, 5, 6], 24, 26, "caution") == (
        base + ": more misses than expected.")
    assert wording.predictions_count(3, [4, 5, 6], 24, 26, "unreliable") == (
        base + ": many more misses than expected — treat predictions with caution.")
    assert wording.predictions_count(3, [4, 5, 6], 24, 26) == base + "."


def test_a_wide_range_is_named_per_output():
    opt = _burger(rounds=1)                          # Firmness, Juiciness on 0-10
    narrow = [_check("Firmness", low=4.0, high=6.0)] * 3        # half-width 1 = 10 %
    wide = [_check("Juiciness", low=2.0, high=8.0)] * 3          # half-width 3 = 30 %
    score = [_check(insights.SCORE, low=40.0, high=90.0)] * 3    # 25 of 100 = 25 %: not over
    assert insights.wide_ranges(opt, narrow + wide + score) == ["Juiciness"]
    assert insights.wide_ranges(opt, [_check(insights.SCORE, low=20.0, high=80.0)]) == [
        insights.SCORE]
    assert insights.WIDE == 0.25


def test_the_wide_line_wording():
    assert wording.WIDE_EVERY == "Each likely range is wide, so this says little yet."
    assert wording.wide_line("Cook loss and the Overall score") == (
        "The likely range for Cook loss and the Overall score is wide, so this says little yet.")
    opt = _burger(rounds=1)
    h = insights.Headline("verdict", [3], 10, 10, "holding", None,
                          ["Firmness", "Juiciness", insights.SCORE])
    assert ui_insights._wide(opt, h) == wording.WIDE_EVERY
    h.wide = ["Juiciness", insights.SCORE]
    assert ui_insights._wide(opt, h) == wording.wide_line("Juiciness and the Overall score")


# ------------------------------------------------------------------ #
#  §3.3 / §9 I7 the Overall score's own record
# ------------------------------------------------------------------ #

def test_the_score_record_needs_five_fully_measured_checks_in_the_headline_rounds():
    four = [_check(insights.SCORE, measured=48.0, predicted=50.0)] * 4
    assert insights.score_record(four, [3]) is None
    five = four + [_check(insights.SCORE, measured=60.0, predicted=50.0, inside=False)]
    rec = insights.score_record(five, [3])
    assert (rec.inside, rec.total) == (4, 5)
    assert rec.mean_miss == pytest.approx((4 * -2.0 + 10.0) / 5)
    other_round = [_check(insights.SCORE, rnd=2)] * 5
    assert insights.score_record(other_round, [3]) is None
    partial = [_check(insights.SCORE, full=False)] * 5
    assert insights.score_record(partial, [3]) is None
    assert insights.SCORE_RECORD_MIN == 5


def test_the_score_record_line_says_which_way():
    assert wording.score_record_line(7, 8, -1.84) == (
        "Overall score: 7 of 8 inside their likely range; on average 1.8 points below prediction.")
    assert wording.score_record_line(7, 8, 0.6) == (
        "Overall score: 7 of 8 inside their likely range; on average 0.6 points above prediction.")
    assert wording.score_record_line(8, 8, 0.02) == (
        "Overall score: 8 of 8 inside their likely range; on average as predicted.")


def test_the_headline_carries_the_record_for_its_own_rounds():
    opt = _burger(rounds=5, noise=0.3)
    checks = insights.check_rounds(opt)
    h = insights.headline(opt, checks)
    want = insights.score_record(checks, h.rounds)
    assert h.score == want
    assert not hasattr(wording, "SCORE_BELOW")


# ------------------------------------------------------------------ #
#  §3.4 / §9 I9 typical miss against the spread between Repeats
# ------------------------------------------------------------------ #

def _with_repeats():
    """Formulation 1 made three times (Firmness 5.0, 5.4, 5.2) and
    Formulation 4 twice (6.0, 6.6): two groups, four pairs."""
    opt = _patty(rows=6)
    for src, dst in ((0, 1), (0, 2), (3, 4)):
        opt.recipe_history[dst] = dict(opt.recipe_history[src])
        opt.X_history[dst] = list(opt.X_history[src])
    for i, v in zip((0, 1, 2, 3, 4), (5.0, 5.4, 5.2, 6.0, 6.6)):
        opt.results_history[i]["Firmness"] = v
    opt._recompute_utilities()
    return opt


def test_repeat_sd_is_pooled_within_repeat_groups():
    opt = _with_repeats()
    spread = insights.repeat_sd(opt, "Firmness")
    ss = (0.2 ** 2 + 0.2 ** 2 + 0.0) + (0.3 ** 2 + 0.3 ** 2)
    assert spread.sd == pytest.approx((ss / (2 + 1)) ** 0.5)
    assert spread.pairs == 3 + 1
    assert insights.repeat_sd(_patty(rows=6), "Firmness") == insights.RepeatSpread(None, 0)


def test_a_chain_of_near_matches_is_one_group():
    opt = _patty(rows=3)
    base = dict(opt.recipe_history[0])
    span = 4.0                                  # Pea 4-8
    for k, step in enumerate((0.0, 0.004, 0.008)):
        opt.recipe_history[k] = dict(base, Pea=base["Pea"] + step * span)
        opt.X_history[k] = opt._encode(opt.recipe_history[k])
    groups = insights.repeat_groups(opt, [0, 1, 2])
    assert groups == [[0, 1, 2]]


def test_the_ratio_column_bands():
    assert wording.repeat_ratio(None) == "no Repeats yet"
    assert wording.repeat_ratio(None, 0.0) == "Repeats matched exactly"
    assert wording.repeat_ratio(0.6) == "about the same"
    assert wording.repeat_ratio(1.5) == "about the same"
    assert wording.repeat_ratio(2.04) == "about 2× larger"
    assert wording.repeat_ratio(3.46) == "about 3.5× larger"
    assert wording.DETAILS_COLUMNS[5] == "Typical miss compared with the spread between Repeats"


def test_repeats_that_agreed_exactly_are_not_no_repeats():
    """A Repeat whose value matched to the last digit gives a spread of 0:
    there are Repeats, and nothing to divide by (found on the demo, Line A
    Juiciness)."""
    opt = _with_repeats()
    opt.results_history[1]["Juiciness"] = opt.results_history[0]["Juiciness"]
    opt.results_history[2]["Juiciness"] = opt.results_history[0]["Juiciness"]
    opt.results_history[4]["Juiciness"] = opt.results_history[3]["Juiciness"]
    assert insights.repeat_sd(opt, "Juiciness").sd == 0.0
    [row] = insights.details(opt, [_check("Juiciness", measured=5.0, predicted=5.4)])
    assert row.repeat_sd == 0.0 and row.ratio is None
    assert wording.repeat_ratio(row.ratio, row.repeat_sd) == "Repeats matched exactly"


def test_details_compare_the_typical_miss_with_the_repeats():
    opt = _with_repeats()
    checks = [_check("Firmness", measured=5.0, predicted=5.0 + d) for d in (0.3, 0.5, 0.7)]
    [row] = insights.details(opt, checks)
    assert row.typical_miss == pytest.approx(0.5)
    assert row.ratio == pytest.approx(0.5 / insights.repeat_sd(opt, "Firmness").sd)


# ------------------------------------------------------------------ #
#  On screen
# ------------------------------------------------------------------ #

def test_the_tab_shows_the_new_header_verdict_and_record(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    tab = at.tabs[3]
    captions = [c.value for c in tab.caption]
    assert captions[0].startswith("Based on 9 scored formulations, up to Round 3. Likely range")
    text = " ".join(m.value for m in tab.markdown)
    assert "holding up" not in text and "R²" not in text


def test_the_new_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    texts = [wording.insights_header(18, 6, 2, "Cook loss"), wording.LIKELY_RANGE_HEADER,
             wording.LIKELY_RANGE_EARLY, wording.WIDE_EVERY, wording.wide_line("Cook loss"),
             wording.score_record_line(7, 8, -1.8), wording.repeat_ratio(2.0),
             wording.repeat_ratio(None, 0.0),
             wording.DETAILS_COLUMNS[5]] + [wording.predictions_count(3, [4, 5, 6], 1, 2, v)
                                            for v in wording.VERDICT]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
