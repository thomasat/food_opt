"""0.8.0 review fixes, Task 5: the Insights header and "Did the predictions
come true?" — scored formulations counted apart from waiting ones, a
past-tense verdict about the likely range, a wide-range warning, the
Overall score's own record always in view, and the typical miss compared
with the spread between Repeats instead of R² (spec 2026-09-29 §3.1-3.5,
§9 B1, I7, I8, I9)."""
import pytest
from tests.demo_build_reason import REVIEW_FOUR
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

def test_header_text_leaves_out_waiting_and_set_aside_rows():
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    _tell(opt, 8, 3, drop=("Cook loss",))
    opt.set_aside_formulation(8)
    text = ui_insights.header_text(opt)
    assert text == ("Based on 6 scored formulations (all results in), up to Round 2. "
                    "Formulation 7 is waiting for Cook loss and is not used yet.")


def _recorded(opt, complete, waiting):
    """`complete` formulations with both results in Round 1, then `waiting`
    with Firmness only (Juiciness to come) in Round 2."""
    for no in range(1, complete + waiting + 1):
        values = {"Firmness": 3.0 + 0.3 * no}
        if no <= complete:
            values["Juiciness"] = 4.0 + 0.2 * no
        opt.tell({"Pea protein": 8.0 + no, "Methylcellulose": 0.5 + 0.2 * no}, values,
                 formulation_no=no, batch_no=1 if no <= complete else 2)


@pytest.mark.parametrize("complete", [3, 4])
def test_the_header_and_the_gate_count_the_same_scored_formulations(burger, complete):
    """Review I-1: with one of 4 (or 5) recorded formulations waiting, the
    header and "So far: N" both count the scored ones, and a waiting one
    does not open the checks early."""
    _recorded(burger, complete, 1)
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    captions = [c.value for c in at.tabs[3].caption]
    assert captions[0] == (
        f"Based on {complete} scored formulations (all results in), up to Round 1. "
        f"Formulation {complete + 1} is waiting for Juiciness and is not used yet."), captions
    # Said once (Task 6), by the Predictions status line since ftue W1.
    gates = [m.value for m in at.tabs[3].markdown if "once 5 formulations have results" in m.value]
    assert gates == [], gates


# ------------------------------------------------------------------ #
#  §3.2 / §9 B1, I8 the verdict and the wide-range line
# ------------------------------------------------------------------ #

def test_a_wide_range_is_named_per_output():
    opt = _burger(rounds=1)                          # Firmness, Juiciness on 0-10
    narrow = [_check("Firmness", low=4.0, high=6.0)] * 3        # half-width 1 = 10 %
    wide = [_check("Juiciness", low=2.0, high=8.0)] * 3          # half-width 3 = 30 %
    score = [_check(insights.SCORE, low=40.0, high=90.0)] * 3    # 25 of 100 = 25 %: not over
    assert insights.wide_ranges(opt, narrow + wide + score) == ["Juiciness"]
    assert insights.wide_ranges(opt, [_check(insights.SCORE, low=20.0, high=80.0)]) == [
        insights.SCORE]
    assert insights.WIDE == 0.25


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


def test_the_headline_carries_the_record_for_every_round_checked():
    """0.8.0 e2e fixes (review I1): the Overall score's record spans every
    Round checked, not the verdict's window."""
    opt = _burger(rounds=5, noise=0.3)
    checks = insights.check_rounds(opt)
    h = insights.headline(opt, checks)
    want = insights.score_record(checks, None)
    assert want.rounds == insights._ordered_rounds(
        [c for c in checks if c.name == insights.SCORE])
    assert h.score is not None and h.score.total >= insights.SCORE_RECORD_MIN
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


def test_details_compare_the_recipes_once_for_every_output(monkeypatch):
    """Review M-4: the O(n²) recipe comparisons run once per details call,
    and give each output the same spread as its own repeat_sd."""
    opt = _with_repeats()
    calls = []
    real = insights._matches
    monkeypatch.setattr(insights, "_matches",
                        lambda o, rows: calls.append(rows) or real(o, rows))
    checks = [_check(n, measured=5.0, predicted=5.4)
              for n in ("Firmness", "Juiciness", insights.SCORE)]
    rows = insights.details(opt, checks)
    assert len(calls) == 1
    monkeypatch.setattr(insights, "_matches", real)
    assert [r.repeat_sd for r in rows] == [insights.repeat_sd(opt, r.name).sd for r in rows]


def test_details_compare_the_typical_miss_with_the_repeats():
    opt = _with_repeats()
    checks = [_check("Firmness", measured=5.0, predicted=5.0 + d) for d in (0.3, 0.5, 0.7)]
    [row] = insights.details(opt, checks)
    assert row.typical_miss == pytest.approx(0.5)
    # Like with like (spec 2026-09-30 review M6): a median miss against the
    # median difference a normal spread of this sd gives.
    assert row.ratio == pytest.approx(
        0.5 / (insights.MAD_OF_NORMAL * insights.repeat_sd(opt, "Firmness").sd))


# ------------------------------------------------------------------ #
#  On screen
# ------------------------------------------------------------------ #

def test_the_tab_shows_the_new_header_verdict_and_record(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    tab = at.tabs[3]
    captions = [c.value for c in tab.caption]
    assert captions[0] == "Based on 9 scored formulations (all results in), up to Round 3."
    text = " ".join(m.value for m in tab.markdown)
    # R² is named once, in Method and details, as the statistic not used
    # (review M6).


@pytest.fixture
def verdict_project():
    """Six Rounds of a noisy _burger: a verdict (holding), Firmness and
    Juiciness with wide likely ranges, the Overall score's not, and 9
    fully measured Overall-score checks in the headline's Rounds. Noise
    1.5, not 2.0, since ftue fix A's Round-grouped folds (cold read I9)
    widen the Overall score's range at 2.0 too, and every range wide is a
    verdict of its own."""
    return _burger(rounds=6, noise=1.5)


