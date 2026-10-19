"""0.8.0 speed and simplicity, Plan B, Task 1: one yardstick for "more than
Repeats differ", a Repeat that disagrees, one verdict per state and the
next step, all worked out from the checks handed in (spec 2026-09-30 §2;
review I2, I4, I5, I6, M5, M6)."""
import math
from types import SimpleNamespace

import pytest

import insights
from tests.test_complete_results import _patty
from tests.test_insights import _burger
from tests.test_track_record import _with_repeats


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _check(name, rnd, no, measured=5.0, predicted=5.0, sd=0.5, full=True):
    """A Round-ahead check with its own mean and sd: inside when the miss is
    under 1.96 sd."""
    low, high = predicted - insights.Z95 * sd, predicted + insights.Z95 * sd
    return insights.Check(no, rnd, name, predicted, low, high, measured,
                          low <= measured <= high, full, predicted, sd)


def _values(rnd, n=10, no=100, miss=0.1, sd=0.5):
    """`n` measurement checks in Round `rnd`, each missing by `miss`."""
    return [_check("Firmness", rnd, no + k, 5.0 + miss, 5.0, sd) for k in range(n)]


def _repeat_of(opt, src, dst):
    """Row `dst` made as a Repeat of row `src`."""
    opt.recipe_history[dst] = dict(opt.recipe_history[src])
    opt.X_history[dst] = list(opt.X_history[src])


# ------------------------------------------------------------------ #
#  noise_yardstick (review I5)
# ------------------------------------------------------------------ #

def test_under_three_pairs_the_yardstick_is_one_point():
    opt = _patty(rows=6)
    _repeat_of(opt, 0, 3)
    y = insights.noise_yardstick(opt)
    assert (y.value, y.pairs, y.from_repeats) == (1.0, 1, False)
    assert y.sd == pytest.approx(abs(opt.Y_history[3] - opt.Y_history[0]) / math.sqrt(2))


def test_from_three_pairs_the_yardstick_is_the_repeats_sd():
    opt = _with_repeats()
    y = insights.noise_yardstick(opt)
    spread = insights.repeat_sd(opt, insights.SCORE)
    assert spread.df >= insights.YARDSTICK_DF
    assert (y.value, y.sd, y.from_repeats) == (spread.sd, spread.sd, True)


def test_a_spread_that_reads_zero_is_no_yardstick():
    opt = _with_repeats()
    for i in (1, 2):
        opt.Y_history[i] = opt.Y_history[0]
    opt.Y_history[4] = opt.Y_history[3]
    y = insights.noise_yardstick(opt)
    assert y.sd == 0.0 and (y.value, y.from_repeats) == (1.0, False)


def test_a_measurement_has_no_one_point_fallback():
    assert insights.noise_yardstick(_patty(rows=6), name="Firmness").value is None


# ------------------------------------------------------------------ #
#  disagreeing_repeat (review I4)
# ------------------------------------------------------------------ #

def _pair(later_score_gap):
    """Formulation 5 (Round 2) a Repeat of Formulation 1, scoring
    `later_score_gap` against it, with Round 2 in the verdict window and a
    Round-ahead Overall-score check for Formulation 5 of sd 2."""
    opt = _patty(rows=6)
    _repeat_of(opt, 0, 4)
    opt.Y_history[4] = opt.Y_history[0] + later_score_gap
    checks = _values(2) + [_check(insights.SCORE, 2, 5, opt.Y_history[4], opt.Y_history[4], 2.0)]
    return opt, checks


def test_a_repeat_beyond_root_two_half_widths_is_named():
    # Ftue fix A (cold read I7): √2 × 1.96 × σ, σ the assumed 1 point here
    # (no fitted noise on the check, no other Repeat): 2.77.
    opt, checks = _pair(-10.0)
    d = insights.disagreeing_repeat(opt, checks)
    assert (d.later, d.earlier, d.row) == (5, 1, 4)
    assert d.diff == pytest.approx(-10.0)


def test_a_repeat_within_root_two_half_widths_is_not():
    opt, checks = _pair(-2.7)                   # under 2.77 (ftue fix A, I7)
    assert insights.disagreeing_repeat(opt, checks) is None


def test_a_higher_repeat_counts_too():
    opt, checks = _pair(8.0)
    assert insights.disagreeing_repeat(opt, checks).diff == pytest.approx(8.0)


def test_only_a_repeat_in_the_verdict_rounds_counts():
    opt, checks = _pair(-10.0)
    later = checks + _values(3, no=200) + _values(4, no=300) + _values(5, no=400)
    assert insights.headline(opt, later).rounds == [3, 4, 5]
    assert insights.disagreeing_repeat(opt, later) is None


def test_in_a_group_of_three_each_is_compared_with_the_first():
    opt, checks = _pair(-3.0)
    _repeat_of(opt, 0, 5)
    opt.Y_history[5] = opt.Y_history[0] - 9.0
    checks.append(_check(insights.SCORE, 2, 6, opt.Y_history[5], opt.Y_history[5], 2.0))
    d = insights.disagreeing_repeat(opt, checks)
    assert (d.later, d.earlier) == (6, 1)


def test_of_two_disagreeing_repeats_in_one_group_the_larger_is_named():
    """Rows 4 and 5 both Repeats of row 0, at -7 and +12: both cross 5.54,
    and the +12 one is named although the -7 one comes first."""
    opt, checks = _pair(-7.0)
    _repeat_of(opt, 0, 5)
    opt.Y_history[5] = opt.Y_history[0] + 12.0
    checks.append(_check(insights.SCORE, 2, 6, opt.Y_history[5], opt.Y_history[5], 2.0))
    d = insights.disagreeing_repeat(opt, checks)
    assert (d.later, d.earlier, d.row) == (6, 1, 5)
    assert d.diff == pytest.approx(12.0)


@pytest.mark.parametrize("gaps", [(-7.0, 12.0), (12.0, -7.0)])
def test_of_two_disagreeing_repeats_in_two_groups_the_larger_is_named(gaps):
    """Row 3 a Repeat of row 1 and row 4 a Repeat of row 0, both crossing:
    whichever group is looked at first, the larger |d| is named."""
    opt = _patty(rows=6)
    checks = _values(2)
    for (src, dst), gap in zip(((1, 3), (0, 4)), gaps):
        _repeat_of(opt, src, dst)
        opt.Y_history[dst] = opt.Y_history[src] + gap
        checks.append(_check(insights.SCORE, 2, dst + 1, opt.Y_history[dst],
                             opt.Y_history[dst], 2.0))
    d = insights.disagreeing_repeat(opt, checks)
    assert d.diff == pytest.approx(12.0)
    assert (d.later, d.earlier) == ((4, 2) if gaps[0] == 12.0 else (5, 1))


def test_the_disagreeing_repeat_is_left_out_of_the_yardstick():
    """Ftue fix A (cold read I7) reverses this: leaving the flagged pair out
    biased the Repeat SD down, so it stays in, for every output."""
    opt, checks = _pair(-10.0)
    assert insights.disagreeing_repeat(opt, checks) is not None
    assert insights.noise_yardstick(opt).pairs == 1
    assert insights.noise_yardstick(opt, checks).pairs == 1
    assert insights.noise_yardstick(opt, checks).left_out == 0
    assert insights.noise_yardstick(opt, name="Firmness").pairs == 1
    assert insights.noise_yardstick(opt, checks, name="Firmness").pairs == 1


# ------------------------------------------------------------------ #
#  verdict_state (review I6)
# ------------------------------------------------------------------ #

def _h(state="verdict", verdict="holding", inside=10, total=10, wide=None, rounds=(2,)):
    return insights.Headline(state, list(rounds), inside, total, verdict, None, wide or [])


def test_the_verdicts_in_order():
    opt = _patty(rows=6)
    checks = _values(2)
    assert insights.verdict_state(opt, _h("none", None), []) == "none"
    assert insights.verdict_state(opt, _h("early", None), checks) == "early"
    assert insights.verdict_state(opt, _h(verdict="unreliable"), checks) == "unreliable"
    # Misses despite wide ranges are news: caution outranks wide.
    assert insights.verdict_state(opt, _h(verdict="caution", wide=["Firmness"]),
                                  checks) == "caution"
    assert insights.verdict_state(opt, _h(wide=["Firmness"]), checks) == "wide"
    assert insights.verdict_state(opt, _h(), checks) == "within"


def test_one_wide_output_of_two_keeps_the_verdict():
    opt = _patty(rows=6)
    checks = _values(2) + [_check("Juiciness", 2, 1)]
    assert insights.verdict_state(opt, _h(wide=["Firmness"]), checks) == "within"


def test_over_wide_needs_twenty_all_inside_and_small_misses():
    opt = _patty(rows=6)
    small = _values(2, n=20, miss=0.1, sd=0.5)            # |z| = 0.2
    assert insights.verdict_state(opt, _h(inside=20, total=20), small) == "over_wide"
    few = _values(2, n=19, miss=0.1, sd=0.5)
    assert insights.verdict_state(opt, _h(inside=19, total=19), few) == "within"
    big = _values(2, n=20, miss=0.4, sd=0.5)              # |z| = 0.8
    assert insights.verdict_state(opt, _h(inside=20, total=20), big) == "within"
    assert insights.verdict_state(opt, _h(inside=19, total=20), small) == "within"



def test_a_median_miss_of_exactly_point_four_is_still_over_wide():
    opt = _patty(rows=6)
    edge = [_check("Firmness", 2, 100 + k, 0.2, 0.0, 0.5) for k in range(20)]   # |z| = 0.4
    assert insights.verdict_state(opt, _h(inside=20, total=20), edge) == "over_wide"


# ------------------------------------------------------------------ #
#  next_step (review I2), one test per row of the table
# ------------------------------------------------------------------ #

def _answer(state):
    return None if state is None else insights.Improvement(state, 0.0, None, 0)


@pytest.fixture
def table(monkeypatch):
    """next_step with its collaborators stood in, so each row is set up
    directly: `set(verdict=…, gap=…, repeat=…, answer=…)`."""
    opt = _patty(rows=6)
    state = {"verdict": "within", "gap": None, "repeat": None, "answer": None, "yard": 1.0,
             "given": None, "seen": "not called"}

    def headline(o, checks):
        score = None if state["gap"] is None else insights.ScoreRecord(5, 10, state["gap"])
        return insights.Headline("verdict", [2], 10, 10, "holding", score, [])
    monkeypatch.setattr(insights, "headline", headline)
    monkeypatch.setattr(insights, "verdict_state", lambda o, h, c: state["verdict"])
    def yardstick(o, c=(), name=insights.SCORE):
        state["given"] = insights.Yardstick(state["yard"], None, 0, False)
        return state["given"]
    monkeypatch.setattr(insights, "noise_yardstick", yardstick)
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda o, c, m=None: state["repeat"])
    # A model with skill: the A3 gate (0.8.0 e2e fixes, review I4) lets the
    # rows that cite the predicted best through; tests/test_102 covers it.
    monkeypatch.setattr(insights, "skill_gate", lambda o, c, fact=None: None)

    def improvement(o, facts=None, spread=None, yardstick=None):
        state["seen"] = yardstick
        return _answer(state["answer"])
    monkeypatch.setattr(insights, "improvement", improvement)

    def run(standing=None, **kw):
        state.update(kw)
        return insights.next_step(opt, [], standing)
    run.state = state
    return run


def _standing(higher=False, tested_near=True, binding=()):
    return SimpleNamespace(higher=higher, tested_near=tested_near, binding=list(binding))


def test_row1_unreliable_with_a_score_gap_beyond_the_yardstick_carries_the_bias(table):
    """0.8.0 e2e fixes (review I2): no "bias" row; the misses row says the
    steady bias as a fact."""
    step = table(verdict="unreliable", gap=-5.0)
    assert (step.kind, step.gap) == ("misses", -5.0)


def test_row1_unreliable_without_such_a_gap_names_the_misses(table):
    assert table(verdict="unreliable", gap=-0.5).kind == "misses"
    assert table(verdict="unreliable", gap=None).kind == "misses"


def test_row2_a_disagreeing_repeat(table):
    d = insights.Disagreement(21, 11, -10.6, 20)
    step = table(verdict="within", repeat=d)
    assert (step.kind, step.disagreement) == ("repeat", d)


def test_caution_is_not_a_trigger(table):
    assert table(verdict="caution", answer="yes").kind == "rising"


@pytest.mark.parametrize("verdict", ["none", "early", "wide"])
def test_row3_too_early(table, verdict):
    assert table(_standing(higher=True, tested_near=False), verdict=verdict).kind == "early"


def test_row4_clearly_higher_and_not_tested_near(table):
    assert table(_standing(higher=True, tested_near=False), answer="yes").kind == "untested"


def test_row5_close_to_the_top_names_what_binds(table):
    bind = SimpleNamespace(kind="highest", name="Coconut oil", text="")
    step = table(_standing(binding=[bind]), answer="not_lately")
    assert (step.kind, step.binding) == ("close", bind)


def test_row6_close_to_the_top_with_nothing_binding(table):
    step = table(_standing(), answer="barely")
    assert (step.kind, step.binding) == ("close", None)


def test_row7_yes_is_still_rising(table):
    assert table(_standing(), answer="yes").kind == "rising"


def test_rising_is_judged_against_the_same_yardstick(table):
    """Line 1 and the next step ask improvement() with one yardstick (I5)."""
    table(_standing(), answer="yes")
    assert table.state["seen"] is table.state["given"] is not None


def test_an_exactly_limit_leaves_row6_to_show(table):
    """An Exactly limit is never what binds (Task 2 review I-1), so a
    predicted best at it is "close to the top" with nothing to widen."""
    opt = _burger(rounds=2)
    opt.add_quantity_constraint(["Pea", "Oil"], exactly=10.0)
    top = float(opt.Y_history[opt.best_index()])
    p = insights.Prediction(top, top - 4, top + 4, top, 2.0)
    best = insights.PredictedBest(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), p, 1,
                                  opt.best_index(), 1.0, p)
    standing = insights.best_standing(opt, best, 1.0)
    assert standing.binding == [] and not standing.higher
    step = table(standing, answer="not_lately")
    assert (step.kind, step.binding) == ("close", None)


@pytest.mark.parametrize("standing,answer", [
    (_standing(), None),                                   # improvement None
    (_standing(higher=True, tested_near=True), "not_lately"),  # higher but tested near
    (None, "not_lately"),                                  # no predicted best to go on
])
def test_row8_anything_else(table, standing, answer):
    assert table(standing, answer=answer).kind == "see"


# ------------------------------------------------------------------ #
#  improvement with the yardstick, and the details table
# ------------------------------------------------------------------ #

def test_barely_is_judged_against_one_point_under_three_pairs():
    opt = _patty(rows=6)
    yard = insights.Yardstick(1.0, 0.3, 1, False)
    f = insights.BestFacts(71.0, 6, 2, 0, True, [])
    earlier = [i for i in opt.best_candidates() if opt.batch_history[i] != 2]
    for i in earlier:
        opt.Y_history[i] = 70.5
    a = insights.improvement(opt, f, yardstick=yard)
    assert (a.state, a.sd) == ("barely", None)


def test_barely_uses_the_repeats_sd_from_three_pairs():
    """Ftue fix A (cold read I6), with the fix A follow-up's decision 3
    (1e065e1): against the Repeat SD × √2 × t_q on the Repeats' 4 df,
    t_4⁻¹(0.95^(1/3)) = 3.17 (Round 2 has 3 formulations): 0.3 × √2 × 3.17
    = 1.34. A rise of 0.5 is within noise, 1.5 is Rising."""
    opt = _patty(rows=6)
    yard = insights.Yardstick(0.3, 0.3, 4, True, df=4)
    f = insights.BestFacts(71.0, 6, 2, 0, True, [])
    for i in opt.best_candidates():
        if opt.batch_history[i] != 2:
            opt.Y_history[i] = 70.5
    a = insights.improvement(opt, f, yardstick=yard)
    assert (a.state, a.sd, a.q, a.df) == ("barely", 0.3, 3, 4)
    assert a.threshold == pytest.approx(1.344, abs=0.001)
    for i in opt.best_candidates():
        if opt.batch_history[i] != 2:
            opt.Y_history[i] = 69.5
    assert insights.improvement(opt, f, yardstick=yard).state == "yes"


def test_details_compare_like_with_like_and_show_the_widening():
    opt = _with_repeats()
    checks = [_check("Firmness", 3, 1, 5.0, 5.0 + d) for d in (0.3, 0.5, 0.7)]
    fitted = {"Firmness": insights.Fitted("Firmness", object(), None, 6, 0, 10, 2.1)}
    [row] = insights.details(opt, checks, fitted)
    # Median miss 0.5 over MAD_OF_NORMAL × the pooled sd sqrt(0.26 / 3)
    # (test_track_record pins that sd): a known number, not the formula.
    assert row.ratio == pytest.approx(2.5181, abs=1e-4)
    assert (row.widen, row.pairs) == (2.1, 4)
    [bare] = insights.details(opt, checks)
    assert bare.widen is None
    unfitted = {"Firmness": insights.Fitted("Firmness", None, None, 6, 0, 10, 2.0)}
    assert insights.details(opt, checks, unfitted)[0].widen is None


def test_details_leave_a_disagreeing_repeat_out_of_every_output():
    opt, checks = _pair(-10.0)
    flagged = insights.disagreeing_repeat(opt, checks)
    rows = {r.name: r for r in insights.details(opt, checks, flagged=flagged)}
    # Ftue fix A (cold read I7): the flagged pair now stays in every output.
    assert rows["Firmness"].pairs == 1 and rows[insights.SCORE].pairs == 1
    assert rows["Firmness"].left_out == 0
    kept = {r.name: r for r in insights.details(opt, checks)}
    assert kept["Firmness"].pairs == 1 and kept[insights.SCORE].pairs == 1
