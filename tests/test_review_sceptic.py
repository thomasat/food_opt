"""Sceptic review (owner-approved, 2026-10-14): two expert reviewers found
every number right and some words claiming more than the data shows. The
stop says what is within noise (the Overall score, of 100) and whether each
off-target measurement is beyond its own noise band; the tasting noise
says its unit and what 1 point is per measurement; the Forecast check says
its count and its reference; a 95 % range over a measurement's whole Range
is "No forecast", counted nowhere as in or out; a verdict's pairs are
called of all with how many right; the stop's Set up "?" checks first; the
rebuilt forecasts' "?" says what they are (never "today").

Synthetic projects only (test_final_demo's yogurt-shaped panel)."""
import dataclasses
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import ui_insights as U
import wording
from tests.test_final_demo import ROWS, _FakeSt, _clean, _facts, _opt, _words


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  1 · the stop box: the score within noise of 100; each miss vs its band
# ------------------------------------------------------------------ #

def test_the_stop_head_names_the_score_and_the_top():
    assert wording.stop_callout_head("100") == "Stop: score within noise of 100"
    assert wording.STOP_CALLOUT_HEAD == "Stop: score within noise of 100"
    assert U.stop_head_text(_opt()) == "Stop: score within noise of 100"
    assert _words(wording.STOP_CALLOUT_HEAD) <= 10


def test_the_on_target_line_says_beyond_or_within_noise():
    one = wording.stop_on_target(3, 4, ["Tangy 1.0 under"], True, [True])
    assert one == "On target 3 of 4; Tangy 1.0 under, beyond noise."
    assert wording.stop_on_target(3, 4, ["Tangy 1.0 under"], True, [False]) == (
        "On target 3 of 4; Tangy 1.0 under, within noise.")
    # Fix round 2: two or more off target, the counts alone (names and
    # amounts in the "?"), at most 10 words.
    offs = ["Tangy 1.0 under", "Firmness 0.5 over"]
    both = wording.stop_on_target(2, 4, offs, True, [True, True])
    assert both == "On target 2 of 4; 2 beyond noise."
    mixed = wording.stop_on_target(2, 4, offs, True, [True, False])
    assert mixed == "On target 2 of 4; 1 beyond, 1 within noise."
    inside = wording.stop_on_target(1, 4, offs + ["Sweet 0.2 over"], True, [False] * 3)
    assert inside == "On target 1 of 4; 3 within noise."
    # A measurement with no noise: nothing said about its noise.
    part = wording.stop_on_target(2, 4, offs, True, [True, None])
    assert part == "On target 2 of 4; 1 beyond noise."
    assert wording.stop_on_target(2, 4, offs) == "On target 2 of 4; 2 off target."
    assert wording.stop_on_target(3, 4, ["Tangy 1.0 under"]) == "On target 3 of 4; Tangy 1.0 under."
    assert wording.stop_on_target(3, 4, ["Tangy 1.0 under"], True, [None]) == (
        "On target 3 of 4; Tangy 1.0 under.")
    assert wording.stop_on_target(4, 4, [], True, []) == "On target 4 of 4."
    # One off target is always named, with its gap and noise verdict: the
    # owner's specific-wording rule outranks a one-word overage (controller
    # ruling), so a two-word name may take the line to 11 words.
    long_ = wording.stop_on_target(3, 4, ["Thickness uniformity 0.5 under"], True, [False])
    assert long_ == "On target 3 of 4; Thickness uniformity 0.5 under, within noise."
    assert _words(long_) <= 11
    for t in (one, both, mixed, inside, part):
        assert _words(t) <= 10, t
    for t in (one, both, mixed, inside, part, long_):
        _clean(t)


@pytest.mark.parametrize("sigma, side, half", [(1, "beyond", "0.6"), (0.5, "beyond", "0.3"),
                                               (2, "within", "1.2"), (3, "within", "1.8")])
def test_the_stop_line_and_its_question_mark_agree(sigma, side, half):
    """Tangy is 1.0 under; its band is ±1.96 × 0.3 × σ: beyond it at 0.5 and 1
    point, within it at 2 and 3. The line and the "?" say the same."""
    opt = _opt(noise=None if sigma == 1 else sigma)
    line = U.stop_reason_text(opt)
    assert line == f"On target 3 of 4; Tangy 1.0 under, {side} noise."
    res = opt.results_history[opt.best_index()]
    tip = U.on_target_help_text(opt, res)
    within = 3 if side == "beyond" else 4
    assert f"Within noise of target (±{half}, where 95 % of single tastings land): " \
           f"{within} of 4." in tip
    assert ("Beyond its noise: Tangy 1.0 under (±" + half + ")." in tip) == (side == "beyond")
    # The band is the same one within_noise_of_target counts by.
    assert U.noise_half(opt, "Tangy") == pytest.approx(insights.Z95 * 0.3 * sigma)


def test_the_callout_draws_the_new_head_and_line(monkeypatch):
    opt = _opt()
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    U._stop_callout(opt, _facts(opt))
    md = [c[1][0] for c in fake.calls if c[0] == "markdown"]
    assert md[0] == "**Stop: score within noise of 100**"
    assert md[2] == "On target 3 of 4; Tangy 1.0 under, beyond noise."


# ------------------------------------------------------------------ #
#  2 · the tasting noise: its unit, per measurement, and the stop's floor
# ------------------------------------------------------------------ #

def test_one_point_is_said_in_each_measurements_units():
    opt = _opt()
    # σ_a = σ × width ÷ (share × √k) = 1 × 15 ÷ (25 × 2) = 0.3.
    assert learned.derived_noise(opt, "Tangy", 1.0) == pytest.approx(0.3)
    assert U.noise_unit_text(opt) == (
        "Here 1 point = 0.3 per measurement, in its own units (Range 0–15).")
    # Different Ranges: each named, rounded values marked.
    assert wording.noise_per_measurement([("A", "0.3", "0–15"), ("B", "≈0.67", "0–10 N")]) == (
        "Here 1 point ≈ A 0.3, B 0.67, each in its own units.")
    assert wording.noise_per_measurement([]) == ""


def test_a_unit_joins_the_value_and_the_range(tmp_path):
    from food_bo import FoodOptimizer
    opt = FoodOptimizer("units")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_objective("Firmness", 50, goal="target", target=5, min_val=0, max_val=10, unit="N")
    opt.add_objective("Juiciness", 50, goal="target", target=5, min_val=0, max_val=10, unit="/10")
    text = U.noise_unit_text(opt)
    # 1 × 10 ÷ (50 × √2) = 0.1414…: rounded, so '≈'; 'N' joins, '/10' does not.
    assert text == "Here 1 point ≈ Firmness 0.14 N, Juiciness 0.14, each in its own units."


@pytest.mark.parametrize("sigma", [0.5, 1, 2, 3])
def test_the_noise_question_mark_says_the_unit_and_where_the_stop_starts(sigma, monkeypatch):
    opt = _opt(noise=None if sigma == 1 else sigma)
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "confirmation_open", lambda: False)
    U._noise_control(opt)
    help_ = next(c for c in fake.calls if c[0] == "markdown")[2]["help"]
    assert help_.startswith("Points of the Overall score (0–100), 1 SD of a single tasting. "
                            "Here 1 point = 0.3 per measurement, in its own units (Range 0–15).")
    # The stop's floor does not move with the setting: it is where the stop starts.
    assert help_.endswith("The stop shows at noise over 0.72 points.")
    # The option labels are unchanged.
    sb = next(c for c in fake.calls if c[0] == "selectbox")
    assert [sb[2]["format_func"](v) for v in sb[1][1]] == [
        "0.5 points", "1 point (assumed)", "2 points", "3 points"]
    _clean(help_)


def test_the_stop_floor_is_the_stops_own_rule_solved_for_the_noise():
    opt = _opt()
    yard = insights.noise_yardstick(opt)
    floor = U.stop_noise_floor(opt, yard)
    gap = round(100 - float(opt.Y_history[opt.best_index()]), 2)
    assert gap == 1.67
    holds = lambda s: gap < round(insights.rise_threshold(s, 1), 2)  # noqa: E731
    assert holds(floor) and not holds(floor - 1e-6)
    assert f"{floor:.2f}" == "0.72"
    # The rule the screen uses agrees on each side of the floor.
    for s, want in ((floor + 1e-3, True), (floor - 1e-3, False), (3.0, True), (0.5, False)):
        insights.set_tasting_noise(opt, s)
        facts = NS(yardstick=insights.noise_yardstick(opt),
                   best=NS(score=float(opt.Y_history[opt.best_index()]), formulation=16))
        assert U.at_ceiling(opt, facts) is want, s


def test_no_stop_floor_where_the_stop_rests_on_more_than_its_rule(monkeypatch):
    opt = _opt()
    yard = insights.noise_yardstick(opt)
    assert U.stop_noise_floor(opt, None) is None
    assert U.stop_noise_floor(opt, NS(from_repeats=True, groups=0)) is None
    assert U.stop_noise_floor(opt, NS(from_repeats=False, groups=1)) is None   # a Repeat made
    monkeypatch.setattr(U, "open_rows", lambda o: [{"formulation": 19}])
    opt.pending_batch = [{"formulation": 19}]
    assert U.stop_noise_floor(opt, yard) is None                                # a Round open
    few = _opt("few", rows=ROWS[:4])
    assert U.stop_noise_floor(few, insights.noise_yardstick(few)) is None       # at the gate


# ------------------------------------------------------------------ #
#  3 · the Forecast check: the count and the reference
# ------------------------------------------------------------------ #

def test_the_forecast_check_says_how_many_inside_what():
    assert wording.check_line(10, 3, 0, 0, "order right: 1 pair, 1 too close, 1 tie") == (
        "**Round 10: all 3 inside their 95 % range.** Order right: 1 pair, 1 too close, 1 tie.")
    assert wording.check_line(6, 1, 0, 0) == "**Round 6: inside its 95 % range.**"
    assert wording.check_line(6, 2, 0, 0, "", "Consistency") == (
        "**Round 6: Consistency, both inside their 95 % range.**")
    # No range left: the results named, and no order beside it.
    assert wording.check_line(6, 0, 0, 0, "order right: 1 pair", "Creaminess", none=3) == (
        "**Round 6: Creaminess, 3 values with no forecast range.**")
    # A miss keeps its old words and its range part.
    assert wording.check_line(6, 1, 2, 0, "order right: 1 pair") == (
        "**Round 6: scored higher than forecast.** 2 of 3 above their range; order right: 1 pair.")
    head = re.sub(r"\*\*", "", wording.check_head(10, 0, 0, total=3))
    assert _words(head) <= 10
    assert "as forecast" not in wording.check_line(10, 3, 0, 0)
    assert not hasattr(wording, "called_landing")


# ------------------------------------------------------------------ #
#  4 · a 95 % range over the whole Range: No forecast
# ------------------------------------------------------------------ #

def _check(name, low, high, measured=5.0, predicted=6.0, r=6, no=16):
    return insights.Check(no, r, name, predicted, low, high, measured, low <= measured <= high,
                          True, raw_mean=predicted, sd=1.0)


@pytest.mark.parametrize("low, high, none", [
    (0.0, 15.0, True),          # the whole Range
    (0.15, 14.85, True),        # each end exactly 1 % of 15 in
    (0.1, 14.9, True),
    (0.16, 15.0, False),        # the low end more than 1 % in
    (0.0, 14.84, False),        # the high end more than 1 % in
    (2.1, 10.3, False),         # a real range
])
def test_whole_range_is_each_end_within_one_percent(low, high, none):
    opt = _opt()
    assert U.no_forecast(opt, _check("Creaminess", low, high)) is none
    assert wording.NO_FORECAST_EDGE == 0.01


def test_the_results_table_reads_no_forecast_and_no_verdict():
    opt = _opt()
    checks = [_check(insights.SCORE, 91.9, 96.8, 98.3, 94.3),
              _check("Creaminess", 0.0, 15.0, 5.0, 5.7),
              _check("Tangy", 2.1, 10.3, 4.0, 6.2)]
    frame = U.round_predicted_frame(opt, checks, 6)
    rows = {r["Result"]: r for r in frame.to_dict("records")}
    pred, landed = wording.RESULTS_PREDICTED_COLUMN, wording.RESULTS_LANDED_COLUMN
    assert rows["Creaminess"][pred] == "No forecast" and rows["Creaminess"][landed] == "—"
    assert rows["Tangy"][pred] == "6.2 (2.1–10.3)" and rows["Tangy"][landed] == "In"
    assert rows["Overall score"][landed] == "Above"
    assert _words(wording.NO_FORECAST) <= 5
    # Fix round 2: the fact, never a cause; widened ×N when it was.
    items = U.no_forecast_items(opt, checks, 6)
    assert items == [("Creaminess", "0–15", None)]
    help_ = wording.results_predicted_help(6, items)
    assert help_.endswith("No forecast: the 95 % range covers the whole Range (Creaminess "
                          "0–15; each end within 1 % of the Range's width of its end), so it "
                          "can't be in or out.")
    assert "too few" not in help_ and "No forecast" not in wording.results_predicted_help(6)
    wide = [dataclasses.replace(c, widen=2.13) for c in checks]
    assert U.no_forecast_items(opt, wide, 6) == [("Creaminess", "0–15", "2.1")]
    assert wording.no_forecast_help([("Creaminess", "0–15", "2.1")]).endswith(
        "Widened ×2.1 after earlier misses.")
    _clean(help_)


def test_a_no_forecast_is_counted_neither_in_nor_outside(monkeypatch):
    """The Forecast check's line, led by a measurement: its whole-Range
    forecasts are left out of the in/outside count."""
    monkeypatch.setattr(U, "_short", lambda opt, text: text)
    opt = _opt()
    cks = [_check("Creaminess", 0.0, 15.0, 5.0, no=16), _check("Creaminess", 0.0, 15.0, 5.0, no=17),
           _check("Creaminess", 4.0, 6.0, 7.0, no=18)]
    call = learned.round_call(cks, "Creaminess", 6, 0.3)
    lf = NS(items=[("Creaminess", call, cks)], called=(6, []))
    line = U.called_line(opt, lf)
    assert line.startswith("**Round 6: Creaminess measured higher than forecast.** Above its range")
    every = [_check("Creaminess", 0.0, 15.0, 5.0, no=16)]
    lf = NS(items=[("Creaminess", learned.round_call(every, "Creaminess", 6, 0.3), every)],
            called=(6, []))
    assert U.called_line(opt, lf) == "**Round 6: Creaminess, 1 value with no forecast range.**"


# ------------------------------------------------------------------ #
#  5 · a verdict's pairs: called of all
# ------------------------------------------------------------------ #

def test_the_verdicts_say_called_of_all_pairs():
    counts = {"Consistency": (11, 11, 24), "Tangy": (5, 5, 9), "Overall score": (5, 5, 12),
              "Creaminess": (1, 5, 12)}
    # Fix round 2: one form everywhere, never hiding how many were right.
    assert wording.learned_strip_counted(["Consistency"], [], [], counts) == (
        "Predicts:", "Consistency (11 of 24 called, 11 right)")
    assert wording.learned_strip_counted([], ["Tangy"], [], counts) == (
        "Partly:", "Tangy (5 of 9 called, 5 right)")
    word, line = wording.learned_strip_counted([], ["Overall score", "Tangy"], [], counts)
    assert line == "Overall score (5 of 12 called, 5 right), Tangy"
    assert _words(f"{word} {line}") <= 10
    # Past 10 words, the first name and '+N'.
    word, line = wording.learned_strip_counted(
        [], ["Overall score", "Thickness uniformity"], [], dict(counts, **{
            "Thickness uniformity": (2, 3, 9)}))
    assert line == "Overall score (5 of 12 called, 5 right) +1"
    assert _words(f"{word} {line}") <= 10
    assert wording.named_pairs(["Overall score", "Tangy"], counts) == (
        "Overall score (5 of 12 pairs called, 5 right), Tangy (5 of 9 called, 5 right)")
    assert wording.not_yet_counted(["Creaminess"], counts) == (
        "Not yet: Creaminess (5 of 12 called, 1 right).")
    assert wording.called_of(7, 9, 24) == "9 of 24 called, 7 right"
    assert "Called: forecasts at least one tasting noise apart and measured values unequal; " \
           "counted of all pairs." in wording.learned_verdicts_help(5)
    # Beside the longest pairs line of the samples (27 words), the "?" still
    # fits its ~120 words, so brief() keeps the definitions.
    pairs = ("Overall score: 9 of 24 pairs called (13 too close, 2 ties); 7 right. Chance by "
             "luck: 10 in 100; 31 in 100 adjusted for checking 5 results.")
    both = f"{pairs}\n\n{wording.learned_verdicts_help(5)}"
    assert wording.brief(both) == both
    for t in (line, wording.learned_verdicts_help(5)):
        _clean(t)


def test_pair_counts_carry_every_pair_of_a_round():
    o = learned.Ordering("Tangy", right=5, pairs=5, ties=1, close=3)
    assert U.pair_counts(NS(), {"Tangy": o}) == {"Tangy": (5, 5, 9)}


# ------------------------------------------------------------------ #
#  6 · the stop's advice: check first
# ------------------------------------------------------------------ #

def test_the_stops_set_up_question_mark_checks_first(monkeypatch):
    opt = _opt()
    text = U.ceiling_setup_text(opt)
    assert text == ("To check the stop, repeat Formulation 16: 3 Repeats give a first estimate "
                    "of your tasting noise (assumed at 1 point until then). Changing a target or "
                    "Share of score changes what “best” means.")
    insights.set_tasting_noise(opt, 2.0)
    assert "(assumed at 2 points until then)" in U.ceiling_setup_text(opt)
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "confirmation_open", lambda: False)
    U._ceiling_foot(opt)
    [btn] = [c for c in fake.calls if c[0] == "button"]
    assert btn[1][0] == wording.SETUP_BUTTON_SHORT
    assert btn[2]["help"].startswith("To check the stop, repeat Formulation 16")
    assert "chase" not in wording.CEILING_MEASURE
    for t in (text, wording.CEILING_SETUP_HELP, wording.CEILING_MEASURE):
        _clean(t)


def test_rebuilt_label_is_as_before_and_its_question_mark_says_what_it_is():
    # Fix round 1: "today" could be false; the "?" says exactly what it is.
    assert wording.REBUILT_LABEL == "Rebuilt from earlier Rounds"
    assert "today" not in wording.rebuilt_help(True)
    assert _words(wording.REBUILT_LABEL) <= 5
    assert _words(wording.RESULTS_PREDICTED_CAPTION) <= 8


# ------------------------------------------------------------------ #
#  Fix round 1 · no forecast is counted nowhere as inside or outside
# ------------------------------------------------------------------ #

def _round_of(r, whole, missed=0, name="Creaminess", start=1):
    """Three checks of `name` in Round `r`: whole-Range ranges when
    `whole`, else 4–6 with `missed` of them measured outside."""
    out = []
    for k in range(3):
        low, high = (0.0, 15.0) if whole else (4.0, 6.0)
        measured = 7.0 if k < missed else 5.0
        out.append(_check(name, low, high, measured, 5.0, r=r, no=start + k))
    return out


def test_check_rounds_marks_each_whole_range_check():
    opt = _opt()
    marked = insights.mark_no_forecast(opt, [_check("Tangy", 0.0, 15.0), _check("Tangy", 0.1, 14.9),
                                             _check("Tangy", 2.1, 10.3),
                                             _check(insights.SCORE, 0.0, 100.0, 90.0, 90.0)])
    assert [c.no_forecast for c in marked] == [True, True, False, True]
    assert insights.NO_FORECAST_EDGE == wording.NO_FORECAST_EDGE == 0.01
    assert insights.ranged(marked) == [marked[2]]


def test_check_rounds_returns_checks_marked_on_their_final_range(monkeypatch):
    """Behavioural (fix round 2): check_rounds marks after its widening, so
    a range widened to the whole Range is no forecast."""
    opt = _opt()
    fake = {r: [_check("Tangy", 2.0, 8.0, 5.0, 5.0, r=r, no=3 * r + k) for k in range(3)]
            for r in (2, 3, 4, 5, 6)}
    fake[6][0] = _check("Tangy", 0.0, 15.0, 5.0, 5.0, r=6, no=18)
    monkeypatch.setattr(insights, "_check_round", lambda o, r, first, names: list(fake.get(r, [])))
    got = insights.check_rounds(opt)
    whole = [(c.round, c.formulation) for c in got if c.no_forecast]
    assert (6, 18) in whole
    assert all(c.no_forecast == U.no_forecast(opt, c) for c in got)


def test_the_coverage_counts_leave_no_forecast_out():
    opt = _opt()
    checks = insights.mark_no_forecast(opt, (
        _round_of(3, False, 1, start=1) + _round_of(4, False, 1, start=4)
        + _round_of(5, True, 0, start=7) + _round_of(5, False, 0, "Tangy", start=7)
        + _round_of(3, False, 0, "Tangy", start=1) + _round_of(4, False, 0, "Tangy", start=4)))
    h = insights.headline(opt, checks)
    # The window is still the last three Rounds checked; Round 5's three
    # Creaminess checks are no forecast, so 15 counted, not 18.
    assert h.rounds == [3, 4, 5] and (h.inside, h.total) == (13, 15)
    vm = insights.value_misses(opt, checks, h)
    assert (vm.misses, vm.total) == (2, 15) and vm.items == [("Creaminess", 2, 6)]
    rows = {r.name: (r.checked, r.inside) for r in insights.details(opt, checks)}
    assert rows["Creaminess"] == (6, 4) and rows["Tangy"] == (9, 9)
    # The Overall score's coverage and the spark leave them out too.
    score = insights.mark_no_forecast(opt, [_check(insights.SCORE, 0.0, 100.0, 90.0, 90.0),
                                            _check(insights.SCORE, 80.0, 95.0, 90.0, 90.0)])
    assert learned.coverage(score) == (1, 1)
    assert U.trust_counts(score)[:2] == (1, 1)
    spark = U.predictions_spark_frame(opt, checks)
    assert list(spark["text"]) == ["5 / 6", "5 / 6", "3 / 3"]   # Round 5: Tangy only


# ------------------------------------------------------------------ #
#  Fix round 2
# ------------------------------------------------------------------ #

def test_error_record_counts_change_but_its_numbers_do_not():
    """The no-forecast rule changes the in/outside COUNT only: the mean
    miss, its sd, se and df stay those of every fully measured check."""
    opt = _opt()
    plain = _round_of(3, False, 1, start=1) + _round_of(4, True, 0, start=4) + \
        _round_of(5, False, 0, start=7)
    marked = insights.mark_no_forecast(opt, plain)
    a = insights.error_record(plain, "Creaminess")
    b = insights.error_record(marked, "Creaminess")
    assert (a.inside, a.total) == (8, 9) and (b.inside, b.total) == (5, 6)
    assert (a.mean_miss, a.gap_sd, a.se, a.df, a.rounds) == \
        (b.mean_miss, b.gap_sd, b.se, b.df, b.rounds)


def test_the_all_10_samples_bias_numbers_are_as_at_a4e1420(tmp_path):
    """The all-10 sample's records, pinned against the base a4e1420
    (before the sceptic review): only inside and total change."""
    import json
    import sample_projects as sp
    from food_bo import FoodOptimizer
    (tmp_path / f"{sp.ALL10}.pkl").write_text(json.dumps(sp.shipped_state(sp.ALL10)))
    opt = FoodOptimizer(sp.ALL10)
    checks = insights.check_rounds(opt)
    base = {insights.SCORE: (-0.3115378309030792, 3.7049177039542873, 1.0026922889392946, 7),
            "Consistency": (1.0620867017898215, 1.8629027526582356, 0.617445568135852, 7),
            "Creaminess": (0.6188393764434577, 1.2321319623639029, 0.3801769442750603, 7),
            "Tangy": (-0.6595265349785322, 1.482564461980301, 0.3188852943382394, 7),
            "Thickness uniformity": (0.003138588301839033, 1.2601456151466814,
                                     0.4115214791545541, 7)}
    for name, (mean, sd, se, df) in base.items():
        r = insights.error_record(checks, name)
        assert r.mean_miss == pytest.approx(mean, abs=1e-9), name
        assert r.gap_sd == pytest.approx(sd, abs=1e-9) and r.se == pytest.approx(se, abs=1e-9)
        assert r.df == df
    h = insights.headline(opt, checks)
    assert insights.score_bias(opt, checks, h) is None
    assert not any(d.bias for d in insights.details(opt, checks))


@pytest.mark.parametrize("floor, text", [(0.72005, "0.72"), (0.7299, "0.72"), (0.36001, "0.36"),
                                         (0.0021, None), (None, None)])
def test_the_stop_floor_rounds_down_and_says_any_noise_at_the_top(floor, text):
    assert U.stop_floor_text(floor) == text
    if floor is not None and text is None:
        assert wording.stop_holds_above(None) == "The stop shows at any noise."


def test_left_out_values_are_named_where_the_counts_show():
    assert wording.no_forecast_left_out(12) == "12 values with no forecast left out."
    assert wording.no_forecast_left_out(1) == "1 value with no forecast left out."
    assert wording.no_forecast_left_out(0) == ""
    # Fix round 3: the gate counts values checked (12 ≥ 10); with none of
    # them forecast, not judged; under 10 checked, the need is said.
    assert wording.trust_early(0, 12) == "Not judged yet: 0 of 12 values checked had a forecast."
    assert wording.trust_early(2, 4) == (
        "Not judged yet: 2 of 6 values checked had a forecast; the verdict needs 10 checked.")
    assert "0 values checked" not in wording.trust_early(0, 12)
    assert wording.tip_coverage_early(["Tangy"], [5], 0, 3).endswith(
        "0 of 3 values checked had a forecast; the coverage check needs 10.")
    opt = _opt()
    checks = insights.mark_no_forecast(opt, _round_of(3, False, 1, start=1)
                                       + _round_of(4, False, 1, start=4)
                                       + _round_of(5, True, 0, start=7)
                                       + _round_of(5, False, 0, "Tangy", start=7))
    h = insights.headline(opt, checks)
    assert h.no_forecast == 3
    vm = insights.value_misses(opt, checks, h)
    assert vm.no_forecast == 3
    assert U.misses_detail(NS(misses=vm)).endswith("3 values with no forecast left out.")
    # Fix round 3: 12 values checked pass the 10-value gate; the verdict is
    # over the 9 with a forecast, and the 3 left out are said.
    assert h.state == "verdict" and (h.inside, h.total) == (7, 9)
    assert U._coverage_tip(opt, h).endswith("in 9 at the 1-in-10 level. 3 values with no "
                                            "forecast left out.")
    assert "No forecast:" in wording.HOW_FORECAST_WORDS
    assert "Before the first forecast:" in wording.HOW_FORECAST_WORDS
    assert wording.STORY_NO_FORECAST != wording.NO_FORECAST
    for t in (wording.HOW_FORECAST_WORDS, wording.trust_early(0, 12)):
        _clean(t)


def test_the_on_target_question_mark_skips_a_measurement_with_no_noise(monkeypatch):
    opt = _opt()
    res = opt.results_history[opt.best_index()]
    real = learned.pair_noise
    monkeypatch.setattr(learned, "pair_noise",
                        lambda o, checks=(): {k: (None if k == "Tangy" else v)
                                              for k, v in real(o).items()})
    tip = U.on_target_help_text(opt, res)
    assert "Tangy" not in tip and "Within noise" not in tip
    assert U.stop_reason_text(opt) == "On target 3 of 4; Tangy 1.0 under."


def test_charts_show_a_whole_range_check_as_no_forecast(monkeypatch):
    opt = _opt()
    cks = [_check("Creaminess", 0.0, 15.0, 5.0, no=16), _check("Creaminess", 4.0, 6.0, 5.0, no=17)]
    frame = U.check_frame(opt, "Creaminess", cks)
    assert sorted(frame["side"]) == ["in", "none"]
    assert set(frame[frame["side"] == "none"]["miss"]) == {""}


# ------------------------------------------------------------------ #
#  Fix round 3
# ------------------------------------------------------------------ #

def test_the_evidence_and_the_count_use_the_one_pairs_form():
    ev = wording.learned_evidence([("Creaminess", 2, 6, 12), ("Thickness uniformity", 5, 10, 24)])
    assert ev == ("Evidence: pairs ordered right from earlier Rounds — Creaminess 6 of 12 called, 2 right; "
                  "Thickness uniformity 10 of 24 called, 5 right.")
    # An old row with no total: never a bare a/b.
    assert wording.learned_evidence([("Creaminess", 2, 6)]) == (
        "Evidence: pairs ordered right from earlier Rounds — Creaminess 2 right of 6 called.")
    assert not re.search(r"\d+/\d+", ev)
    assert wording.learned_count("Overall score", 7, 9, total=9) == \
        "Overall score 9 of 9 pairs called, 7 right"
    assert wording.learned_count_needed("Overall score", 4, 4, 5, total=4) == \
        "Overall score 4 of 4 pairs called, 4 right (needs 5)"


def test_the_overall_scores_no_forecast_checks_are_named():
    opt = _opt()
    score = [_check(insights.SCORE, 0.0, 100.0, 90.0, 90.0, r=r, no=10 + r) for r in (3, 4)] + \
        [_check(insights.SCORE, 85.0, 95.0, 90.0 + k, 90.0, r=5, no=20 + k) for k in range(4)]
    marked = insights.mark_no_forecast(opt, score)
    rec = insights.error_record(marked, insights.SCORE)
    assert (rec.inside, rec.total, rec.no_forecast) == (4, 4, 2)
    h = NS(score=rec)
    line = U.score_line(opt, marked, h=h)
    assert "4 of 4 inside" in line and "(n = 6)" not in line
    assert line.endswith("2 Overall score values with no forecast left out.")
    assert wording.no_forecast_left_out(6, "Overall score") == (
        "6 Overall score values with no forecast left out.")


def test_the_forecast_check_names_some_and_all_no_forecast():
    # Some (not all) with no forecast range: named beside the count.
    assert wording.check_line(6, 2, 0, 0, "order right: 1 pair", "Creaminess", none=1) == (
        "**Round 6: Creaminess, both inside their 95 % range.** 1 with no forecast; "
        "order right: 1 pair.")
    # Every one: the subject named, the Overall score too.
    assert wording.check_line(6, 0, 0, 0, "x", None, none=3) == (
        "**Round 6: Overall score, 3 values with no forecast range.**")


@pytest.mark.parametrize("targets, name, limit", [
    (True, "Tangy", 10), (True, "Thickness uniformity", 11), (True, "Water holding capacity", 12),
    (False, "Juiciness", 11), (False, "Thickness uniformity", 12)])
def test_one_miss_is_always_named_under_either_head(targets, name, limit):
    """One off target is named with its gap and verdict (controller ruling:
    specific wording outranks a word or two): up to 11 words after "On
    target" with a two-word name, 12 with a three-word one; 12 after "At
    their best" with a two-word name."""
    kind = "under" if targets else "short"
    line = wording.stop_on_target(3, 4, [f"{name} 0.5 {kind}"], targets, [False])
    assert line.endswith(f"{name} 0.5 {kind}, within noise.")
    assert _words(line) <= limit, line


def test_the_count_forms_keep_to_their_limits_under_either_head():
    offs = ["Juiciness 1.0 short", "Firmness 2.0 short"]
    best = wording.stop_on_target(2, 4, offs, False, [True, False])
    assert best == "At their best 2 of 4; 1 beyond, 1 within noise."
    # "At their best" is the head's own words: 11 for its mixed count form.
    assert _words(best) <= 11
    on = wording.stop_on_target(2, 4, offs, True, [True, False])
    assert _words(on) <= 10


def test_widening_is_said_from_one_point_o_five():
    opt = _opt()
    base = [_check("Creaminess", 0.0, 15.0, 5.0, 5.7)]
    for f, said in ((1.0, None), (1.04, None), (1.05, "1.1"), (3.0, "3.0")):
        cks = [dataclasses.replace(c, widen=f) for c in base]
        assert U.no_forecast_items(opt, cks, 6) == [("Creaminess", "0–15", said)], f
    assert "×1.0 " not in wording.no_forecast_help([("Creaminess", "0–15", None)])


def test_definitions_are_unambiguous():
    assert "Overall score's whole Range" in wording.NO_FORECAST_DEFINED
    assert "until 5 formulations had results" in wording.HOW_FORECAST_WORDS
    assert wording.LEARNED_CHART_X == "Of pairs called, % right"
    assert not hasattr(U, "score_landing")


def test_the_gates_count_values_checked_not_values_forecast():
    """Fix round 3: the 10-value gate (headline) and the Overall score's
    SKILL_MIN gate count the checks made, as before the no-forecast rule;
    with no forecast at all, nothing is judged."""
    opt = _opt()
    checks = insights.mark_no_forecast(opt, _round_of(3, False, 0, start=1)
                                       + _round_of(4, True, 0, start=4)
                                       + _round_of(5, False, 0, start=7)
                                       + _round_of(5, False, 0, "Tangy", start=7))
    h = insights.headline(opt, checks)
    assert h.state == "verdict" and (h.total, h.no_forecast) == (9, 3)
    every = insights.mark_no_forecast(opt, _round_of(3, True, 0, start=1)
                                      + _round_of(4, True, 0, start=4)
                                      + _round_of(5, True, 0, start=7)
                                      + _round_of(6, True, 0, start=10))
    assert insights.headline(opt, every).state == "early"
    rec = insights.ScoreRecord(4, 4, 0.0, None, [1, 2], no_forecast=6)
    assert insights.score_verdict_state(opt, NS(state="verdict", score=rec, wide=[]), []) \
        != "early"
    rec0 = insights.ScoreRecord(0, 0, 0.0, None, [1, 2], no_forecast=12)
    assert insights.score_verdict_state(opt, NS(state="verdict", score=rec0, wide=[]), []) \
        == "early"
