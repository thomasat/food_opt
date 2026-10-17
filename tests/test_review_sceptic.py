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


# ------------------------------------------------------------------ #
#  2 · the tasting noise: the stop's "?" says the noise in use
# ------------------------------------------------------------------ #


def test_the_stop_moves_with_the_setting():
    # The stop rule (at_ceiling) at the project's Tasting noise: Best so far
    # 1.67 below the top shows the stop from about 0.72 points of noise.
    opt = _opt()
    gap = round(100 - float(opt.Y_history[opt.best_index()]), 2)
    assert gap == 1.67
    for s, want in ((0.73, True), (0.71, False), (3.0, True), (0.5, False)):
        insights.set_tasting_noise(opt, s)
        facts = NS(yardstick=insights.noise_yardstick(opt),
                   best=NS(score=float(opt.Y_history[opt.best_index()]), formulation=16))
        assert U.at_ceiling(opt, facts) is want, s


# ------------------------------------------------------------------ #
#  3 · the Forecast check: the count and the reference
# ------------------------------------------------------------------ #

def test_the_forecast_check_says_how_many_inside_what():
    """Restored with the Round line (one-glance Review, fix round 1); fix
    round 2: the landing fact alone, at most 10 words; the order is in the
    line's "?" (test_the_check_line_counts_ties)."""
    assert wording.check_line(10, 3, 0, 0) == "**Round 10: all 3 inside their forecast range.**"
    assert wording.check_line(6, 1, 0, 0) == "**Round 6: inside its forecast range.**"
    assert wording.check_line(6, 2, 0, 0, "Consistency") == (
        "**Round 6: Consistency, both inside their forecast range.**")
    # No range left: the results named.
    assert wording.check_line(6, 0, 0, 0, "Creaminess", none=3) == (
        "**Round 6: Creaminess, 3 values with no forecast range.**")
    # A miss: how many, which side, of how many.
    assert wording.check_line(6, 1, 2, 0) == "**Round 6: 2 of 3 above their forecast range.**"
    assert wording.check_line(6, 2, 0, 1) == "**Round 6: 1 of 3 below its forecast range.**"
    assert wording.check_line(6, 0, 3, 0) == "**Round 6: all 3 above their forecast range.**"
    # Both ways: each side named, "below" too.
    assert wording.check_line(6, 1, 1, 1) == (
        "**Round 6: 1 inside, 1 above, 1 below forecast range.**")
    assert wording.check_line(6, 0, 2, 1) == "**Round 6: 2 above, 1 below their forecast range.**"
    for args in ((10, 3, 0, 0), (6, 1, 2, 0), (6, 1, 1, 1), (6, 2, 0, 0)):
        assert _words(wording.check_line(*args)) <= 10
    assert "as forecast" not in wording.check_line(10, 3, 0, 0)
    assert "order" not in wording.check_line(6, 1, 2, 0)
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
    assert wording.called_of(7, 9, 24) == "9 of 24 called, 7 right"
    # Final review: the Definitions' rule, the result's own noise and the
    # at-cap rule (learned.too_close).
    assert "Called: measured values unequal, forecasts at least the result's noise apart " \
           "(Overall score in points, a measurement in its own units) and not both at the " \
           "top; counted of all pairs." in wording.learned_verdicts_help(5)
    # Beside the longest pairs line of the samples (27 words), the "?" still
    # fits its ~120 words, so brief() keeps the definitions.
    pairs = ("Overall score: 9 of 24 pairs called (13 too close, 2 measured equal); 7 right. "
             "Chance by "
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
    (before the sceptic review): only inside and total change. Re-pinned
    when the sample took the pilot's own limits (the model reads amounts
    as shares of each Lowest-Highest): Tangy's predictions now run high
    over Rounds 3-10, beyond its noise; the Overall score's do not."""
    import json
    import sample_projects as sp
    from food_bo import FoodOptimizer
    (tmp_path / f"{sp.ALL10}.pkl").write_text(json.dumps(sp.shipped_state(sp.ALL10)))
    opt = FoodOptimizer(sp.ALL10)
    checks = insights.check_rounds(opt)
    base = {insights.SCORE: (-0.6989407206590066, 3.2753873686595236, 0.7908622282691944, 7),
            "Consistency": (0.9367823847804867, 1.7650620976165172, 0.5993300420438497, 7),
            "Creaminess": (0.6095510888328993, 1.2235456484352254, 0.36900302965143467, 7),
            "Tangy": (-0.7730958575129686, 1.3654880880458824, 0.24230436210362377, 7),
            "Thickness uniformity": (-0.04256151085781501, 1.2345902897386172,
                                     0.4018420682708456, 7)}
    for name, (mean, sd, se, df) in base.items():
        r = insights.error_record(checks, name)
        assert r.mean_miss == pytest.approx(mean, abs=1e-9), name
        assert r.gap_sd == pytest.approx(sd, abs=1e-9) and r.se == pytest.approx(se, abs=1e-9)
        assert r.df == df
    h = insights.headline(opt, checks)
    assert insights.score_bias(opt, checks, h) is None
    assert [d.name for d in insights.details(opt, checks) if d.bias] == ["Tangy"]


# ------------------------------------------------------------------ #
#  Fix round 3
# ------------------------------------------------------------------ #

def test_the_evidence_and_the_count_use_the_one_pairs_form():
    ev = wording.learned_evidence([("Creaminess", 2, 6, 12), ("Thickness uniformity", 5, 10, 24)])
    assert ev == ("Evidence: pairs ordered right from earlier Rounds — Creaminess 6 of 12 called, 2 right; "
                  "Thickness uniformity 10 of 24 called, 5 right.")
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
    # Fix round 4: scoped to the inside count (the mean error is over all 6).
    assert line.endswith("2 Overall score values with no forecast left out of the inside "
                         "count.")
    assert wording.no_forecast_left_out(6, "Overall score") == (
        "6 Overall score values with no forecast left out.")


def test_the_forecast_check_names_some_and_all_no_forecast():
    # Some (not all) with no forecast range: named beside the count.
    assert wording.check_line(6, 2, 0, 0, "Creaminess", none=1) == (
        "**Round 6: Creaminess, both inside their forecast range; 1 no forecast.**")
    assert wording.check_line(6, 2, 0, 0, "Creaminess", none=2).endswith(
        "; 2 no forecast.**")
    # Every one: the subject named, the Overall score too.
    assert wording.check_line(6, 0, 0, 0, None, none=3) == (
        "**Round 6: Overall score, 3 values with no forecast range.**")


@pytest.mark.parametrize("targets, name, limit", [
    (True, "Tangy", 10), (True, "Thickness uniformity", 11), (True, "Water holding capacity", 12),
    (False, "Juiciness", 11), (False, "Thickness uniformity", 12),
    (False, "Water holding capacity", 13)])
def test_one_miss_is_always_named_under_either_head(targets, name, limit):
    """One off target is named with its gap and verdict (controller ruling:
    specific wording outranks a word or two): up to 11 words after "On
    target" with a two-word name, 12 with a three-word one; after "At
    their best" one more, 12 and 13 (fix round 4: the 13 is this named
    single-miss form's alone; every other status stays at 10)."""
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


# ------------------------------------------------------------------ #
#  Fix round 4
# ------------------------------------------------------------------ #

def _early_window_checks(opt):
    """Rounds 1–2 forecast and inside; Rounds 3–6 every range over the
    whole Range: the window (Rounds 3–6, 12 values checked) has no value
    with a forecast, though earlier Rounds had."""
    return insights.mark_no_forecast(opt, _round_of(1, False, 0, start=1)
                                     + _round_of(2, False, 0, start=4)
                                     + _round_of(3, True, 0, start=7)
                                     + _round_of(4, True, 0, start=10)
                                     + _round_of(5, True, 0, start=13)
                                     + _round_of(6, True, 0, start=16))


def test_the_overall_score_counts_say_of_and_name_their_left_out():
    """Finding 8: no bare a/b in the evidence; the misses row names the
    Overall score's checks with no forecast."""
    assert wording.ev_predictions(10, 12, "0.69", 12, 27, 36) == (
        "Overall score 10 of 12 in range, skill 0.69 (n = 12); measurements 27 of 36 in range")
    assert not re.search(r"\d+/\d+", wording.ev_predictions(10, 12, "0.69", 12, 27, 36))
    assert wording.misses_scoped(["Tangy"], 4, 4, 2).endswith(
        "The Overall score landed in range 4 of 4 times. 2 Overall score values with no "
        "forecast left out of the inside count.")
    assert wording.misses_scoped([], 4, 4).endswith("landed in range 4 of 4 times.")
    assert wording.score_landed(0, 0, 3) == ""
    for t in (wording.misses_scoped(["Tangy"], 4, 4, 2), wording.ev_predictions(1, 2)):
        _clean(t)


