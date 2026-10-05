"""Review round 2 (2026-10-05): the fix plan from four critics (visual,
statistics, demo walkthrough, wording), P1 → P4.

- One widening everywhere: each measurement's Round-ahead checks widened by
  its own record from earlier Rounds (insights.nest_measure_widen), so Make a
  round's Predicted measurements, the Forecast check and Results' In / Above
  / Below read one 95 % range per forecast.
- Holm across the results checked before Predicts; the chance bar drawn at
  each result's Holm level, so dot and verdict agree.
- Two forecasts both at the top of the Range count as too close to call.
- Implied (expected): E[score] over each measurement's forecast.
- The stop: a concrete line under the headline, its "?" in numbers, the
  noisy-perfect score; "3 Repeats would measure the noise".
- Are we improving?: the zone within noise of the top; Start vs best's labels;
  the suggestions folded under Stop and cached; the Forecast check its own
  section; the profiler's cramped columns named, not drawn; secondary
  buttons under Stop; one form per idea.

Synthetic projects and hand-made checks only; no pilot data."""
import ast
import contextlib
import json
import math
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import ui_batch
import ui_insights as U
import ui_results
import wording
from food_bo import FoodOptimizer
from insights import Check
from tests.test_food_bo import _BANNED

NBSP = " "


def _check(no, r, name, pred, meas, sd=0.5, raw=None, low=None, high=None, full=True):
    raw = pred if raw is None else raw
    low = raw - insights.Z95 * sd if low is None else low
    high = raw + insights.Z95 * sd if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=raw, sd=sd)


def _stub(names=("Firmness",), top=15.0):
    return NS(objectives=[{"name": n, "goal": "target", "target": 7, "weight": 100 / len(names),
                           "min_val": 0.0, "max_val": top} for n in names],
              utility_ceiling=lambda: 100.0)


def _project(tmp_path, monkeypatch, name="final", k=4):
    """k target measurements on 0–15, equal shares, targets 12/5/5/13."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Oat", 0, 100)
    opt.add_ingredient("Water", 0, 100)
    targets = [("Consistency", 12), ("Creaminess", 5), ("Tangy", 5), ("Thickness", 13)][:k]
    for n, t in targets:
        opt.add_objective(n, 100.0 / k, goal="target", target=t, min_val=0, max_val=15)
    return opt


# ------------------------------------------------------------------ #
#  P1 #1 — one widening everywhere
# ------------------------------------------------------------------ #

def _measure_checks():
    """Rounds 3–6 of one measurement; Rounds 3–5 miss by 2 sd each, so
    the record widens Round 6's range by 2."""
    out = []
    no = 1
    for r in (3, 4, 5):
        for k in range(4):
            out.append(_check(no, r, "Firmness", 7.0, 7.0 + (1.0 if k % 2 else -1.0)))
            no += 1
    out.append(_check(no, 6, "Firmness", 7.0, 8.5))     # 3 sd above: outside 1.96 sd, inside 3.92
    return out


def test_one_widening_is_the_same_range_on_make_forecast_check_and_results(monkeypatch):
    opt = _stub()
    checks = insights.nest_measure_widen(opt, _measure_checks())
    r6 = [c for c in checks if c.round == 6][0]
    # Make a round widens the open Round by every other Round's record:
    make = insights.measure_widen(checks, "Firmness", skip_round=6)
    assert make == pytest.approx(2.0)
    assert r6.widen == pytest.approx(make)
    p = insights.widened(NS(low=0.0, high=15.0),
                         insights.Prediction(7.0, 6.0, 8.0, 7.0, 0.5), make)
    assert (p.low, p.high) == pytest.approx((r6.low, r6.high))
    # Round 6 was a miss unwidened; inside the one range it is not.
    assert r6.inside and r6.low <= 8.5 <= r6.high
    # Results and the Forecast check read that same range.
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)
    monkeypatch.setattr(insights, "_range_of", lambda o, n: (0.0, 15.0))
    fr = U.check_frame(opt, "Firmness", [r6])
    assert (fr["low"][0], fr["high"][0], fr["side"][0]) == pytest.approx(
        (r6.low, r6.high, "in"), rel=1e-9) or (fr["low"][0] == pytest.approx(r6.low)
                                              and fr["side"][0] == "in")
    # The earliest Round has no record: unwidened.
    assert [c for c in checks if c.round == 3][0].widen == 1.0


def test_measure_widen_reads_the_models_own_sd_not_the_nested_widening():
    checks = insights.nest_measure_widen(_stub(), _measure_checks())
    plain = [c for c in _measure_checks() if c.round < 6]
    assert insights.measure_widen(checks, "Firmness", 6) == insights.measure_widen(plain,
                                                                                    "Firmness")


def test_a_too_wide_cell_reads_a_dash_with_its_reason_in_the_tip():
    assert wording.TOO_UNCERTAIN == "—"
    assert wording.too_uncertain_tip("6.0 (0.0–15.0)").startswith(f"95{NBSP}% range over")


# ------------------------------------------------------------------ #
#  P1 #2 — Holm
# ------------------------------------------------------------------ #

def _ordering(name, p_exact, pairs=6, right=6):
    o = learned.Ordering(name, right=right, pairs=pairs)
    # A null whose tail at the observed stat is p_exact: S = stat with
    # probability p_exact, else 0.
    stat = 2 * right - pairs
    o.nulls = [{stat: p_exact, -stat: 1.0 - p_exact}]
    o.rounds = [3]
    return o


def test_holm_adjusts_before_predicts_and_the_bar_agrees():
    ords = {"a": _ordering("a", 0.004), "b": _ordering("b", 0.03), "c": _ordering("c", 0.3)}
    learned.holm(ords)
    assert ords["a"].p_holm == pytest.approx(0.012)
    assert ords["b"].p_holm == pytest.approx(0.06)
    assert ords["a"].verdict == learned.PREDICTABLE
    # p 0.03 alone would be Predictable; adjusted for 3 results it is not.
    assert ords["b"].verdict == learned.PARTLY
    assert ords["c"].verdict == learned.NOT_YET
    assert {o.tested for o in ords.values()} == {3}
    # The dot clears its bar exactly when the verdict is Predicts.
    for o in ords.values():
        lo, hi = o.chance_share()
        assert (o.share > hi + 1e-9) == (o.verdict == learned.PREDICTABLE), o.name


def test_holm_keeps_a_later_result_inside_its_bar_after_one_fails():
    ords = {"a": _ordering("a", 0.03), "b": _ordering("b", 0.04)}
    learned.holm(ords)
    assert ords["a"].verdict != learned.PREDICTABLE and ords["b"].verdict != learned.PREDICTABLE
    # b alone (0.04 < 0.05) would clear 0.05; held at the failed level it does not.
    assert ords["b"].level == ords["a"].level == pytest.approx(0.025)
    lo, hi = ords["b"].chance_share()
    assert not ords["b"].share > hi + 1e-9


def test_one_result_alone_is_not_adjusted():
    ords = learned.holm({"a": _ordering("a", 0.03)})
    assert ords["a"].p_holm == pytest.approx(0.03) and ords["a"].verdict == learned.PREDICTABLE


def test_the_learned_caption_and_rule_say_the_adjustment():
    assert wording.learned_caption(5) == "Past the bar: Predicts · adjusted for 5 results"
    assert "Holm" in wording.LEARNED_RULE
    assert wording.learned_multiplicity(5).startswith("Adjusted for 5 results")


# ------------------------------------------------------------------ #
#  P1 #3 — both at the cap: too close to call
# ------------------------------------------------------------------ #

def test_two_forecasts_at_the_top_are_too_close_to_call():
    a = _check(1, 2, insights.SCORE, 100.0, 97.0, raw=104.0)
    b = _check(2, 2, insights.SCORE, 100.0, 95.0, raw=101.0)
    c = _check(3, 2, insights.SCORE, 92.0, 90.0)
    assert learned.at_cap(a) and learned.at_cap(b) and not learned.at_cap(c)
    assert learned.too_close(a, b, 1.0)
    assert learned.too_close(a, b, 0.0)
    assert not learned.too_close(a, c, 1.0)
    call = learned.round_call([a, b, c], insights.SCORE, 2, 1.0)
    assert call.close == 1 and call.counted == 2


# ------------------------------------------------------------------ #
#  P1 #4 — Implied (expected)
# ------------------------------------------------------------------ #

def test_implied_is_the_expected_score_over_each_forecast(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    on = {o['name']: o['target'] for o in opt.objectives}
    sure = {n: insights.Prediction(v, v, v, v, 0.0) for n, v in on.items()}
    assert insights.expected_score(opt, sure) == pytest.approx(100.0)
    # At the target with sd σ, each measurement loses 25/15 × σ × √(2/π).
    sigma = 0.3
    unsure = {n: insights.Prediction(v, v, v, v, sigma) for n, v in on.items()}
    want = 100 - 4 * 25 / 15 * sigma * math.sqrt(2 / math.pi)
    assert insights.expected_score(opt, unsure) == pytest.approx(want, abs=0.02)
    # Below the score of the means whenever a forecast is unsure.
    assert insights.expected_score(opt, unsure) < opt._compute_utility(on)
    assert insights.expected_score(opt, {"Consistency": sure["Consistency"]}) is None
    assert wording.IMPLIED_COLUMN == "Implied (expected)"
    assert "two models" in wording.PREDICTED_MEASUREMENTS_HELP.lower()


# ------------------------------------------------------------------ #
#  P1 #5 — the stop's concrete line and its "?" numbers
# ------------------------------------------------------------------ #

def test_the_stop_line_names_what_is_on_and_off_target(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    opt.tell({"Oat": 40.0, "Water": 60.0},
             {"Consistency": 12, "Creaminess": 5, "Tangy": 4, "Thickness": 13},
             formulation_no=1, batch_no=1)
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    assert U.ceiling_reason_text(opt) == "On target: 3 of 4. Tangy 1.0 under target."
    assert len(U.ceiling_reason_text(opt).split()) <= 12


def test_the_stop_help_gives_the_gap_the_threshold_and_the_noisy_perfect_score(
        tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    perfect = U.perfect_noisy_score(opt)
    # 100 − Σ (share ÷ width) × σ_a × √(2/π), σ_a = 1 × 15 ÷ (25 × 2) = 0.3.
    assert perfect == pytest.approx(100 - 4 * (25 / 15) * 0.3 * math.sqrt(2 / math.pi),
                                    abs=0.02)
    assert f"{perfect:.1f}" == "98.4"
    text = wording.ceiling_help("98.3", "100", "1.7", "2.3", "98.4", True)
    assert text.startswith("Best so far 98.3 is 1.7 below the top (100); 2.3 is needed to "
                           "tell two single results apart.")
    assert ("A formulation exactly on every target would score about 98.4 here, because "
            "tasting noise only pulls the score down (assumed noise).") in text
    assert ("A single new result can't be told apart from this best in the Overall score; a "
            "single measurement still can") in text
    assert "Further Rounds" not in text
    assert text.endswith("3 Repeats would measure the noise.")


def test_repeat_wording_is_one_form_everywhere():
    assert wording.YARDSTICK_REPEATS == insights.YARDSTICK_DF == 3
    assert wording.REPEAT_CONFIRMS == "3 Repeats would measure the noise."
    assert wording.card_repeat_ask(16, 7) == "Repeat Formulation 16 to check noise"
    assert wording.repeat_link(21) == "Repeat Formulation 21 to check noise"
    assert "A Repeat would confirm" not in wording.check_noise([("Overall score", "1.0")])
    assert "A Repeat would confirm" not in wording.story_improving_help("84.2", "96.1", 1, 6)


def test_progress_help_says_what_the_rise_needed_is():
    assert wording.progress_need_plain("3.0", "1", True, 3).startswith(
        "3.0: the best of 3 new results must beat noise by this much.")
    assert wording.NOISE_BAND == "±1 point: assumed noise (1 SD)"


# ------------------------------------------------------------------ #
#  P2 #7 — the top zone on Are we improving?
# ------------------------------------------------------------------ #

def _near_top(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch, k=1)
    values = [[9, 10, 8], [11, 12, 11.5]]
    no = 1
    for r, row in enumerate(values, 1):
        for v in row:
            opt.tell({"Oat": 10.0 * no, "Water": 50.0}, {"Consistency": v},
                     formulation_no=no, batch_no=r)
            no += 1
    return opt


def test_the_improving_chart_shades_the_zone_within_noise_of_the_top(tmp_path, monkeypatch):
    opt = _near_top(tmp_path, monkeypatch)
    spec = U.improving_chart(opt).to_dict()
    text = json.dumps(spec, ensure_ascii=False)
    assert wording.TOP_ZONE in text and wording.NOISE_BAND in text
    zone = next(v for v in spec["datasets"].values() if v and v[0].get("what") == wording.TOP_ZONE)
    need = U.tie_need(insights.noise_yardstick(opt))
    assert zone[0]["hi"] == pytest.approx(100.0) and zone[0]["lo"] == pytest.approx(100 - need)
    # The band is clipped at the top score.
    band = next(v for v in spec["datasets"].values()
                if v and v[0].get("what") == wording.NOISE_BAND)
    assert max(r["hi"] for r in band) <= 100.0 + 1e-9
    # The dotted line is named on its tick; Round averages are printed.
    assert "Top score 100" in text
    for _r, _a, _b, m, _n in U.round_averages(opt):
        assert f'"text": "{m:.1f}"' in text
    assert wording.STORY_IMPROVING_CAPTION == "Line: best so far. Bars: Round average."


# ------------------------------------------------------------------ #
#  P2 #8 — the suggestions folded under Stop; the preview cached
# ------------------------------------------------------------------ #

class _FakeSt:
    """Records what a builder draws: markdown, captions, buttons, folds."""

    def __init__(self, fold_open=False):
        self.calls, self.fold_open = [], fold_open
        self.session_state = {}

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "button":
                return False
            if name in ("container", "empty", "spinner"):
                return contextlib.nullcontext(self)
            return None
        return call

    @contextlib.contextmanager
    def expander(self, label, **kw):
        self.calls.append(("expander", (label,), kw))
        yield NS(open=self.fold_open)


def test_the_next_suggestions_fold_under_stop_and_work_out_only_when_open(monkeypatch):
    fake = _FakeSt(fold_open=False)
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "readiness", lambda o: (True, None))
    monkeypatch.setattr(U, "ceiling_stop", lambda o, f: True)
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    worked = []
    monkeypatch.setattr(U, "_preview_body", lambda *a, **k: worked.append(1))
    opt = NS(pending_batch=None, complete_count=lambda: 10)
    places = NS(preview=fake)
    U._next_preview(opt, places, {"failed": False}, NS(next=None))
    assert ("expander", (wording.LEARNED_PREVIEW_TITLE,)) in [(c[0], c[1]) for c in fake.calls]
    assert not worked
    fake.fold_open = True
    U._next_preview(opt, places, {"failed": False}, NS(next=None))
    assert worked


def test_the_preview_is_cached_per_project_state(monkeypatch):
    U._PREVIEW_CACHE.clear()
    made = []
    monkeypatch.setattr(learned, "preview_next", lambda o, n: made.append(1) or [])
    monkeypatch.setattr(insights, "setup_key", lambda o: "state-1")
    monkeypatch.setattr(U, "_preview_count", lambda o: 3)
    opt = NS(project_name="p", next_formulation_no=19)
    assert not U.preview_cached(opt)
    # An empty preview is a failure: never kept.
    U.preview_rows(opt, {})
    assert not U.preview_cached(opt)
    monkeypatch.setattr(learned, "preview_next", lambda o, n: made.append(1) or None)
    U._PREVIEW_CACHE[U.preview_key(opt)] = ["row"]
    assert U.preview_cached(opt) and U.preview_rows(opt, {}) == ["row"]
    n = len(made)
    U.preview_rows(opt, {})
    assert len(made) == n          # a revisit works nothing out
    other = NS(project_name="q", next_formulation_no=19)
    assert not U.preview_cached(other)
    U._PREVIEW_CACHE.clear()


# ------------------------------------------------------------------ #
#  P2 #9 — the Forecast check is its own section
# ------------------------------------------------------------------ #

def _round6():
    out = []
    for no, sp, sm in zip((16, 17, 18), (94.3, 94.7, 93.7), (98.3, 98.3, 91.7)):
        out.append(_check(no, 6, insights.SCORE, sp, sm, sd=1.2))
    return out


def _lf(opt, checks):
    orderings = learned.orderings(opt, checks, {insights.SCORE: 1.0})
    call = learned.round_call(checks, insights.SCORE, 6, 1.0)
    items = [(insights.SCORE, call, learned.round_checks(checks, insights.SCORE, 6))]
    lf = U.LearnedFacts(orderings, {}, (6, []), call, (1, 2, 0), "", items)
    lf.noise, lf.checks = {insights.SCORE: 1.0}, checks
    return lf


class _Obj:
    def __init__(self):
        self.objectives = [{"name": "Consistency", "goal": "target", "target": 12,
                            "weight": 100, "min_val": 0, "max_val": 15}]
        self.formulation_ids = list(range(1, 40))

    def measurements_by_importance(self):
        return list(self.objectives)


def test_the_forecast_check_is_a_section_with_a_wide_chart(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0 if n == insights.SCORE
                                                                else 15.0))
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)
    opt = _Obj()
    lf = _lf(opt, _round6())
    spec = U.called_chart(opt, lf).to_dict()
    assert "vconcat" in spec
    panel = spec["vconcat"][0]
    assert panel["width"] >= 600 and panel["height"] == {"step": 40}
    frame = U.check_frame(opt, insights.SCORE, lf.items[0][2])
    assert set(frame["miss"]) == {"98.3 above", ""}
    text = json.dumps(spec, ensure_ascii=False)
    assert '"symbolOpacity": 1' in text
    # Every Overall score landed in or above: no "below", and the legend
    # names only what is drawn.
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "show_chart", lambda *a, **k: None)
    U._called_top(opt, fake, lf)
    marks = [c[1][0] for c in fake.calls if c[0] == "markdown"]
    assert f"#### {wording.FORECAST_CHECK_TITLE}" in marks
    assert U.SECTION_ORDER.index("next") + 1 == U.SECTION_ORDER.index("predictions")


def test_a_verdict_absent_from_the_check_is_said():
    assert wording.called_shown(10) == "Shown: results predictable before Round 10."
    assert "Tangy" in wording.called_shown_help(["Tangy"], 10)


# ------------------------------------------------------------------ #
#  P2 #10 — Start vs best labels
# ------------------------------------------------------------------ #

def _rows():
    def r(name, start, best, target):
        return {"name": name, "goal": "target", "target": target, "low": 0.0, "high": 15.0,
                "start": start, "best": best, "unit": "/15"}
    return [r("Consistency", 8, 12, 12), r("Creaminess", 4, 5, 5), r("Tangy", 5, 4, 5),
            r("Thickness", 13, 13, 13)]


def test_start_vs_best_reads_one_axis_and_says_where_the_dot_moved(monkeypatch):
    monkeypatch.setattr(U, "target_rows", lambda opt: _rows())
    monkeypatch.setattr(U, "story_start", lambda opt: (0, 1))
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    spec = U.target_chart(NS(best_index=lambda: 1)).to_dict()
    text = json.dumps(spec, ensure_ascii=False)
    assert wording.story_target_start(1) == "Start (Round 1's best)" and "Start (Round 1's best)" in text
    data = [p for v in spec["datasets"].values() for p in v]
    # One shared axis under the chart: 0, 5, 10, 15 once each.
    ticks = [p["text"] for p in data if "t" in p and p.get("k") == 3 and "what" not in p
             and "tip" not in p and "who" not in p]
    assert ticks == ["0", "5", "10", "15"]
    assert not [p for p in data if p.get("align") in ("left", "right")]
    # The start's legend swatch is hollow, the best's filled.
    fills = [l["encoding"]["fill"]["scale"] for l in spec["layer"]
             if "fill" in l.get("encoding", {})]
    assert fills and dict(zip(fills[0]["domain"], fills[0]["range"]))[
        "Start (Round 1's best)"] == "transparent"
    # Tangy moved down (5 → 4): "moved away" sits left of the best's dot.
    away = [l for l in spec["layer"] if l.get("mark", {}).get("type") == "text"
            and l["mark"].get("fontStyle") == "italic"]
    assert away and all(l["mark"]["align"] == "right" for l in away)


# ------------------------------------------------------------------ #
#  P2 #13 — the profiler's cramped columns
# ------------------------------------------------------------------ #

def test_profiler_columns_with_too_little_room_are_named_not_drawn(monkeypatch):
    def col(name, lo, hi):
        return NS(name=name, path=NS(low_end=NS(amount=lo), high_end=NS(amount=hi)))
    profile = NS(columns=[col("Tap water", 41.9, 42.0), col("Salt", 0.2, 1.5)])
    monkeypatch.setattr(insights, "_var", lambda o, n: {"bounds": (25.0, 42.0) if n == "Tap water"
                                                        else (0.0, 2.5)})
    assert U.cramped_columns(None, profile) == ["Tap water"]
    assert wording.profiler_cramped(["Tap water", "Coconut cream"]) == (
        "Too little room to move: Tap water, Coconut cream.")
    line = wording.answer_line(wording.drives_limited("Batch total"), "")
    assert line == "**Single-ingredient moves are limited by the Batch total.**"


# ------------------------------------------------------------------ #
#  P2 #16 — the primaries go secondary under Stop
# ------------------------------------------------------------------ #

def test_the_strips_set_up_button_is_secondary_at_the_stop(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "confirmation_open", lambda: False)
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    U._ceiling_foot(NS())
    [btn] = [c for c in fake.calls if c[0] == "button"]
    assert btn[2]["type"] == "secondary"


def test_start_the_next_round_is_secondary_with_its_reason_under_stop(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(ui_results, "st", fake)
    monkeypatch.setattr(ui_results, "readiness", lambda o: (True, None))
    monkeypatch.setattr(ui_results, "confirmation_open", lambda: False)
    monkeypatch.setattr(U, "review_says_stop", lambda o: True)
    ui_results._foot(NS(pending_batch=None))
    [btn] = [c for c in fake.calls if c[0] == "button"]
    assert btn[2]["type"] == "secondary"
    assert ("caption", (wording.REVIEW_SAYS_STOP,)) in [(c[0], c[1]) for c in fake.calls]
    assert wording.REVIEW_SAYS_STOP == "Review says stop."
    src = pathlib.Path(ui_batch.__file__).read_text()
    assert "review_says_stop(opt)" in src and "wording.REVIEW_SAYS_STOP" in src


# ------------------------------------------------------------------ #
#  P3 #17–19 — one form per idea
# ------------------------------------------------------------------ #

def test_one_form_per_idea():
    assert wording.learned_strip(["Consistency"], ["Tangy"]) == ("Predicts:", "Consistency")
    assert wording.learned_strip([], ["Overall score", "Tangy"]) == ("Partly:",
                                                                     "Overall score, Tangy")
    assert wording.learned_strip([], [], ["Overall score"]) == ("Not yet:", "Overall score")
    assert wording.learned_answer(["Consistency"], ["Tangy"], ["Creaminess"]) == (
        "**Predicts: Consistency.** Partly: Tangy. Not yet: Creaminess.")
    assert list(wording.LEARNED_WORDS.values()) == ["Predicts", "Partly", "Not yet"]
    assert wording.learned_not_predictable(["Creaminess"]) == "Not yet: Creaminess."
    assert wording.CARD_CEILING == "Stop, or change the target"
    assert wording.CEILING_TAIL == "Stop, or change the target."
    assert wording.review_status_strip("Rising", "Predicts: Overall score", wording.CARD_CEILING) == (
        "Review: **Rising** · Predicts: Overall score · Stop, or change the target")
    assert wording.tie_line(21, "99.2", "0.83", 19, "2.3") == (
        "Formulation 19 is within noise: 0.83 lower.")
    assert "tied" not in wording._tied_with(18)
    assert wording.STORY_LANDED_OUT == wording.CHECK_KEY_OUT == "Measured, outside range"
    assert wording.single_record(36, 36, 3) == (
        f"Single measurements, last 3 Rounds: 36 of 36 in their 95{NBSP}% range.")
    assert "0.3: 1 score point in Consistency's units, split over 4 measurements." in (
        wording.check_noise([("Overall score", "1.0"), ("Consistency", "0.3")], True, 4))
    assert wording.DOWNLOAD_BATCH_SHEETS == "Download the Round workbook (Excel)"
    assert wording.back_to_batch_label(6, 3) == "Back to Round 6"
    assert wording.check_order(1, 1, 1, ties=1) == "order right: 1 pair, 1 too close, 1 tie"
    assert wording.called_so_far([("Overall", 8, 10)]) == "So far: Overall score 8 of 10."
    assert wording.ENTERED_ROUND_CAPTION.endswith(
        "For suggestions too: Discard this Round and generate again.")
    assert wording.plateau_rule(7) == "Plateau: no rise since Round 7, after rising before it."
    assert wording.rule_line("Many more misses than expected.") == (
        "Rule: many more misses than expected.")
    assert wording.method_rising("3.0").endswith("by at least the rise needed (3.0 here), "
                                                 "to the hundredth.")
    assert wording.NO_GAIN_HEAD == "No gain beyond noise within these limits."
    assert insights.TOGETHER == insights.MOVED_R == 0.7
    assert "at least 0.8" not in json.dumps(wording.glossary_rows())
    assert not [r for r in wording.glossary_rows() if r[0] == "Where you have tested"]
    assert "95 %" not in wording.PREDICTED_MEASUREMENTS        # a non-breaking space


def test_captions_and_labels_keep_to_their_word_budgets():
    def words(t):
        return len(re.sub(r"\*\*", "", t).split())
    for text, limit in ((wording.ceiling_reason(3, 4, ["Tangy 1.0 under target"]), 12),
                        (wording.REVIEW_SAYS_STOP, 10), (wording.MAP_DIAMONDS_CAPTION, 8),
                        (wording.PREDICTED_MEASUREMENTS_CAPTION, 8),
                        (wording.STORY_IMPROVING_CAPTION, 8),
                        (wording.learned_caption(5), 10), (wording.TOP_ZONE, 5),
                        (wording.CARD_CEILING, 5), (wording.back_to_batch_label(6, 3), 5),
                        (wording.called_shown(10), 10), (wording.OFF_BY_TITLE, 5)):
        assert words(text) <= limit, text


# ------------------------------------------------------------------ #
#  _BANNED
# ------------------------------------------------------------------ #

def _round2_source():
    src = pathlib.Path(wording.__file__).read_text()
    return src[src.index("# --- round 2 fixes ---"):]


def test_no_round_two_string_says_a_banned_word():
    tree = ast.parse(_round2_source())
    # Docstrings say what a function returns, not what reaches the screen;
    # _GLOSSARY_GONE names rows taken off the screen.
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr):
            skip.add(id(n.body[0].value))
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_GLOSSARY_GONE"
                                             for t in n.targets):
            skip |= {id(c) for c in ast.walk(n.value)}
    shown = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and id(n) not in skip]
    bad = [t for t in shown if any(p.search(t) for p in _BANNED)]
    assert not bad, bad
    rendered = [wording.ceiling_help("98.3", "100", "1.7", "2.3", "98.4", True, ["x"]),
                wording.PREDICTED_MEASUREMENTS_HELP, wording.LEARNED_RULE,
                wording.story_improving_help("84.2", "96.1", 1, 6, True, "2.3"),
                wording.why_chart_caption_full("3,000", True, True, True),
                wording.called_shown_help(["Tangy"], 10), wording.PROFILER_CRAMPED_HELP,
                wording.METHOD_STOP, *[c for r in wording.glossary_rows() for c in r[:1]]]
    assert not [t for t in rendered if any(p.search(t) for p in _BANNED)]
