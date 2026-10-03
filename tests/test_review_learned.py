"""Review learned (prototype, 2026-10-02; owner: the Review "doesn't inspire
confidence in the tool"): "What has the model learned?" leads with honest,
checkable evidence, right after "Are we improving?".

- Ordering accuracy: each Round's within-Round pairs, forecast before
  tasting (the Round-ahead checks), against chance; a measured tie is not
  counted.
- Verdict: Predictable at a one-sided binomial p < 0.05 with at least 5
  pairs; Partly when it leans right (p < 0.20); Not yet otherwise or when
  it barely varies.
- Clear effect (review learned 2, strict): a 95 % posterior interval over
  the tested stretch that excludes 0, with one sign, in EACH of the last 4
  Round-ahead refits; not in a moved-together pair; at least one noise SD.
  Anything less names no effect; the section has no effects column.
- Make a round's forecast before tasting: Predictable measurements only,
  never at or past the top of the Range, from the same models the Round's
  check uses.
- Called it?: at the top, under the strip, as charts (forecast range, tick,
  measured dot) with one answer line from the facts; honest on a miss;
  above the range reads "better than forecast"; only once the project's
  latest Round is scored.
- The section: one answer line, the ordering chart, its caption; no table.
- What next?: the next suggestions previewed on a copy, the project and
  the RNG untouched, the same as Generate.

Synthetic: hand-made checks and the demo projects; no pilot data."""
import copy
import json
import re

import pytest
import torch
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import learned
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from insights import Check, Prediction
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _install, _leaves, _text


def _check(no, r, name, pred, meas, raw=None, low=None, high=None, full=True):
    low = pred - 1 if low is None else low
    high = pred + 1 if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=pred if raw is None else raw, sd=0.5)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


# ------------------------------------------------------------------ #
#  Ordering accuracy and the verdict rule
# ------------------------------------------------------------------ #

def test_chance_tail_is_the_one_sided_binomial():
    assert learned.chance_tail(6, 6) == pytest.approx(1 / 64)
    assert learned.chance_tail(0, 4) == 1.0
    assert learned.chance_tail(16, 18) == pytest.approx(172 / 2 ** 18)


@pytest.mark.parametrize("right,pairs,ties,verdict,reason", [
    (6, 6, 3, learned.PREDICTABLE, ""),       # p = 0.016, 5+ pairs
    (5, 5, 0, learned.PREDICTABLE, ""),       # p = 0.031
    (4, 4, 0, learned.NOT_YET, "few"),        # under 5 pairs
    (16, 18, 6, learned.PREDICTABLE, ""),
    (15, 22, 2, learned.PARTLY, ""),          # p = 0.067: leans right
    (13, 22, 2, learned.NOT_YET, "chance"),   # p = 0.29
    (8, 18, 6, learned.NOT_YET, "chance"),    # below half
    (6, 6, 7, learned.NOT_YET, "flat"),       # more than half its pairs equal
])
def test_verdict_rule(right, pairs, ties, verdict, reason):
    o = learned.judge(learned.Ordering("m", right, pairs, ties))
    assert (o.verdict, o.reason) == (verdict, reason)


def test_ordering_counts_pairs_ties_and_the_top_pick():
    checks = [
        # Round 3: all three in the right order; the top one picked.
        _check(7, 3, "Firm", 8.0, 10), _check(8, 3, "Firm", 7.0, 9), _check(9, 3, "Firm", 6.0, 8),
        # Round 4: one measured tie (not counted), both other pairs wrong.
        _check(10, 4, "Firm", 5.0, 12), _check(11, 4, "Firm", 6.0, 12),
        _check(12, 4, "Firm", 7.0, 11),
    ]
    o = learned.ordering(checks, "Firm")
    assert (o.right, o.pairs, o.ties) == (3, 5, 1)
    assert o.chance == 2.5
    # Round 3 picked F7 (highest); Round 4 picked F12, measured lowest.
    assert (o.top_right, o.top_rounds) == (1, 2)
    assert o.top_chance == pytest.approx(1 / 3 + 2 / 3)


def test_ordering_reads_the_unclipped_mean():
    # Both clipped to the top (15) but the model's own means differ.
    checks = [_check(1, 2, "Firm", 15.0, 14, raw=16.0), _check(2, 2, "Firm", 15.0, 12, raw=15.5)]
    call = learned.round_call(checks, "Firm", 2)
    assert [(p.high, p.low, p.right) for p in call.pairs] == [(1, 2, True)]


def test_overall_score_counts_fully_measured_only():
    checks = [_check(1, 2, insights.SCORE, 90, 95), _check(2, 2, insights.SCORE, 80, 85),
              _check(3, 2, insights.SCORE, 85, 99, full=False)]
    o = learned.ordering(checks, insights.SCORE)
    assert (o.right, o.pairs) == (1, 1)


# ------------------------------------------------------------------ #
#  Called it?
# ------------------------------------------------------------------ #

def _called_opt():
    class _O:
        def measurements_by_importance(self):
            return [{"name": "Firm"}]
    return _O()


def test_called_it_is_honest_on_a_miss():
    # Rounds 1-2 earn Predictable (6 of 6); Round 3 is called in reverse.
    checks = []
    for r, base in ((1, 0), (2, 3)):
        checks += [_check(base + k, r, "Firm", 10 - k, 10 - k) for k in range(3)]
        checks += [_check(base + 10 + k, r, "Firm", 5 - k, 5 - k) for k in range(1)]
    checks += [_check(20 + k, 3, "Firm", 10 - k, 5 + k) for k in range(3)]
    r, calls = learned.called_it(_called_opt(), checks)
    assert r == 3 and [n for n, _ in calls] == ["Firm"]
    call = calls[0][1]
    assert (call.right, call.counted) == (0, 3)
    assert wording.called_pairs("Firm", call.right, call.counted, len(call.pairs)) == \
        "Firm: 0 of 3 pairs in the right order."
    marks = [wording.called_pair(p.high, p.low, p.right) for p in call.pairs]
    assert all(m.endswith("✗") for m in marks)
    assert wording.called_top(call.top_predicted, call.top_measured, call.top_right) == \
        "Forecast highest: F20 ✗ (measured: F22)"


def test_called_it_picks_by_the_verdict_before_the_round():
    # Not Predictable before Round 3 (too few pairs): nothing is called.
    checks = [_check(1, 1, "Firm", 2, 2), _check(2, 1, "Firm", 1, 1),
              _check(3, 2, "Firm", 2, 1), _check(4, 2, "Firm", 1, 2)]
    assert learned.called_it(_called_opt(), checks)[1] == []


def test_called_it_overall_score_landing_words():
    assert wording.called_range(1, 2, 0) == "1 of 3 in range, 2 better than forecast."
    assert wording.called_range(2, 0, 1) == "2 of 3 in range, 1 below forecast."
    # A measured tie is said as one; all equal: nothing to order.
    assert wording.called_pair(17, 16, None) == "F16 = F17 (tie)"
    assert wording.called_pairs("Tangy", 0, 0, 3) == "Tangy: all measured the same, nothing to order."


def _items(score_meas, firm_meas, score_pred=(90, 85, 80), firm_pred=(9, 8, 7)):
    """Called it? items for Round 3: the Overall score and Firm, three
    formulations (F7-F9) each, ranges ±1 (the score ±3)."""
    sc = [_check(7 + k, 3, insights.SCORE, p, m, low=p - 3, high=p + 3)
          for k, (p, m) in enumerate(zip(score_pred, score_meas))]
    fm = [_check(7 + k, 3, "Firm", p, m) for k, (p, m) in enumerate(zip(firm_pred, firm_meas))]
    checks = sc + fm
    return [(n, learned.round_call(checks, n, 3), learned.round_checks(checks, n, 3))
            for n in (insights.SCORE, "Firm")]


def _answer(items, record=(8, 8)):
    got = learned.called_summary(items)
    lab = lambda n: "Overall score" if n == insights.SCORE else n
    return got, wording.called_answer(3, got.verdict, lab(got.lead) if got.lead else None,
                                      record, got.top, [lab(n) for n in got.right],
                                      [lab(n) for n in got.off])


def test_called_it_leads_with_the_record_then_the_latest_call():
    got, line = _answer(_items((91, 84, 79), (9.2, 8.1, 6.8)))
    assert got.verdict == learned.CALLED and got.lead == "Firm" and got.top == 7
    # Persona fixes: the order, then the record so far; no "highest".
    assert line == "**Round 3: order called right.** So far: Firm 8 of 8."
    assert _words(line) <= 14


def test_called_it_is_judged_on_order_not_range():
    # Every value far outside its range, every order right: called (the
    # order is the claim); the ranges are in the "?".
    got, _ = _answer(_items((70, 60, 50), (20, 15, 1)))
    assert got.verdict == learned.CALLED


def test_called_it_above_the_range_is_better_than_forecast():
    got, _ = _answer(_items((97, 84, 79), (9.2, 8.1, 6.8)))
    assert got.verdict == learned.CALLED and got.above == 1
    assert wording.called_range(2, 1, 0) == "2 of 3 in range, 1 better than forecast."


def test_called_it_is_honest_on_a_miss_in_its_line():
    # Firm in reverse order: said plainly.
    got, line = _answer(_items((91, 84, 79), (6, 8.1, 9.5)))
    assert got.verdict == learned.CALLED_PARTLY and got.off == ["Firm"]
    assert line == "**Round 3: partly.** Overall score right; Firm order off."
    # Everything in reverse: missed.
    got, line = _answer(_items((60, 65, 70), (5, 6.5, 10)))
    assert got.verdict == learned.CALLED_MISSED
    assert line == "**Round 3: missed.** Overall score, Firm order off."


def test_a_result_measured_all_the_same_is_left_out():
    items = _items((91, 84, 79), (8, 8, 8))
    got = learned.called_summary(items)
    assert [n for n, _, _ in learned.shown(items)] == [insights.SCORE]
    assert got.tied == ["Firm"] and got.lead == insights.SCORE
    assert wording.called_tied("Tangy") == "Tangy: all measured the same, not drawn."


def test_slopegraph_marks_crossings_and_shares_tied_points(monkeypatch):
    class _O:
        pass
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0))
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    items = _items((91, 84, 79), (9, 9, 9.5))      # F7 = F8 measured; F9 highest: crossed
    name, call, cks = items[1]
    lines, labels = U.slope_frames(_O(), name, call, cks)
    measured = lines[lines.x == 1].set_index("f")
    assert measured.loc[7, "y"] == measured.loc[8, "y"] < measured.loc[9, "y"] == 1
    assert set(lines[lines.crossed].f) == {7, 8, 9}
    right = list(labels[labels.x == 1].text)
    assert len(right) == 2 and any("F7, F8" in t for t in right)
    # Held order: nothing crossed.
    name, call, cks = _items((91, 84, 79), (9.2, 8.1, 6.8))[1]
    lines, _ = U.slope_frames(_O(), name, call, cks)
    assert not lines.crossed.any()
    # Review readable: two near labels move apart as a block centred on
    # their points (was: the upper one pushed up); a lone one stays put.
    got = U._spread([0.0, 0.05, 1.0])
    assert got[1] - got[0] == pytest.approx(U._SLOPE_GAP)
    assert (got[0] + got[1]) / 2 == pytest.approx(0.025) and got[2] == 1.0


def test_called_it_shows_only_for_the_latest_scored_round():
    class _O:
        batch_history = [1, 1, 2, 2]
        pending_batch = None
        pending_batch_no = None
    o = _O()
    assert learned.is_latest(o, 2) and not learned.is_latest(o, 1)
    o.pending_batch, o.pending_batch_no = [{"formulation": 5}], 3   # Round 3 open
    assert not learned.is_latest(o, 2)


# ------------------------------------------------------------------ #
#  Clear effects (the strict rule)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-learned"))
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        memo = {}
        with insights.memo_in_use(memo):
            checks = insights.check_rounds(opt)
        yield opt, checks, memo
    mp.undo()


def test_refits_are_the_round_ahead_training_rows(line_a):
    opt, _, _ = line_a
    order = insights.round_order(opt)
    refits = learned.refit_rows(opt)
    assert len(refits) == learned.STABLE_REFITS
    # Through each Round: the next Round's check trains on exactly these.
    for (r, rows), (nxt, first) in zip(refits, order[-learned.STABLE_REFITS + 1:]):
        assert rows == list(range(first))
    assert refits[-1][1] == list(range(len(opt.Y_history)))


@pytest.mark.parametrize("got,together,noise,steady,claim", [
    ([(0, 1, 0.5, 0.1)] * 4, False, 0.3, True, True),                   # all +, every CI > 0
    ([(0, 1, -0.5, 0.1)] * 4, False, 0.3, True, True),                  # all −
    ([(0, 1, 0.5, 0.1)] * 3 + [(0, 1, -0.5, 0.1)], False, 0.3, False, False),   # sign flips
    ([(0, 1, 0.1, 0.1)] + [(0, 1, 0.5, 0.1)] * 3, False, 0.3, False, False),    # an early CI spans 0
    ([None] + [(0, 1, 0.5, 0.1)] * 3, False, 0.3, False, False),        # a refit without it
    ([(0, 1, 0.5, 0.1)] * 4, True, 0.3, True, False),                   # moved together
    ([(0, 1, 0.5, 0.1)] * 4, False, 0.6, True, False),                  # under one noise SD
    ([(0, 1, 0.5, 0.1)] * 4, False, None, True, False),                 # noise unknown
])
def test_clear_effect_rule(got, together, noise, steady, claim):
    assert learned.judge_effect(got, together, noise) == (steady, claim)


def _patched_effects(opt, monkeypatch, by_fit, moved=(), noise=0.1):
    refits = learned.refit_rows(opt)
    sizes = [len(rows) for _, rows in refits]

    class _F:
        model = None

        def __init__(self, name, rows):
            self.name, self.rows = name, rows

    monkeypatch.setattr(learned.insights, "fit", lambda o, n, rows: _F(n, rows))
    monkeypatch.setattr(learned.insights, "moved_pairs", lambda o: list(moved))
    monkeypatch.setattr(learned, "noise_of", lambda o, n, f, c=(): noise)
    var, first = learned._columns(opt)[0]
    monkeypatch.setattr(learned, "effect_of", lambda o, f, rows, col:
                        by_fit[sizes.index(len(rows))] if col == first else None)
    return var, learned.effects(opt)


def test_unstable_sign_names_no_effect(line_a, monkeypatch):
    opt, _, _ = line_a
    _, eff = _patched_effects(opt, monkeypatch, [(0, 1, 0.5, 0.1), (0, 1, -0.4, 0.1),
                                                 (0, 1, 0.5, 0.1), (0, 1, 0.5, 0.1)])
    assert len(eff[insights.SCORE]) == 1 and not eff[insights.SCORE][0].stable
    assert learned.stable(eff, insights.SCORE) == []


def test_moved_together_names_no_effect(line_a, monkeypatch):
    opt, _, _ = line_a
    var = learned._columns(opt)[0][0]
    steady = [(0, 1, 0.5, 0.1)] * learned.STABLE_REFITS
    _, eff = _patched_effects(opt, monkeypatch, steady, moved=[(var, "Other", 0.8)])
    e = eff[insights.SCORE][0]
    assert e.steady and e.together and not e.stable
    assert learned.stable(eff, insights.SCORE) == []
    # The same, not moved together: named.
    _, eff = _patched_effects(opt, monkeypatch, steady)
    assert [e.var for e in learned.stable(eff, insights.SCORE)] == [var]


def test_too_small_names_no_effect(line_a, monkeypatch):
    opt, _, _ = line_a
    steady = [(0, 1, 0.5, 0.1)] * learned.STABLE_REFITS
    _, eff = _patched_effects(opt, monkeypatch, steady, noise=0.8)
    assert learned.stable(eff, insights.SCORE) == []


def test_noise_stand_in_is_conservative():
    checks = [_check(1, 2, "Firm", 5, 7), _check(2, 2, "Firm", 5, 3)]
    assert learned.forecast_miss(checks, "Firm") == pytest.approx(2.0)
    assert learned.forecast_miss(checks, "Other") is None


def test_effects_need_four_rounds(monkeypatch, line_a):
    opt, _, _ = line_a
    monkeypatch.setattr(learned, "refit_rows", lambda o: [(1, [0, 1, 2, 3, 4])])
    assert learned.effects(opt) == {}


def test_effect_of_is_finite_on_a_real_model(line_a):
    opt, _, memo = line_a
    rows = list(range(len(opt.Y_history)))
    with insights.memo_in_use(memo):
        f = insights.fit(opt, insights.SCORE, rows)
    lo, hi, mean, sd = learned.effect_of(opt, f, rows, learned._columns(opt)[0][1])
    assert lo < hi and sd > 0 and abs(mean) < 100


# ------------------------------------------------------------------ #
#  Before you taste: the forecast panel
# ------------------------------------------------------------------ #

def test_capped_never_reads_100():
    assert U.capped(15.0, 15.0) == "≥ 15"
    assert U.capped(14.96, 15.0) == "≥ 15"
    assert U.capped(13.52, 15.0) == "13.5"
    assert U.capped(99.6, 100.0, 0) == "≥ 99"
    assert U.score_text(Prediction(99.7, 95.2, 100.0)) == "≥ 99 (95–≥ 99)"
    assert "100" not in U.score_text(Prediction(98.0, 92.0, 104.0))


def test_forecast_panel_shows_predictable_measurements_only(line_a, monkeypatch):
    opt, checks, memo = line_a
    monkeypatch.setattr(U.st, "session_state", {U.MEMO_KEY: (opt.project_name, memo)})
    ords = learned.orderings(opt, checks)
    predictable = [n for n, o in ords.items()
                   if n != insights.SCORE and o.verdict == learned.PREDICTABLE]
    assert predictable          # the demo's clean panel orders Firmness right
    lines, help_ = U.round_forecast_lines(opt, checks)
    assert [ln.split(":")[0] for ln in lines] == predictable
    assert all(not re.search(r"\b1\d\d(\.\d)?\b", ln) for ln in lines)   # no 100+
    open_nos = [r["formulation"] for r in U.open_rows(opt)]
    assert all(f"F{no} " in lines[0] for no in open_nos)
    assert "Predictable" in help_
    # Nothing Predictable: no lines.
    flat = [c for c in checks if c.name == insights.SCORE]
    assert U.round_forecast_lines(opt, flat) == ([], None)


def test_forecast_is_the_round_ahead_check(line_a):
    """The forecast before the Round's results is the one its check reads
    after: the same rows, the same model."""
    opt, checks, memo = line_a
    last = insights._ordered_rounds(checks)[-1]
    rows = [i for i, b in enumerate(opt.batch_history) if b == last]
    probe = copy.copy(opt)
    probe.pending_batch_no = last
    open_rows = [{"formulation": int(opt.formulation_ids[i]), "recipe": opt.recipe_history[i]}
                 for i in rows]
    name = next(o["name"] for o in opt.objectives)
    with insights.memo_in_use(memo):
        got = learned.forecast(probe, open_rows, [name])
    for c in checks:
        if c.round == last and c.name == name:
            assert got[c.formulation][name].raw_mean == pytest.approx(c.raw_mean, abs=1e-6)


# ------------------------------------------------------------------ #
#  What next?: the preview
# ------------------------------------------------------------------ #

def test_preview_is_what_generate_makes_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = D.load(D.UPGRADE)
    assert not opt.pending_batch
    before = opt.export_json()
    torch.manual_seed(123)
    rng = torch.random.get_rng_state()
    got = learned.preview_next(opt, 3)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert opt.export_json() == before and not opt.pending_batch
    assert len(got) == 3
    # The real Generate (ask() on the project itself) makes the same three.
    with insights.RNG_LOCK:
        made = opt.ask(n_suggestions=3)
    assert made == got


def test_preview_copy_never_saves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = D.load(D.UPGRADE)
    clone = learned.preview_copy(opt)
    assert clone.storage is not opt.storage
    clone.save()
    clone.storage.save("x", "{}")
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ #
#  The screen: section order, strip, words
# ------------------------------------------------------------------ #

def _headings(at):
    return [_text(el)[5:] for el, _, _ in _leaves(at.tabs[3])
            if el.type == "markdown" and _text(el).startswith("#### ")]


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-learned-ui"))
    with bo_views.fresh():
        _install(D.LINE_A)
        at = AppTest.from_file(APP_PATH, default_timeout=900)
        at.session_state["_loaded_project"] = D.LINE_A
        at.run()
        at.session_state["main_tab"] = wording.TAB_INSIGHTS
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        yield at
    mp.undo()


def _section_leaves(at, title):
    """The leaves of the section headed `title`, up to the next heading."""
    out, inside = [], False
    for el, _, _ in _leaves(at.tabs[3]):
        t = _text(el)
        if el.type == "markdown" and t.startswith("#### "):
            inside = t[5:] == title
            continue
        if inside:
            out.append(el)
    return out


def test_section_order(screen):
    assert U.SECTION_ORDER[:2] == ("progress", "predictions")
    assert U._QUESTIONS["predictions"] == wording.LEARNED_TITLE
    assert _headings(screen) == [wording.Q2_TITLE, wording.LEARNED_TITLE, wording.Q3_TITLE,
                                 wording.Q4_TITLE, wording.Q5_TITLE]
    text = " | ".join(_text(el) for el, _, _ in _leaves(screen.tabs[3]))
    # The old trust section is gone from the default view.
    assert wording.Q1_TITLE not in text
    assert wording.learned_not_predictable([]).split(":")[0] in text
    # Round 6 is open: no Called it? until it is scored.
    assert "called it" not in text.lower() and "Round 5: partly" not in text


def test_section_is_one_line_one_chart_one_caption(screen):
    els = _section_leaves(screen, wording.LEARNED_TITLE)
    kinds = [el.type for el in els]
    assert not any(k in ("table", "dataframe", "arrow_data_frame", "arrow_table") for k in kinds)
    texts = [_text(el) for el in els if _text(el)]
    assert len(texts) == 2, texts            # the answer and the caption
    assert texts[0].startswith("**Predicts") or texts[0].startswith("**Partly")
    assert texts[1] == wording.LEARNED_CAPTION
    assert sum("vega" in k or "chart" in k for k in kinds) == 1
    every = " | ".join(_text(el) for el, _, _ in _leaves(screen.tabs[3]))
    for gone in ("Learned effects", "Raised by", "Lowered by", "Picked the top one",
                 "Measurements miss more than expected"):
        assert gone not in every


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """Line A with Round 6 closed unrecorded: Round 5 is the latest scored."""
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-learned-scored"))
    with bo_views.fresh():
        state = json.loads((D.HERE / f"{D.LINE_A}.json").read_text())
        opt = FoodOptimizer(state["project_name"])
        opt.import_json(state)
        opt._drop_pending_batch()
        opt.save()
        at = AppTest.from_file(APP_PATH, default_timeout=900)
        at.session_state["_loaded_project"] = D.LINE_A
        at.run()
        at.session_state["main_tab"] = wording.TAB_INSIGHTS
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        yield at
    mp.undo()


def test_called_it_sits_at_the_top_as_charts(scored):
    leaves = [(el, _text(el)) for el, _, _ in _leaves(scored.tabs[3])]
    texts = [t for _, t in leaves]
    first_heading = next(i for i, t in enumerate(texts) if t.startswith("#### "))
    assert texts[first_heading] == f"#### {wording.Q2_TITLE}"
    called = [i for i, t in enumerate(texts)
              if re.match(r"\*\*Round 5: (partly|missed|order called right|Overall order called "
                          r"right)\.\*\*", t)]
    assert called and called[0] < first_heading
    line = texts[called[0]]
    # Persona fixes: the coordinator's headline, the Overall score's order
    # first ("F16, F17 above F18; scored about 4 better than forecast").
    assert _words(line) <= 16, line
    assert "calls right" not in line
    # Visual: a chart and the caption, before the first heading.
    between = leaves[called[0] + 1:first_heading]
    assert any("vega" in el.type or "chart" in el.type for el, _ in between)
    assert wording.CALLED_CAPTION in [t for _, t in between]
    # The pairs, one ✓ or ✗ each, are in its "?", not on the page.
    el = leaves[called[0]][0]
    assert "✓" in el.help or "✗" in el.help or "tie" in el.help
    assert not any("✓" in t or "✗" in t for t in texts)


def test_called_chart_is_at_most_three_small_panels(scored):
    # Any LearnedFacts the scored screen worked out: its chart's panels.
    lf = scored.session_state[U.LEARNED_KEY][1]
    drawn = learned.shown(lf.items)
    assert 1 <= len(drawn) <= 1 + learned.CALLED_PANELS
    assert lf.items[0][0] == insights.SCORE
    # Review readable: taller (120 -> 160 px) so 12 px end labels clear each other.
    assert U.CALLED_W <= 260 and U.CALLED_H <= 170


def test_profiler_headline_names_no_effect(screen):
    els = _section_leaves(screen, wording.Q3_TITLE)
    answer = next(_text(el) for el in els if el.type == "markdown" and _text(el))
    assert answer in {wording.profiler_guide(k) for k in ("ingredients", "settings", "both")}
    assert wording.PROFILER_NO_CLAIM in answer_help(els)
    for kind in ("ingredients", "settings", "both"):
        assert _words(wording.profiler_guide(kind)) <= 8


def answer_help(els):
    return next(el.help for el in els if el.type == "markdown" and _text(el))


def test_strip_reads_the_model_verdict(screen):
    labels = [b.label for b in screen.tabs[3].button if b.key == f"{U.SEE_CHART_KEY}_predictions"]
    assert labels and labels[0].startswith("**Predicts**")
    assert _words(labels[0]) <= 6


def test_no_prediction_reads_100_on_the_review(screen):
    for el, _, _ in _leaves(screen.tabs[3]):
        if el.type in ("markdown", "caption"):
            assert not re.search(r"\(\d+–100\)|\b100 \(", _text(el)), _text(el)


def test_wording_passes_banned_and_limits():
    texts = [wording.LEARNED_TITLE, wording.LEARNED_STRIP_LABEL, *wording.LEARNED_WORDS.values(),
             *wording.LEARNED_REASONS.values(), wording.LEARNED_COL_RESULT,
             wording.LEARNED_COL_VERDICT, wording.LEARNED_COL_PAIRS, wording.LEARNED_COL_TOP,
             wording.LEARNED_CHART_X, wording.LEARNED_CHANCE,
             wording.LEARNED_CAPTION, wording.LEARNED_NONE_YET,
             wording.CALLED_CAPTION, wording.CALLED_RULE, *wording.CALLED_WORDS.values(),
             wording.CALLED_TIP_FORECAST, wording.CALLED_TIP_MEASURED,
             wording.called_answer(6, "called", "Thickness uniformity", (16, 18), 18),
             wording.called_answer(10, "partly", right=["Consistency"],
                                   off=["Overall score", "Thickness uniformity"]),
             wording.called_answer(10, "missed", off=["Firmness"]),
             wording.learned_counts([("Overall score", 10, 11, 5.5)]),
             wording.learned_tops([("Overall score", 3, 4, 1.7)]),
             wording.LEARNED_FORECAST_TITLE, wording.LEARNED_FORECAST_NONE,
             wording.LEARNED_PREVIEW_TITLE, wording.LEARNED_PREVIEW_CAPTION,
             wording.LEARNED_RULE, wording.LEARNED_TOP_RULE, wording.PROFILER_NO_CLAIM,
             *(wording.profiler_guide(k) for k in ("ingredients", "settings", "both")),
             wording.called_tied("Tangy"), wording.called_ranges(["F16 13.5 (11.1–≥ 15)"]),
             wording.METHOD_LEARNED_TITLE, wording.METHOD_FORECAST_CHART,
             wording.learned_answer(["Firmness"], ["Juiciness"], ["Cook loss"]),
             wording.learned_skill_line(22, 24, 0.61, -0.11, 24),
             wording.forecast_help(15, 6), wording.preview_help(3, 10),
             wording.preview_line(1, "explores", "98 (85–≥ 99)", "Salt +0.10 g"),
             wording.learned_effect_tip("Salt", "Tangy", "0.2", "1.2", 0.8, 0.1, 1.5),
             wording.review_status_learned("Predicts", "Flat")]
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t
        assert not re.search(r"\bgain\b", t, re.I), t
    # Captions at most 8 words, labels at most 5, the strip at most 6.
    for c in (wording.LEARNED_CAPTION, wording.CALLED_CAPTION,
              wording.LEARNED_PREVIEW_CAPTION, wording.LEARNED_FORECAST_NONE):
        assert _words(c) <= 8, c
    for label in (wording.LEARNED_STRIP_LABEL, wording.LEARNED_COL_PAIRS, wording.LEARNED_COL_TOP,
                  wording.LEARNED_CHART_X, wording.LEARNED_FORECAST_TITLE):
        assert _words(label) <= 5, label
    for line in (wording.called_answer(6, "called", "Thickness uniformity", (16, 18), 18),
                 wording.called_answer(10, "partly", right=["Consistency"],
                                       off=["Overall score", "Tangy"])):
        assert _words(line) <= 12, line
    for names in (["Firmness"], ["Consistency", "Tangy", "Creaminess"], []):
        word, line = wording.learned_strip(names, [])
        assert _words(f"{word} {line}") <= 6
