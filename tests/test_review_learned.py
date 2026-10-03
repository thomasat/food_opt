"""Review learned (prototype, 2026-10-02; owner: the Review "doesn't inspire
confidence in the tool"): "What has the model learned?" leads with honest,
checkable evidence, right after "Are we improving?".

- Ordering accuracy: each Round's within-Round pairs, forecast before
  tasting (the Round-ahead checks), against chance; a measured tie is not
  counted.
- Verdict: Predictable at a one-sided binomial p < 0.05 with at least 5
  pairs; Partly when it leans right (p < 0.20); Not yet otherwise or when
  it barely varies.
- Stable effect: a 95 % posterior interval over the tested stretch that
  excludes 0, and the same sign under the last 3 Round-ahead refits.
- Make a round's forecast before tasting: Predictable measurements only,
  never at or past the top of the Range, from the same models the Round's
  check uses.
- Called it?: honest on a miss; above the range reads "better than
  forecast".
- What next?: the next suggestions previewed on a copy, the project and
  the RNG untouched, the same as Generate.

Synthetic: hand-made checks and the demo projects; no pilot data."""
import copy
import re

import pytest
import torch
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import learned
import ui_insights as U
import wording
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


# ------------------------------------------------------------------ #
#  Stable effects
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


@pytest.mark.parametrize("by_fit,stable", [
    ([(0, 1, 0.5, 0.1)] * 3, True),                                           # all +, CI > 0
    ([(0, 1, 0.5, 0.1), (0, 1, -0.2, 0.1), (0, 1, 0.5, 0.1)], False),         # a flip
    ([(0, 1, 0.5, 0.1), (0, 1, 0.4, 0.1), (0, 1, 0.1, 0.1)], False),          # CI spans 0
    ([None, (0, 1, 0.4, 0.1), (0, 1, 0.5, 0.1)], False),                      # not there
    ([(0, 1, -0.5, 0.1)] * 3, True),                                          # all −
])
def test_stable_effect_rule(line_a, monkeypatch, by_fit, stable):
    opt, _, _ = line_a
    refits = learned.refit_rows(opt)
    sizes = [len(rows) for _, rows in refits]

    class _F:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows

    monkeypatch.setattr(learned.insights, "fit", lambda o, n, rows: _F(n, rows))
    first = learned._columns(opt)[0][1]
    monkeypatch.setattr(learned, "effect_of", lambda o, f, rows, col:
                        by_fit[sizes.index(len(rows))] if col == first else None)
    eff = learned.effects(opt)
    got = [e for e in eff[insights.SCORE]]
    assert len(got) == 1 and got[0].stable is stable
    assert [e.stable for e in learned.stable(eff, insights.SCORE)] == ([True] if stable else [])


def test_effects_need_three_rounds(monkeypatch, line_a):
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


def test_section_order(screen):
    assert U.SECTION_ORDER[:2] == ("progress", "predictions")
    assert U._QUESTIONS["predictions"] == wording.LEARNED_TITLE
    assert _headings(screen) == [wording.Q2_TITLE, wording.LEARNED_TITLE, wording.Q3_TITLE,
                                 wording.Q4_TITLE, wording.Q5_TITLE]
    text = " | ".join(_text(el) for el, _, _ in _leaves(screen.tabs[3]))
    # The old trust section is gone from the default view.
    assert wording.Q1_TITLE not in text
    assert "Called it" in text or "called it" in text
    assert wording.learned_not_predictable([]).split(":")[0] in text


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
             wording.LEARNED_COL_EFFECTS, wording.LEARNED_CHART_X, wording.LEARNED_CHANCE,
             wording.LEARNED_CAPTION, wording.LEARNED_EFFECTS_CAPTION, wording.LEARNED_NONE_YET,
             wording.LEARNED_FORECAST_TITLE, wording.LEARNED_FORECAST_NONE,
             wording.LEARNED_PREVIEW_TITLE, wording.LEARNED_PREVIEW_CAPTION,
             wording.LEARNED_RULE, wording.LEARNED_TOP_RULE, wording.LEARNED_STABLE_RULE,
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
    for c in (wording.LEARNED_CAPTION, wording.LEARNED_EFFECTS_CAPTION,
              wording.LEARNED_PREVIEW_CAPTION, wording.LEARNED_FORECAST_NONE):
        assert _words(c) <= 8, c
    for label in (wording.LEARNED_STRIP_LABEL, wording.LEARNED_COL_PAIRS, wording.LEARNED_COL_TOP,
                  wording.LEARNED_CHART_X, wording.LEARNED_FORECAST_TITLE):
        assert _words(label) <= 5, label
    for names in (["Firmness"], ["Consistency", "Tangy", "Creaminess"], []):
        word, line = wording.learned_strip(names, [])
        assert _words(f"{word} {line}") <= 6
