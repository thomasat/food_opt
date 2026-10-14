"""Review critique (prototype, 2026-10-04): one fix pass from three
critiques of the Review (clarity, statistics, confidence).

- Forecast check: one shared axis per result — a row per formulation in
  forecast order, the 95 % range a light bar, the forecast a tick, the
  measured value a dot, a miss in the accent colour labelled above/below;
  measured ties counted aloud.
- Stop signal: within the tie threshold of the top score, "Within noise of
  the top score. Stop, or change the target." and no widening advice;
  else every binding limit named, a moved-together ingredient said.
- Order: an exact within-Round permutation test; the chart's chance range
  is the same null, so the dot clears it exactly when the verdict is
  Predictable; under 5 pairs: "too few pairs", no dot.
- Noise: a measurement's share of the Overall score's assumed 1 point,
  σ × w ÷ (s × √k): 0.3 on a 0–15 target worth 25 of 4.
- Page order; "Show as lines" gone; Start vs best, against target; range
  notation; Results' In / Above / Below; tooltips of about 120 words.

Synthetic projects and hand-made checks only; no pilot data."""
import ast
import json
import os
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

import bo_views
import insights
import learned
import ui_insights as U
import wording
from insights import Check
from tests.test_food_bo import _BANNED


def _check(no, r, name, pred, meas, low=None, high=None, full=True):
    low = pred - 1 if low is None else low
    high = pred + 1 if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=pred, sd=0.5)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0 if n == insights.SCORE
                                                                else 15.0))
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)


class _Obj:
    """A project the helpers read: one target measurement."""

    def __init__(self):
        self.objectives = [{"name": "Consistency", "goal": "target", "target": 12,
                            "weight": 100, "min_val": 0, "max_val": 15}]
        self.formulation_ids = list(range(1, 40))

    def measurements_by_importance(self):
        return list(self.objectives)


def _round6():
    """Round 6: F16 and F17 measured equal (a tie), both above their range;
    F18 inside; forecasts 84.3 / 84.7 / 83.7."""
    checks = []
    for r, base in ((4, 10), (5, 13)):
        checks += [_check(base + k, r, insights.SCORE, 95 - 3 * k, 95 - 3 * k, low=92 - 3 * k,
                          high=98 - 3 * k) for k in range(3)]
    for no, sp, sm in zip((16, 17, 18), (84.3, 84.7, 83.7), (88.0, 88.0, 81.5)):
        checks.append(_check(no, 6, insights.SCORE, sp, sm, low=sp - 2.5, high=sp + 2.5))
    return checks


def _lf(opt, checks, r, noise):
    orderings = learned.orderings(opt, checks, noise)
    call = learned.round_call(checks, insights.SCORE, r, noise[insights.SCORE])
    items = [(insights.SCORE, call, learned.round_checks(checks, insights.SCORE, r))]
    lf = U.LearnedFacts(orderings, {}, (r, []), call, (1, 2, 0), "", items)
    lf.noise, lf.checks = noise, checks
    return lf


# ------------------------------------------------------------------ #
#  1. Forecast check: shared axis, misses flagged, ties counted
# ------------------------------------------------------------------ #

def test_forecast_check_rows_follow_the_forecast_and_flag_misses(plain):
    opt = _Obj()
    cks = learned.round_checks(_round6(), insights.SCORE, 6)
    frame = U.check_frame(opt, insights.SCORE, cks)
    # Row order is the forecast order, highest first.
    assert list(frame["f"]) == ["F17", "F16", "F18"]
    assert dict(zip(frame["f"], frame["side"])) == {"F17": "above", "F16": "above", "F18": "in"}
    # Round 2: a miss prints its measured value ("88.0 above").
    assert set(frame["miss"]) == {"88.0 above", ""}


def test_forecast_check_chart_is_one_axis_per_result_with_an_accent_for_misses(plain):
    opt = _Obj()
    lf = _lf(opt, _round6(), 6, {insights.SCORE: 1.0, "Consistency": 0.3})
    spec = U.called_chart(opt, lf).to_dict()
    panel = spec["vconcat"][0]     # round 2: panels one under another
    marks = [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
             for layer in panel["layer"]]
    assert marks[:3] == ["bar", "tick", "point"] and marks.count("text") == 2
    # One x scale: range, forecast and measured all on the same values.
    xs = [layer["encoding"]["x"] for layer in panel["layer"]]
    assert {x["field"] for x in xs} == {"low", "predicted", "measured"}
    domains = {json.dumps(x["scale"]["domain"]) for x in xs}
    assert len(domains) == 1
    colour = panel["layer"][2]["encoding"]["color"]["scale"]
    assert colour["domain"][-1] == wording.CHECK_KEY_OUT
    assert colour["range"][-1] == U.outline_colour()
    assert wording.CALLED_CAPTION == "Rows: forecast order, highest first."


def test_the_check_line_counts_ties(plain):
    opt = _Obj()
    lf = _lf(opt, _round6(), 6, {insights.SCORE: 1.0, "Consistency": 0.3})
    line = U.called_line(opt, lf)
    # Round 2: no brackets inside the line (14 words at most).
    assert "order right: 1 pair, 1 too close, 1 tie" in line
    assert wording.check_order(1, 1, 1, ties=1) == "order right: 1 pair, 1 too close, 1 tie"
    assert wording.check_order(0, 0, 2) == "order too close to call"


# ------------------------------------------------------------------ #
#  2. The stop signal at the top score
# ------------------------------------------------------------------ #

def _yard(value=1.0, assumed=True):
    return insights.Yardstick(value, None, 0, False, assumed=assumed, df=0, groups=0, left_out=0)


def test_ceiling_reads_within_the_tie_threshold():
    opt = NS(utility_ceiling=lambda: 100.0)
    need = U.tie_need(_yard())
    assert need == pytest.approx(2.33, abs=0.01)
    assert U.at_ceiling(opt, NS(best=NS(score=99.2), yardstick=_yard()))
    assert U.at_ceiling(opt, NS(best=NS(score=98.3), yardstick=_yard()))
    assert not U.at_ceiling(opt, NS(best=NS(score=95.0), yardstick=_yard()))
    assert not U.at_ceiling(opt, NS(best=None, yardstick=_yard()))


def test_the_ceiling_card_says_stop_and_never_widen(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    opt = NS(pending_batch=[], pending_batch_no=None, next_batch_no=lambda: 11,
             utility_ceiling=lambda: 100.0)
    step = insights.NextStep("close", binding=insights.Binding("highest", "Water", "Water"))
    facts = NS(next=step, best=NS(score=99.2, formulation=21), yardstick=_yard())
    card = U.next_card(opt, facts)
    assert card.word == wording.CARD_CEILING and card.lines == []
    assert wording.answer_line(wording.CEILING_HEAD, wording.CEILING_TAIL) == (
        "**Within noise of the top score.** Stop, or change the target.")
    # Round 2: the gap and the threshold in numbers, and the noisy-perfect score.
    tip = wording.ceiling_help("99.2", "100", "0.8", "2.3", "98.4", True,
                               ["Water's Highest (42 g)"])
    assert "Water's Highest (42 g)" in tip and "assumed noise" in tip
    assert tip.endswith(wording.REPEAT_CONFIRMS)


def test_widening_names_every_limit_and_a_moved_pair():
    line = wording.widen_all(["Tap water's Highest (42 g)", "Tapioca starch's Highest (2.6 g)",
                              "the Batch total (≤ 98.7 g)"], predicted=True,
                             moved=[("Tap water", "Coconut cream")])
    assert line.startswith("The predicted best sits at Tap water's Highest (42 g), Tapioca "
                           "starch's Highest (2.6 g) and the Batch total (≤ 98.7 g). To go "
                           "further, widen them together.")
    assert "Tap water moved together with Coconut cream" in line


# ------------------------------------------------------------------ #
#  3. Ordering: the exact within-Round permutation test
# ------------------------------------------------------------------ #

def _perfect(rounds, n=3, name="Consistency"):
    checks = []
    for r in range(rounds):
        checks += [_check(10 * r + k, r + 1, name, 10 - 2 * k, 10 - 2 * k) for k in range(n)]
    return checks


def test_permutation_p_is_exact():
    # Each Round of 3 in perfect order: P(S = 3) = 1/6; two Rounds 1/36.
    o = learned.ordering(_perfect(2), "Consistency", 0.0)
    assert (o.right, o.pairs) == (6, 6)
    assert o.p == pytest.approx(1 / 36)
    # The binomial on pairs claimed 1/64 (pairs within a Round treated as
    # independent).
    assert learned.chance_tail(6, 6) == pytest.approx(1 / 64)
    one = learned.ordering(_perfect(1), "Consistency", 0.0)
    assert one.p == pytest.approx(1 / 6)


def test_round_null_sums_to_one_and_is_symmetric():
    cks = _perfect(1, n=4)
    null = learned.round_null(cks)
    assert sum(null.values()) == pytest.approx(1.0)
    assert null[6] == pytest.approx(1 / 24) and null[-6] == pytest.approx(1 / 24)


@pytest.mark.parametrize("seed", range(12))
def test_the_chance_range_and_the_verdict_always_agree(seed):
    import random
    rng = random.Random(seed)
    checks = []
    for r in range(rng.randint(2, 6)):
        preds = [rng.uniform(0, 10) for _ in range(3)]
        for k, p in enumerate(preds):
            meas = p + rng.gauss(0, rng.choice([0.5, 2.0, 5.0]))
            checks.append(_check(10 * r + k, r + 1, "Consistency", p, round(meas)))
    o = learned.ordering(checks, "Consistency", 0.0)
    if o.pairs < learned.MIN_PAIRS:
        return
    lo, hi = o.chance_share()
    beyond = o.right / o.pairs > hi + 1e-12
    assert beyond == (o.p < learned.VERDICT_P)
    if o.verdict == learned.PREDICTABLE:
        assert beyond


def test_too_few_pairs_draw_no_dot(plain):
    opt = _Obj()
    checks = _perfect(1)          # 3 pairs: under MIN_PAIRS
    lf = NS(orderings={"Consistency": learned.ordering(checks, "Consistency", 0.0)})
    [row] = U.learned_rows(opt, lf)
    assert row["few"] and row["pct"] is None and row["lo"] is None
    spec = json.dumps(U.learned_chart(opt, lf).to_dict())
    # Specific wording (2026-10-12), re-pinned: "under 5 pairs", not "too few".
    assert wording.learned_too_few(learned.MIN_PAIRS) in spec


def test_the_ordering_chart_has_a_legend_and_the_multiplicity_line(plain):
    opt = _Obj()
    lf = NS(orderings={"Consistency": learned.ordering(_perfect(3), "Consistency", 0.0)})
    spec = json.dumps(U.learned_chart(opt, lf).to_dict())
    assert wording.LEARNED_KEY_CHANCE in spec and '"orient": "bottom"' in spec
    # Round 2: Holm across the results checked.
    assert wording.learned_multiplicity(5) == (
        "Adjusted for 5 results checked: one alone could pass by chance.")
    assert "binomial" not in wording.LEARNED_RULE


# ------------------------------------------------------------------ #
#  Noise: each measurement's share of the Overall score's 1 point
# ------------------------------------------------------------------ #

def test_noise_derivation_gives_0_3(monkeypatch):
    monkeypatch.setattr(learned, "_repeat_noise", lambda opt, name: None)
    objs = [{"name": n, "goal": "target", "target": 5, "weight": 25, "min_val": 0,
             "max_val": 15} for n in ("A", "B", "C", "D")]
    opt = NS(objectives=objs, measurements_by_importance=lambda: objs)
    noise = learned.pair_noise(opt)
    assert noise[insights.SCORE] == 1.0
    assert all(noise[n] == pytest.approx(0.3) for n in "ABCD")
    # Two measurements, 60 / 40 on 0–10: 1 × 10 ÷ (60 × √2), 1 × 10 ÷ (40 × √2).
    two = [dict(objs[0], weight=60, max_val=10), dict(objs[1], weight=40, max_val=10)]
    opt2 = NS(objectives=two, measurements_by_importance=lambda: two)
    got = learned.pair_noise(opt2)
    assert got["A"] == pytest.approx(10 / (60 * 2 ** 0.5))
    assert got["B"] == pytest.approx(10 / (40 * 2 ** 0.5))


def test_the_rise_says_its_assumed_noise():
    assert wording.progress_rise_answer("95.0", "98.3", "3.3", "3.0") == (
        "Best 95.0 → 98.3 (+3.3; needs 3.0 if tasting noise ≈ 1 point)")
    assert wording.progress_rise_answer("95.0", "98.3", "3.3", "3.0", assumed=False) == (
        "Best 95.0 → 98.3 (+3.3; needs 3.0)")


# ------------------------------------------------------------------ #
#  Start vs best, against target
# ------------------------------------------------------------------ #

def _rows():
    def r(name, start, best, target=None, goal="target"):
        return {"name": name, "goal": goal, "target": target, "low": 0.0, "high": 15.0,
                "start": start, "best": best, "unit": "/15"}
    return [r("Consistency", 8, 12, 12), r("Creaminess", 4, 5, 5), r("Tangy", 5, 5, 5),
            r("Thickness", 13, 12.5, 13)]


def test_target_moves_and_answer_line(monkeypatch):
    monkeypatch.setattr(U, "target_rows", lambda opt: _rows())
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    assert [U.target_move(r)[0] for r in _rows()] == ["onto", "onto", "stayed", "away"]
    assert U.target_answer(None) == "2 moved onto target, 1 stayed on target, 1 moved 0.5 away."
    assert wording.STORY_TARGET_TITLE == "Start vs best, against target"


def test_target_chart_colours_segments_and_never_hides_the_start(monkeypatch):
    # Slim Review (re-pinned): target_rows takes the slider Round's best.
    monkeypatch.setattr(U, "target_rows", lambda opt, best=None: _rows())
    monkeypatch.setattr(U, "story_start", lambda opt: (0, 1))
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    opt = NS(best_index=lambda: 1)
    spec = U.target_chart(opt).to_dict()
    layers = spec["layer"]
    datasets = spec["datasets"]
    joins = next(v for v in datasets.values() if v and "t0" in v[0] and "what" in v[0])
    kinds = {j["what"] for j in joins}
    assert kinds == {wording.TARGET_KEY_CLOSER, wording.TARGET_KEY_AWAY}
    assert any(j["note"] == wording.TARGET_AWAY_LABEL for j in joins)
    # The target a diamond, the start a ring larger than the best's dot.
    pts = [l for l in layers if isinstance(l.get("mark"), dict) and l["mark"]["type"] == "point"]
    sizes = {l["mark"].get("shape", "circle"): l["mark"].get("size") for l in pts
             if l["mark"].get("filled") is False}
    assert sizes["diamond"] > sizes["circle"]
    best_dot = next(l for l in pts if l["mark"].get("filled") is True and l["mark"].get("size") == 70)
    assert best_dot
    # Tangy: start = best = target; its start point is still drawn.
    drawn = [p for v in datasets.values() for p in v
             if isinstance(p, dict) and p.get("k") == 2 and "tip" in p]
    # Round 2: the start's legend reads "Start (Round 1's best)".
    assert {p["what"] for p in drawn} >= {"Start (Round 1's best)", wording.BEST_SO_FAR_COLUMN}


# ------------------------------------------------------------------ #
#  Range notation; Results' flags
# ------------------------------------------------------------------ #

def test_range_never_prints_a_top_below_its_point():
    p = insights.Prediction(99.5, 93.4, 100.0, 99.5, 3.0)
    # Round 2: the range's top with the point's decimals.
    assert U.capped_range(p, 100.0) == "99.5 (93.4–100.0)"
    assert U.capped_range(p, 100.0, 0) == "≥ 99 (93–100)"
    assert U.capped_range(insights.Prediction(13.5, 11.1, 15.0, 13.5, 1.0), 15.0) == (
        "13.5 (11.1–15.0)")
    assert U.capped_range(insights.Prediction(94.3, 91.9, 96.8, 94.3, 1.0), 100.0) == (
        "94.3 (91.9–96.8)")
    assert U.score_text(insights.Prediction(99.4, 91.0, 100.0, 99.4, 3.0)) == "99 (91–100)"
    for text in (U.capped_range(p, 100.0), U.score_text(p)):
        assert "–≥" not in text


def test_results_rows_flag_in_above_below(plain):
    opt = _Obj()
    checks = _round6()
    checks += [_check(no, 6, "Consistency", 10.0, m, low=9.0, high=11.0)
               for no, m in ((16, 12.0), (17, 10.5), (18, 8.0))]
    frame = U.round_predicted_frame(opt, checks, 6)
    col = wording.RESULTS_LANDED_COLUMN
    assert col == "In / Above / Below range"
    got = {(r[wording.RESULTS_RESULT_COLUMN], r[wording.FORMULATION_CAP]): r[col]
           for _, r in frame.iterrows()}
    # Round 2: F16 in the table's Formulation column.
    assert got[(wording.OVERALL_SCORE_COLUMN, "F16")] == "Above"
    assert got[(wording.OVERALL_SCORE_COLUMN, "F18")] == "In"
    assert got[("Consistency", "F16")] == "Above"
    assert got[("Consistency", "F18")] == "Below"
    # A row per measurement, not only the Predictable ones.
    assert sum(1 for k in got if k[0] == "Consistency") == 3


# ------------------------------------------------------------------ #
#  Tooltips, words, _BANNED
# ------------------------------------------------------------------ #

def test_brief_keeps_about_120_words_plain_first_experts_last():
    long = "\n\n".join(["Plain words. " * 30, "More plain words here. " * 20,
                        wording.for_experts("Term one; term two. " * 10)])
    out = wording.brief(long)
    assert _words(out) <= wording.TOOLTIP_WORDS
    assert out.startswith("Plain words.")
    assert out.split("\n\n")[-1].startswith("For experts:")


def test_the_new_rules_fit():
    for text in (wording.CALLED_RULE, wording.LEARNED_RULE,
                 wording.story_path_help(), wording.story_target_help(3, 21, 1),
                 wording.profiler_help("Best so far", 21, gain=True),
                 wording.ceiling_help("99.2", "100", "2.3", ["a", "b", "c"]),
                 wording.story_improving_help("84.2", "96.7", 1, 10)):
        assert _words(text) <= wording.TOOLTIP_WORDS, text
    assert "(as in the chart at the top)" not in wording.profiler_help("Best so far", 21)
    assert "Lowest to its Highest" not in wording.profiler_help("Best so far", 21)
    for caption in (wording.CALLED_CAPTION, wording.LEARNED_CAPTION,
                    wording.STORY_TARGET_CAPTION, wording.STORY_PATH_CAPTION,
                    wording.STORY_IMPROVING_CAPTION, wording.PROFILER_CAPTION):
        assert _words(caption) <= 10, caption
    assert _words(wording.answer_line(wording.CEILING_HEAD, wording.CEILING_TAIL)) <= 14
    assert wording.TRACK_VALUES_MISS == "Measurements: more misses than expected."
    # The tried gaps (owner, 2026-10-05): the count first, then the largest.
    assert wording.tried_gaps(7, 7, "ingredient", wording.gap_span(
        "Salt", "0.6", "1.0 g")) == (
        "Gaps in all 7 ingredients; largest: Salt 0.6–1.0 g")


def _critique_strings():
    """Every string literal in wording.py's critique-fixes block."""
    text = pathlib.Path(wording.__file__).read_text()
    block = text[text.index("# --- critique fixes ---"):]
    out = []
    for node in ast.walk(ast.parse(block)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append("".join(v.value for v in node.values if isinstance(v, ast.Constant)))
    return out


def test_critique_strings_pass_banned():
    texts = [t for t in _critique_strings() if " " in t and not t.startswith(("'", '"'))]
    texts = [t for t in texts if "\n" not in t[:1]]     # docstrings start with a word too
    bad = [t for t in texts if any(p.search(t) for p in _BANNED)
           and not t.lstrip().startswith(("Every", "The", "A ", "'", "Review"))]
    assert bad == []
    for t in (wording.CALLED_RULE, wording.LEARNED_RULE, wording.CEILING_HEAD,
              wording.CEILING_TAIL, wording.CARD_CEILING, wording.drives_limited("Formulation total"),
              wording.target_moves(2, 1, [], ["0.5"]), wording.MAP_Y_TITLE,
              wording.MAP_GAIN_LEGEND, wording.PATH_AT_LIMIT, wording.PATH_GAP,
              wording.widen_all(["a", "b"]), wording.REPEAT_CONFIRMS, wording.NOISE_BAND,
              wording.implied_line("90", "94"), wording.RESULTS_LANDED_COLUMN):
        assert not any(p.search(t) for p in _BANNED), t
        assert "spread" not in t.lower() and "interval" not in t.lower()


def test_show_as_lines_is_gone():
    assert not hasattr(wording, "STORY_SHOW_LINES")
    assert not hasattr(U, "tried_chart") and not hasattr(U, "PATH_SHOW_LINES_KEY")


# ------------------------------------------------------------------ #
#  Page order (a synthetic project through the app)
# ------------------------------------------------------------------ #

def test_section_order_constant():
    assert U.SECTION_ORDER == ("progress", "next", "predictions", "tried", "drives")


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    from tests import demo_projects as D
    from tests.test_review_cards import _install, _open
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-critique"))
    try:
        with bo_views.fresh():
            _install(D.LINE_A)
            yield _open(D.LINE_A), D.load(D.LINE_A)
    finally:
        os.chdir(here)


def test_the_page_reads_in_the_ruled_order(line_a):
    from tests.test_review_cards import _leaves, _text
    at, opt = line_a
    say = lambda t: wording.for_project(opt, t)  # noqa: E731
    seen = []
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            seen.append(text[5:])
        elif el.type == "markdown" and text == f"**{say(wording.FORECAST_CHECK_TITLE)}**":
            seen.append("Forecast check")
        elif el.type == "expander" and not folds:
            seen.append(el.label)
    heads = [say(wording.Q2_TITLE), say(wording.Q5_TITLE)]
    assert seen[:2] == heads
    rest = seen[2:]
    # Slim Review (re-pinned): no model section of its own; the Forecast
    # check holds its verdict line and, in a closed fold, its chart; then
    # What have we tried? and the profiler, both closed.
    model = say(wording.FORECAST_CHECK_TITLE)
    assert say(wording.LEARNED_TITLE) not in rest
    assert rest[:2] == [model, say(wording.LEARNED_DETAILS_TITLE)]
    assert rest.index(say(wording.Q4_TITLE)) > rest.index(model)
    assert rest.index(say(wording.Q3_TITLE)) == rest.index(say(wording.Q4_TITLE)) + 1
    assert rest[-1] == say(wording.FOLD_METHOD)
    # No toggle left on the tab.
    assert not [t for t in at.tabs[3].toggle]


def test_review_tooltips_are_about_120_words(line_a):
    from tests.test_review_cards import _leaves
    at, _ = line_a
    long = []
    for el, folds, _ in _leaves(at.tabs[3]):
        if folds:       # Method and definitions keeps the whole text
            continue
        help_ = getattr(getattr(el, "proto", None), "help", "") or ""
        if help_ and _words(help_) > wording.TOOLTIP_WORDS:
            long.append((_words(help_), help_[:80]))
    assert long == []


# ------------------------------------------------------------------ #
#  Follow-up (owner): the stop signal after a rise too
# ------------------------------------------------------------------ #

def test_the_ceiling_stop_fires_on_a_rising_round(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    opt = NS(pending_batch=[], pending_batch_no=None, next_batch_no=lambda: 7,
             utility_ceiling=lambda: 100.0)
    facts = NS(next=insights.NextStep("rising", assumed=False),
               best=NS(score=98.3, formulation=16), yardstick=_yard())
    assert U.ceiling_stop(opt, facts)
    card = U.next_card(opt, facts)
    assert card.word == wording.CARD_CEILING and card.lines == []
    # The "?" says what a further Round can and cannot show.
    tip = U.ceiling_tip(opt, facts)
    assert wording.CEILING_MEASURE in tip and _words(wording.brief(tip)) <= wording.TOOLTIP_WORDS
    # Round 2: what one new result can and cannot show, said exactly.
    # Sceptic review (2026-10-14), re-pinned: no advice to move a target to
    # chase a measurement; On target says which are beyond their noise.
    assert wording.CEILING_MEASURE == (
        "A single new result can't be told apart from this best in the Overall score; a "
        "single measurement still can (On target says which are beyond their noise).")
    # Below the threshold, Rising keeps Make Round 7.
    facts.best = NS(score=95.0, formulation=10)
    assert not U.ceiling_stop(opt, facts)
    assert U.next_card(opt, facts).word == wording.card_make(7)
    # The misses row: the stop, its Repeat request kept under it.
    facts.best = NS(score=99.2, formulation=21)
    facts.next = insights.NextStep("misses")
    card = U.next_card(opt, facts)
    assert card.word == wording.CARD_CEILING
    assert card.lines == [(wording.card_noise_repeat(21), False)]
    # A disagreeing Repeat (check the panel first), or a Round still open,
    # is never replaced.
    facts.next = insights.NextStep("repeat")
    assert not U.ceiling_stop(opt, facts)
    facts.next = insights.NextStep("rising")
    monkeypatch.setattr(U, "open_rows", lambda o: [{"formulation": 22}])
    opt.pending_batch = [{"formulation": 22}]
    assert not U.ceiling_stop(opt, facts)
    for text in (wording.CEILING_MEASURE, wording.CEILING_SETUP_HELP):
        assert not any(p.search(text) for p in _BANNED), text
