"""Review story (prototype, 2026-10-02): the Review read as a story, in the
order a food scientist asks — Are we improving? Can we trust the
predictions? What drives each result? What have we tried? — the cards in
the same order (Progress · Predictions · Next).

- Are we improving?: two model-free charts side by side — the best so far
  with each Round's average and the start and the best named; Closer to
  target, one row per measurement (the first Round's best hollow, Best so
  far in ink, the target dashed or an arrow at the better end).
- Can we trust the predictions?: forecast against actual over time — each
  checked formulation's Round-ahead 95 % range (light blue) and the score
  measured (filled inside, hollow outside); the answer counts the landings.
- What drives each result?: a finding, not the Next card's word.
- What have we tried?: the search path (one dot plot per row, Rounds down),
  the parallel coordinates on a "Show as lines" toggle.

Review story 2: the cards became one status strip above the charts
(Progress · Predictions · Next, the one primary at its right end) and the
closing section What next? (the Next card's word, lines and secondaries).

A synthetic project (the demo projects) runs in CI; the yogurt pilot's
checks skip without the confidential files, and assert no pilot names."""
import json
import os
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _install, _leaves, _text, cards, section, strip
from tests.test_review_diag import ALL10, PILOT, R6, needs_pilot

# Review critique (fix 5, re-pinned): Are we improving?, What next?, What has
# the model learned? (the Forecast check above it), What have we tried?;
# What drives each result? is a closed fold under them (not a heading).
# Slim Review (owner, 2026-10-08, re-pinned): the Forecast check is the
# model's section (its verdict line in it); What have we tried? is a closed
# fold too, so the headings are three.
STORY = [wording.Q2_TITLE, wording.Q5_TITLE, wording.FORECAST_CHECK_TITLE]
CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def sections(at):
    """{question: [(type, text)]} in screen order."""
    out, cur = {}, None
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            cur = text[5:]
            out[cur] = []
            continue
        if el.type == "expander":
            # Slim Review: What have we tried? is a fold; the Forecast
            # check's details fold stays in its section.
            if el.label == wording.LEARNED_DETAILS_TITLE:
                continue
            cur = el.label if el.label == wording.Q4_TITLE and not folds else None
            if cur is not None:
                out[cur] = []
            continue
        if cur is not None and (text or el.type in CHART or el.type == "toggle"):
            out[cur].append((el.type, text or getattr(el, "label", "")))
    return out


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-story"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (D.LINE_A, D.SNACK, D.PROCESS_ONLY):
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  The order: a story
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_sections_and_cards_in_story_order(screens, name):
    at = screens[name]
    opt = D.load(name)
    assert list(sections(at)) == [wording.for_project(opt, q) for q in STORY] + [
        wording.for_project(opt, wording.Q4_TITLE)]
    # The cards in the same order, Next last.
    labels = [c["label"] for c in cards(at, order="screen")]
    # Review learned: Model once the model's verdicts are in (the gate
    # keeps Predictions).
    model = wording.CARD_PREDICTIONS if name == D.SNACK else wording.LEARNED_STRIP_LABEL
    assert labels == [wording.CARD_PROGRESS, model, wording.CARD_NEXT]
    assert U.SECTION_ORDER == ("progress", "next", "predictions", "tried", "drives")
    assert U._STATUS_ORDER == ("progress", "predictions", "next")


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_improving_has_two_charts_and_tried_has_its_toggle(screens, name):
    got = sections(screens[name])
    opt = D.load(name)
    improving = got[wording.for_project(opt, wording.Q2_TITLE)]
    assert sum(t in CHART for t, _ in improving) == 2
    # Re-pinned (sensory profile, 2026-10-08): from U.SPIDER_MIN measurements
    # Start vs best's dot chart is the spider, under its own title.
    title = (wording.SPIDER_TITLE if len(opt.objectives) >= U.SPIDER_MIN
             else wording.STORY_TARGET_TITLE)
    assert ("markdown", f"**{wording.for_project(opt, title)}**") in improving
    tried = got[wording.for_project(opt, wording.Q4_TITLE)]
    # Slim Review (re-pinned): a closed fold, worked out only once opened.
    assert sum(t in CHART for t, _ in tried) == 0
    # Review critique (fix 5): no "Show as lines" toggle any more.
    assert not any(t == "toggle" for t, _ in tried)


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_word_limits(screens, name):
    at = screens[name]
    for q, items in sections(at).items():
        assert _words(q) <= 5
        firsts = [t for k, t in items if k == "markdown" and not t.startswith("**Closer")]
        if firsts:
            # Review critique: a headline up to 14 words (a rise says its
            # assumed noise: "needs 3.2 if tasting noise ≈ 1 point").
            assert _words(firsts[0]) <= 14, firsts[0]
        for kind, text in items:
            if kind == "caption" and text != wording.for_project(D.load(name), wording.Q_WAIT):
                assert _words(text) <= 10, text
            if kind in ("button", "toggle"):
                assert _words(text) <= 5, text
    for c in cards(at):
        for line in c["lines"]:
            assert _words(line) <= 10, line


# ------------------------------------------------------------------ #
#  Are we improving?
# ------------------------------------------------------------------ #

def test_the_start_is_the_first_rounds_best():
    opt = D.load(D.LINE_A)
    i, r = U.story_start(opt)
    first = [k for k, b in enumerate(opt.batch_history) if b == r]
    assert r == min(b for b in opt.batch_history if b is not None)
    assert opt.Y_history[i] == max(opt.Y_history[k] for k in first if k in
                                   set(opt.best_candidates()))
    s, r0, b, rb, bno = U.story_numbers(opt)
    assert (s, r0) == (float(opt.Y_history[i]), r)
    assert b == float(opt.Y_history[opt.best_index()]) and bno == int(
        opt.formulation_ids[opt.best_index()])


def test_round_averages_are_the_rounds_means():
    opt = D.load(D.LINE_A)
    pool = set(opt.best_candidates())
    for r, a, b, mean, n in U.round_averages(opt):
        mine = [float(opt.Y_history[i]) for i, x in enumerate(opt.batch_history)
                if x == r and i in pool]
        assert n == len(mine) and mean == pytest.approx(sum(mine) / len(mine))


def test_improving_chart_names_the_start_and_the_best():
    opt = D.load(D.LINE_A)
    spec = json.dumps(U.improving_chart(opt).to_dict())
    s, r0, b, rb, _ = U.story_numbers(opt)
    assert wording.story_point(wording.STORY_START, f"{s:.1f}", r0) in spec
    assert wording.story_point(wording.STORY_BEST, f"{b:.1f}", rb) in spec
    assert wording.STORY_ROUND_AVERAGE in spec and wording.Q2_X_TITLE in spec
    assert "'R1'" in spec                       # the Rounds marked along the top


def test_closer_to_target_rows_targets_and_directions():
    opt = D.load(D.LINE_A)
    rows = U.target_rows(opt)
    assert [r["name"] for r in rows] == [o["name"] for o in opt.measurements_by_importance()]
    s, b = U.story_start(opt)[0], opt.best_index()
    for r in rows:
        obj = next(o for o in opt.objectives if o["name"] == r["name"])
        assert r["goal"] == obj["goal"]
        assert r["target"] == (obj["target"] if obj["goal"] == "target" else None)
        assert (r["low"], r["high"]) == (obj["min_val"], obj["max_val"])
        assert r["start"] == opt.results_history[s].get(r["name"])
        assert r["best"] == opt.results_history[b].get(r["name"])
    spec = U.target_chart(opt).to_dict()
    text = json.dumps(spec, ensure_ascii=False)
    # Line A: two targets and one "lower is better" (Cook loss): its arrow
    # points to the low end; a label per row says which.
    goals = {r["name"]: r["goal"] for r in rows}
    assert "min" in goals.values() and "target" in goals.values()
    for r in rows:
        assert wording.story_target_row(r["name"], r["goal"], r["target"]) in text
    assert "triangle-left" in text
    # Review critique (fix 6): the target a large ink diamond (was a dashed tick).
    assert '"shape": "diamond"' in text
    # Higher is better: an arrow at the high end.
    obj = next(o for o in opt.objectives if o["goal"] == "min")
    obj["goal"] = "max"
    assert "triangle-right" in json.dumps(U.target_chart(opt).to_dict())
    assert wording.story_target_row("Firmness", "max") == "Firmness · higher"
    assert wording.story_target_row("Cook loss", "min") == "Cook loss · lower"
    assert wording.story_target_row("Juiciness", "target", 7.0) == "Juiciness · target 7"


def test_the_progress_card_goes_from_start_to_best():
    assert wording.card_progress("87.8", "90.5", "same for 2 Rounds") == \
        "87.8 → 90.5 · same for 2 Rounds"
    assert wording.card_progress(None, "96.0", "Formulation 1") == "Best 96.0 · Formulation 1"
    # Review story 2: "of 100" on a 0–100 score only.
    assert wording.score_of("96.4", 100.0) == "96.4 of 100"
    assert wording.score_of("9.2", 10.0) == "9.2"
    assert wording.card_progress("88.4", wording.score_of("96.4", 100), "same for 3 Rounds") == \
        "88.4 → 96.4 of 100 · same for 3 Rounds"


# ------------------------------------------------------------------ #
#  Can we trust the predictions?
# ------------------------------------------------------------------ #

def _checks():
    S = insights.SCORE
    return [insights.Check(7, 3, S, 85.0, 80.0, 90.0, 86.0, True, True),
            insights.Check(8, 3, S, 85.0, 80.0, 90.0, 93.0, False, True),
            insights.Check(9, 3, S, 85.0, 80.0, 90.0, 77.0, False, True),
            insights.Check(10, 4, S, 86.0, 82.0, 90.0, 88.0, True, True),
            insights.Check(10, 4, "Firmness", 6.0, 5.0, 7.0, 6.2, True, True)]


def test_forecast_chart_spec():
    opt = D.load(D.LINE_A)
    chart = U.forecast_chart(opt, _checks())
    spec = chart.to_dict()
    text = json.dumps(spec, ensure_ascii=False)   # round 2: "95 %" has a non-breaking space
    rows = [r for v in spec["datasets"].values() for r in v if "measured" in r]
    # The Overall score's checks only, one row per formulation checked.
    assert sorted(r[wording.FORMULATION_CAP] for r in rows) == [7, 8, 9, 10]
    assert {r["state"] for r in rows} == {wording.STORY_LANDED_IN, wording.STORY_LANDED_OUT}
    # The model blue bar, filled ink inside, orange outside (round 2: a
    # miss in the one accent colour, as on the Forecast check); no red.
    assert U.model_blue() in text and U.outline_colour() in text
    assert U.mark("bar_latest")[0] in text
    assert "#d62728" not in text.lower() and '"red"' not in text
    for w in (wording.STORY_FORECAST, wording.STORY_LANDED_IN, wording.STORY_LANDED_OUT,
              wording.Q2_X_TITLE, wording.STORY_NO_FORECAST):
        assert w in text, w
    # The same x axis as the improving chart.
    assert U._x_domain(opt) == [min(map(int, opt.formulation_ids)) - 0.7,
                                max(map(int, opt.formulation_ids)) + 0.7]
    assert json.dumps(U._x_domain(opt)) in text
    assert json.dumps(U._x_domain(opt)) in json.dumps(U.improving_chart(opt).to_dict())
    assert U.forecast_chart(opt, []) is None


def test_trust_counts_and_lines():
    assert U.trust_counts(_checks()) == (2, 4, 1, 1)
    assert wording.story_trust_answer(13, 15) == "13 of 15 landed in their predicted range"
    assert wording.story_misses(2, 0) == "Both misses scored higher than predicted."
    assert wording.story_misses(0, 1) == "The miss scored lower than predicted."
    assert wording.story_misses(1, 1) == "1 scored higher than predicted, 1 lower."
    assert wording.story_misses(0, 0) == ""
    assert "BO term" in wording.story_trust_help(13, 15)


def test_trust_answer_on_line_a(screens):
    # Review learned: the trust evidence is what the model learned.
    # Slim Review (re-pinned): the verdicts are one line in the Forecast
    # check; beside the Overall score's pair count they would run past 14
    # words, so the groups alone.
    line = next(t for k, t in sections(screens[D.LINE_A])[wording.FORECAST_CHECK_TITLE]
                if k == "markdown" and re.match(r"(So far|Predicts|Partly|Not yet)", t))
    # Persona fixes 2: pairs under one tasting noise apart left out.
    # Review critique: the exact permutation test (and each measurement's
    # share of the noise): they lean right, not clearly.
    # Specific wording (2026-10-12), re-pinned: each name with its pairs.
    assert line == ("Partly: Overall score (5 of 5 pairs), Firmness (12/16), "
                    "Cook loss (5/5).")


# ------------------------------------------------------------------ #
#  What drives each result?
# ------------------------------------------------------------------ #

class _Col:
    def __init__(self, name, effect, end=None):
        self.name, self.effect, self.near_effect, self.end = name, effect, effect, end


class _Prof:
    def __init__(self, cols, ref="Best so far"):
        self.columns, self.ref_label = cols, ref


def test_the_finding_line():
    fine, noise = insights.Guard([]), insights.Guard(["noise"])
    p = _Prof([_Col("Oat flour", 9.0, "Highest"), _Col("Salt", 2.0)])
    assert U.driver_finding(None, p, fine) == \
        "**Oat flour matters most.** Best so far uses its Highest."
    p = _Prof([_Col("Salt", 4.0), _Col("Pea protein", 2.0)], ref="Suggestion 1")
    assert U.driver_finding(None, p, fine) == "**Salt matters most.** Then Pea protein."
    assert U.driver_finding(None, _Prof([_Col("Salt", 4.0)]), fine) == "**Salt matters most.**"
    for p, g in ((_Prof([_Col("Salt", 4.0)]), noise), (_Prof([_Col("Salt", 0.0)]), fine),
                 (None, None)):
        # Review story 3: one line for every no-leader case.
        assert U.driver_finding(None, p, g) == f"**{wording.story_no_leader()}.**"
    for line in ("**Oat flour matters most.** Best so far uses its Highest.",
                 "**Textured soy protein matters most.** Then Methylcellulose."):
        assert _words(line) <= 10


def test_the_finding_is_not_the_next_cards_word(screens):
    # Review critique (fix 4): Q3 is a closed fold, worked out once opened:
    # nothing of it (and no Next word) on the closed screen.
    at = screens[D.LINE_A]
    assert wording.Q3_TITLE not in sections(at)
    assert [e for e in at.tabs[3].expander if e.label == wording.Q3_TITLE]


def test_the_profiler_defaults_to_four_columns():
    assert U.PROFILE_COLUMNS == 4
    # Review readable: at most PROFILE_WRAP (4) panels a row; more columns wrap
    # into a second block, so 7 columns draw at the 4-column width.
    assert U.panel_width(3) > U.panel_width(4) == U.panel_width(7) == U._PANEL_W


# ------------------------------------------------------------------ #
#  What have we tried? — the search path
# ------------------------------------------------------------------ #

def test_search_path_panels():
    opt = D.load(D.LINE_A)
    axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
    chart = U.search_path_chart(opt, axes, lines)
    spec = chart.to_dict()
    assert len(spec["concat"]) == len(axes) and spec["columns"] == U.PATH_COLUMNS
    titles = [p["title"]["text"] for p in spec["concat"]]
    assert titles == [U._axis_label(a) for a in axes]
    df = U.path_frame(opt, axes, lines)
    # One dot per formulation per row; the open Round's suggestions in its row.
    assert set(df["role"]) >= {wording.STORY_PATH_TESTED, wording.TRIED_BEST,
                               wording.TRIED_SUGGESTED}
    sugg = df[df["role"] == wording.TRIED_SUGGESTED]
    assert set(sugg["round"]) == {opt.pending_batch_no}
    best = df[df["role"] == wording.TRIED_BEST]
    assert set(best["no"]) == {int(opt.formulation_ids[opt.best_index()])}
    # Each continuous panel spans its allowed Lowest–Highest.
    text = json.dumps(spec)
    for a in axes:
        if a.options is None:
            assert json.dumps([a.low, a.high]) in text
    # Palette: ink Best so far, orange suggestions, greys; no score ramp.
    for colour in (U.mark("bar_latest")[0], U.outline_colour(), U._TRIED_GREYS[U.theme()][1]):
        assert colour in text
    assert not any(c in text for c in U.score_colours())
    # Rounds down, the first at the top.
    rounds = sorted(set(df["round"]))
    assert spec["concat"][0]["layer"][1]["encoding"]["y"]["scale"]["domain"] == \
        [wording.round_tick(r) for r in rounds]


def test_search_path_on_settings_only_and_empty():
    opt = D.load(D.PROCESS_ONLY)
    assert U.search_path_chart(opt, insights.tried_axes(opt), insights.tried_lines(opt))
    assert "trial" in wording.for_project(opt, wording.STORY_PATH_CAPTION)
    assert U.search_path_chart(opt, [], []) is None


# ------------------------------------------------------------------ #
#  Vocabulary and word limits
# ------------------------------------------------------------------ #

def _story_strings():
    return [wording.card_progress("87.8", "90.5", "same for 2 Rounds"),
            wording.card_checked_so_far(7, 7, 10), wording.STORY_START, wording.STORY_BEST,
            wording.story_point("Start", "87.8", 1), wording.STORY_ROUND_AVERAGE,
            wording.STORY_EACH, wording.STORY_IMPROVING_CAPTION,
            wording.story_improving_help("83.8", "89.5", 1, 5),
            wording.story_target_start(1), wording.STORY_TARGET_BEST, wording.STORY_TARGET,
            wording.STORY_TARGET_TITLE, wording.STORY_TARGET_CAPTION,
            wording.story_target_row("Juiciness", "target", 7), wording.story_target_help(3, 21, 1),
            wording.STORY_FORECAST, wording.STORY_LANDED_IN, wording.STORY_LANDED_OUT,
            wording.STORY_NO_FORECAST, wording.STORY_TRUST_CAPTION,
            wording.story_trust_answer(13, 15), wording.story_trust_so_far(7, 7, 10),
            wording.story_misses(2, 0), wording.story_trust_help(13, 15, "x"),
            wording.story_driver("Salt"), wording.story_driver_end("Best so far", "Highest"),
            wording.story_driver_then("Salt"), wording.STORY_NO_DRIVER,
            wording.story_driver_help("Best so far"), wording.STORY_PATH_CAPTION,
            wording.STORY_PATH_TESTED,
            wording.story_path_help(), wording.story_path_help(True), wording.STORY_VALUE]


def test_story_strings_pass_the_vocabulary_guard():
    for text in _story_strings():
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
        # No claim of Rounds or time saved, nor any comparison with another method.
        assert not re.search(r"\bsaved?\b|faster|fewer rounds|than (DoE|random|manual)",
                             text, re.I), text
    # Labels at most 5 words, captions at most 8, status lines at most 10.
    for label in (wording.STORY_ROUND_AVERAGE, wording.story_target_start(1),
                  wording.STORY_TARGET_TITLE, wording.STORY_FORECAST, wording.STORY_LANDED_IN,
                  wording.STORY_LANDED_OUT, wording.STORY_NO_FORECAST,
                  wording.STORY_PATH_TESTED, wording.story_target_row("Textured pea protein",
                                                                      "target", 13)):
        assert _words(label) <= 5, label
    for caption in (wording.STORY_IMPROVING_CAPTION, wording.STORY_TARGET_CAPTION,
                    wording.STORY_TRUST_CAPTION, wording.STORY_PATH_CAPTION):
        assert _words(caption) <= 8, caption
    for status in (wording.card_progress("87.8", "90.5", "+2.5 (beat 2.0)"),
                   wording.card_checked_so_far(7, 7, 10), wording.story_trust_answer(13, 15),
                   wording.story_trust_so_far(7, 7, 10)):
        assert _words(status) <= 10, status
    # The section sits under its own heading in wording.py.
    src = open(wording.__file__).read()
    assert "# --- review story ---" in src
    assert src.index("# --- review story ---") < src.index("def card_progress(")


# ------------------------------------------------------------------ #
#  The yogurt pilot (confidential: skips without it)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-story-pilot")
    for n in (R6, ALL10):
        shutil.copy(PILOT / f"{n}.pkl", work)
    os.chdir(work)
    try:
        with bo_views.fresh():
            yield work
    finally:
        os.chdir(here)


@needs_pilot
def test_pilot_story(pilot):
    opt = FoodOptimizer(ALL10)
    s, r0, b, rb, _ = U.story_numbers(opt)
    assert r0 == 1 and b > s and rb > r0
    avgs = U.round_averages(opt)
    assert avgs[-1][3] > avgs[0][3]                 # the Round average rose too
    rows = U.target_rows(opt)
    assert all(r["goal"] == "target" for r in rows)
    # The best is closer to every target than the start, or as close.
    assert sum(abs(r["best"] - r["target"]) for r in rows) < \
        sum(abs(r["start"] - r["target"]) for r in rows)
    at = _open(ALL10)
    got = sections(at)
    # Round 2 (P2 #9): the Forecast check is its own section after What next?.
    assert list(got) == STORY + [wording.Q4_TITLE]       # slim Review
    line = next(t for k, t in got[wording.Q2_TITLE] if k == "markdown")
    # Review story 2: "of 100", so Flat at 96.4 reads as near the ceiling.
    # Persona fixes 2: Flat after a rise reads "Plateau since Round N".
    # Round 2: the plateau line names its best.
    assert re.fullmatch(r"\*\*Plateau since Round \d+\.\*\* \d+\.\d → \d+\.\d of 100 · "
                        r"Formulation \d+\.", line), line
    items, _ = strip(at)
    # The strip, by pattern only (no pilot numbers in the repository).
    # Review learned: the Model item (by pattern only: no pilot names).
    assert re.fullmatch(r"Plateau since Round \d+", items[0]["word"]) and \
        items[1]["word"] == "Predicts:"
    # Persona fixes: 'Flat · best 99.2 of 100' (the start stays on the chart).
    assert re.fullmatch(r"· best \d+\.\d of 100", items[0]["line"]), items[0]
    # Specific wording (2026-10-12), re-pinned: each name with its pairs.
    assert re.fullmatch(r"[A-Z][\w ]+ \((\d+ of \d+ pairs|\d+/\d+)\)"
                        r"(, [A-Z][\w ]+ \(\d+/\d+\))?( \+\d+)?", items[1]["line"]), items[1]
    # Slim Review (re-pinned): the verdict line in the Forecast check.
    line = next(t for k, t in got[wording.FORECAST_CHECK_TITLE]
                if k == "markdown" and t.startswith("So far: "))
    assert re.fullmatch(r"So far: Overall score \d+ of \d+ pairs ordered right\. "
                        r"Predicts: .+ \(\d+/\d+\)\.", line), line
    # Review critique (fix 4): the profiler is a closed fold.
    assert wording.Q3_TITLE not in got
    before = cards(_open(R6), order="screen")
    assert [c["word"] for c in before[:2]] == ["No new best", "Partly:"]


# ------------------------------------------------------------------ #
#  Review story 2: the strip above the charts, What next? at the end
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_the_strip_has_three_items_and_the_one_primary(screens, name):
    at = screens[name]
    items, buttons = strip(at)
    # Review learned: the Predictions item reads Model once the model's
    # verdicts are in (the gate keeps Predictions); its key is unchanged.
    model = wording.CARD_PREDICTIONS if name == D.SNACK else wording.LEARNED_STRIP_LABEL
    assert [i["label"] for i in items] == [wording.CARD_PROGRESS, model, wording.CARD_NEXT]
    keys = {wording.LEARNED_STRIP_LABEL: "predictions"}
    for i in items:
        # Review critique (fix 13): about 120 words, plain first.
        assert i["word"] and i["help"] and len(i["help"].split()) <= wording.TOOLTIP_WORDS, i
        assert i["key"] == f"{U.SEE_CHART_KEY}_{keys.get(i['label'], i['label'].lower())}"
    # The page's one primary sits at the strip's right end, and nowhere else.
    assert [t for t, _ in buttons] == ["primary"], buttons
    primaries = [el.label for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert primaries == [buttons[0][1]]


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_no_chart_above_the_first_section_and_what_next_last(screens, name):
    at = screens[name]
    heads = [_text(el)[5:] for el, _, _ in _leaves(at.tabs[3])
             if el.type == "markdown" and _text(el).startswith("#### ")]
    # Review critique (fix 5): What next? straight after Are we improving?.
    assert heads[:2] == [wording.Q2_TITLE, wording.Q5_TITLE]
    for el, _, _ in _leaves(at.tabs[3]):
        if el.type == "markdown" and _text(el).startswith("#### "):
            break
        assert el.type not in CHART and el.type != "subheader", el.type
    # What next? holds the Next word with its "?" and no primary.
    rows = section(at, wording.Q5_TITLE)
    # Persona fixes 2: the Next word is a small answer line, not a subheader.
    heads = [el for el, _ in rows if el.type == "subheader" or (
        el.type == "markdown" and re.fullmatch(r"(:red\[)?\*\*[^*]+\.\*\*\]?", _text(el)))]
    assert len(heads) == 1 and heads[0].proto.help
    assert not [el for el, _ in rows if el.type == "button" and el.proto.type == "primary"]


def test_what_next_holds_the_secondary_actions(screens):
    at = screens[D.LINE_A]
    nxt = cards(at)[2]
    assert nxt["next_word"] == nxt["word"] == "Record Round 6"
    assert nxt["next_help"] == nxt["help"]
    assert "Repeat Formulation 15 to check noise" in nxt["lines"]   # round 2: one form
    # The Set up secondary and the one link sit here, never in the strip.
    keys = [el.key for el, _ in section(at, wording.Q5_TITLE) if el.type == "button"]
    assert keys and set(keys) <= {"insights_change_setup", "insights_why_link",
                                  U.REPEAT_LINK_KEY, "insights_explored_link"}, keys
    _, buttons = strip(at)
    assert all(t == "primary" for t, _ in buttons)


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_strip_word_limits(screens, name):
    items, _ = strip(screens[name])
    for i in items:
        assert _words(i["label"]) <= 5, i
        assert _words(f"{i['word']} {i['line']}") <= 10, i
    for el, text in section(screens[name], wording.Q5_TITLE):
        if el.type in ("markdown", "subheader"):
            assert _words(text) <= 10, text
        if el.type == "button":
            assert _words(text) <= 5, text


def test_strip_helpers():
    class _C:
        word, tone, lines = "Flat", None, [("x", False)]
    assert U.strip_label(_C, "88.4 → 96.4 of 100", False) == "**Flat** 88.4 → 96.4 of 100"
    _C.word, _C.tone = "Check panel first", "off"
    assert U.strip_label(_C, "", False) == ":red[**Check panel first**]"
    _C.word, _C.tone = "Not yet", None
    assert U.strip_label(_C, "3 / 16 missed", True) == "**Not yet** :red[**3 / 16 missed**]"
    assert wording.strip_short("Overall score: 13 / 15 in range") == "13 / 15 in range"
    assert wording.strip_progress(None, "96.0 of 100") == "Best 96.0 of 100"
    for text in (wording.Q5_TITLE, wording.SEE_CHART_HELP, wording.EV_METHOD,
                 wording.REVIEW_STATUS_HELP, wording.GAIN_REASON_NOT_YET,
                 wording.GAIN_REASON_OFF, wording.tip_hero_hidden("x"),
                 wording.tip_gain_greyed("x"), wording.strip_jump_help(wording.Q2_TITLE)):
        assert not any(p.search(text) for p in _BANNED), text
        assert "card" not in text.lower(), text
    src = open(wording.__file__).read()
    assert src.index("# --- review story ---") < src.index("# --- review story 2 ---") < \
        src.index("def strip_progress(")
