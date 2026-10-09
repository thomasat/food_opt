"""Slim Review (owner, 2026-10-08): four changes to the Review tab.

1. One Round slider at the top of "Are we improving?" drives all three
   parts: the score chart (the chosen Round shaded, a marker on Best so far
   there; every later Round still drawn), the sensory profile and What
   changed (Best so far by that Round; its limits line judged on the
   formulations recorded by then). The section's answer line stays about
   the latest Round; a "Showing Round N" cue sits by the slider. One
   fragment. (Layout study, 2026-10-09: the slider is now the Replay
   panel's ◀ Round N of M ▶ control; test_review_layout.)
2. "What has the model learned?" is no section of its own: one verdict
   line in the Forecast check, merged with its "So far:" count, and the
   ordering chart in a closed fold, "Details: ordering by result".
3. "What have we tried?" is a closed fold after the Forecast check, worked
   out only while open.
4. "What drives each result?" is left out when more than half of the varied
   rows have too little room to move (the profiler's own ROOM_MIN rule).

Synthetic projects only."""
import inspect
import itertools
import json
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import learned
import insights
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _leaves, _text

_n = itertools.count(1)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _words(text):
    """Every word, numbers too (the owner's limits count them)."""
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _banned(text):
    return [p.pattern for p in _BANNED if p.search(text)]


# Four measurements on 0–15 (targets 12 / 5 / 5 / 13), four Rounds of three.
# Best so far: Round 1 F2, Round 2 F4, Round 3 F7 (F7 and F8 tie; F7 made
# first), Round 4 still F7.
MEASURES = [("Firmness", 12), ("Sweetness", 5), ("Sourness", 5), ("Smoothness", 13)]
ROWS = [  # (Round, number, Firmness, Sweetness, Sourness, Smoothness)
    (1, 1, 5, 3, 9, 12), (1, 2, 8, 4, 5, 13), (1, 3, 6, 4, 8, 13),
    (2, 4, 10, 4.5, 6, 12), (2, 5, 7, 5, 7, 11), (2, 6, 9, 3, 8, 12),
    (3, 7, 12, 5, 4, 13), (3, 8, 12, 5, 6, 13), (3, 9, 11, 6, 6, 12),
    (4, 10, 11, 5, 7, 12),
]


def _spider_opt(rows=ROWS):
    opt = FoodOptimizer(f"slim_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Starch", 0, 5)
    for name, target in MEASURES:
        opt.add_objective(name, 25, goal="target", target=target, min_val=0, max_val=15)
    for k, (r, no, *vals) in enumerate(rows):
        opt.tell({"Oil": 2.0 + k, "Starch": 1.0 + 0.1 * k},
                 {name: v for (name, _t), v in zip(MEASURES, vals)},
                 formulation_no=no, batch_no=r)
    return opt


def _score(opt, no):
    return float(opt.Y_history[[int(f) for f in opt.formulation_ids].index(no)])


def _review(opt):
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _specs(at):
    return [json.loads(e.proto.spec) for e in at.get("vega_lite_chart")]


def _band_and_marker(spec):
    """(band x, x2) and the marker's data, or (None, None)."""
    band = marker = None
    for layer in U._plot_layers(spec):
        mark = layer.get("mark") or {}
        if "values" not in (layer.get("data") or {}):
            continue        # the highlight's layers carry their data inline
        if mark.get("type") == "rect" and "y" not in layer.get("encoding", {}):
            v = layer["data"]["values"][0]
            band = (v["x"], v["x2"])
        if mark.get("shape") == "diamond":
            marker = layer["data"]["values"][0]
    return band, marker


# ------------------------------------------------------------------ #
#  1 · One Round slider for the section
# ------------------------------------------------------------------ #

def _step(at, key):
    """Click a Replay control (by key) and rerun, the Review kept open
    (AppTest reruns the whole script and drops the tab widget's state; the
    browser reruns only the fragment)."""
    at.button(key=key).click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return [m.value for m in at.tabs[3].markdown]


def _label(r, last):
    return wording.replay_label(r, last)


def test_one_control_drives_the_chart_the_profile_and_what_changed():
    # Layout study (re-pinned): the slider is now the Replay panel's
    # control — ◀ Round N of M ▶, Replay Rounds — over all three parts.
    opt = _spider_opt()
    at = _review(opt)
    assert not [s for s in at.tabs[3].select_slider if s.label == wording.ROUND_CAP]
    keys = {b.key for b in at.tabs[3].button}
    assert {"insights_replay_prev", "insights_replay_next", "insights_replay_play"} <= keys
    assert at.button(key="insights_replay_next").disabled       # on the latest
    assert not at.button(key="insights_replay_prev").disabled
    texts = [m.value for m in at.tabs[3].markdown]
    answer = next(t for t in texts if t.startswith("**") and "→" in t)
    assert _label(4, 4) in texts
    assert "**What changed by Round 4**" in texts
    # The control sits above all three parts, under the section's answer line.
    k = texts.index(_label(4, 4))
    assert texts.index(answer) < k < texts.index("**What changed by Round 4**") \
        < texts.index(f"**{wording.SPIDER_TITLE}**")
    # ◀ twice: Round 2, every part follows.
    _step(at, "insights_replay_prev")
    texts = _step(at, "insights_replay_prev")
    assert _label(2, 4) in texts
    assert any(t.startswith("Round 2: on target") for t in texts)
    assert "**What changed by Round 2**" in texts
    table = next(t for t in texts if t.startswith("| Ingredient"))
    nb = "\u00a0"
    assert f"| Best so far (F4,{nb}Round{nb}2) |" in table and "| Start (F2) |" in table
    # The chart: Round 2's Formulations shaded, the marker on F4 itself.
    band, marker = next((b, m) for b, m in map(_band_and_marker, _specs(at)) if b)
    assert band == (3.5, 6.5)
    assert marker["v"] == pytest.approx(_score(opt, 4)) and marker["f"] == "F4"
    assert marker["x"] == 4
    # The section's answer line is about the latest Round, whatever the panel.
    assert answer in texts


def test_the_section_reruns_as_one_fragment():
    src = inspect.getsource(U)
    assert "@st.fragment\ndef _progress_fragment(" in src
    body = inspect.getsource(U._progress_fragment)
    assert "_replay_bar(" in body and "shown_round(" in body
    for part in ("_improving_part(", "_profile_part(", "_changes_block("):
        assert part in body
    # No slider, no fragment of the profile's own.
    assert "select_slider" not in inspect.getsource(U._replay_bar)
    assert not hasattr(U, "_spider_fragment")
    assert inspect.getsource(U._spider_block).count("select_slider") == 0
    # The answer line is drawn with the cards, outside the fragment.
    assert "progress_answer_text" not in body


def test_one_scored_round_draws_no_control_and_no_highlight():
    opt = _spider_opt([r for r in ROWS if r[0] == 1])
    at = _review(opt)
    assert not [s for s in at.tabs[3].select_slider if s.label == wording.ROUND_CAP]
    assert not [b for b in at.tabs[3].button if (b.key or "").startswith("insights_replay")]
    texts = [m.value for m in at.tabs[3].markdown]
    assert not any(t.startswith(f"**{wording.ROUND_CAP} ") and "** of " in t for t in texts)
    assert all(_band_and_marker(s) == (None, None) for s in _specs(at))


# ------------------------------------------------------------------ #
#  1a · The score chart's highlight
# ------------------------------------------------------------------ #

def test_the_chart_highlights_the_round_and_keeps_every_later_round():
    opt = _spider_opt()
    plain = U.improving_chart(opt).to_dict()
    lit = U.improving_chart(opt, highlight=2).to_dict()
    # Two layers more, nothing removed: the dots of Rounds 3 and 4 stay.
    body = lambda d: d["layer"][1]["layer"]     # noqa: E731 — under the Rounds' axis
    assert len(body(lit)) == len(body(plain)) + 2
    band = body(lit)[0]
    assert band["mark"]["type"] == "rect" and "y" not in band["encoding"]
    data = lit["datasets"][band["data"]["name"]][0]
    assert (data["x"], data["x2"]) == (3.5, 6.5)
    marker = next(l for l in body(lit) if l["mark"].get("shape") == "diamond")
    m = lit["datasets"][marker["data"]["name"]][0]
    assert m["v"] == pytest.approx(_score(opt, 4)) and m["round"] == wording.round_tick(2)
    assert m["x"] == 4                  # on F4's own dot
    dots = next(l for l in body(lit) if l["mark"].get("type") == "point"
                and l["mark"].get("size") == 30)
    assert len(lit["datasets"][dots["data"]["name"]]) == len(ROWS)
    # The kept spec the slider sends draws the same band and marker.
    band2, marker2 = _band_and_marker(U.improving_spec(opt, 2))
    assert band2 == (3.5, 6.5) and marker2["v"] == pytest.approx(m["v"])
    assert _band_and_marker(U.improving_spec(opt, None)) == (None, None)
    # Round 3: the marker on the tie's first-made formulation, F7.
    assert U.round_highlight(opt, 3)[4] == 7
    # Round 4: Best so far is still F7, made in Round 3: the band over
    # Round 4, the marker on F7, outside it.
    a, b, x, v, no = U.round_highlight(opt, 4)
    assert (a, b, x, no) == (9.5, 10.5, 7.0, 7) and v == pytest.approx(_score(opt, 7))
    band4, marker4 = _band_and_marker(U.improving_spec(opt, 4))
    assert band4 == (9.5, 10.5) and marker4["x"] == 7 and marker4["f"] == "F7"


def test_round_bests_follow_best_index_rule():
    opt = _spider_opt()
    rounds, at = U.round_bests(opt)
    assert rounds == [1, 2, 3, 4]
    assert {r: int(opt.formulation_ids[i]) for r, i in at.items()} == {1: 2, 2: 4, 3: 7, 4: 7}
    assert at[4] == opt.best_index()


def test_the_kept_spec_is_private_and_reused():
    opt = _spider_opt()
    U._SPEC_CACHE.clear()
    a = U.improving_spec(opt, None)
    assert a["usermeta"] == {"embedOptions": {"actions": False}}
    assert U.improving_spec(opt, None) is a                 # worked out once
    b = U.improving_spec(opt, 3)
    assert b is not a and len(U._plot_layers(b)) == len(U._plot_layers(a)) + 2
    assert len(U._plot_layers(a)) == len(U._plot_layers(U.improving_spec(opt, None)))


# ------------------------------------------------------------------ #
#  1c · What changed at an earlier Round
# ------------------------------------------------------------------ #

# Start F2 (Round 1's best). Round 2's best F4 puts Starch at 5 g, its
# Highest once the limit is narrowed from 6 to 5; Round 3's F5 was made at
# 5.8 g, beyond it. So "nothing tested beyond" is true by Round 2 and false
# by Round 3.
CHANGE_ROWS = [  # (Round, number, Oil, Sugar, Starch, Firmness)
    (1, 1, 6.0, 1.0, 3.0, 3.0), (1, 2, 10.0, 0.0, 2.0, 5.0), (1, 3, 14.0, 1.5, 1.5, 4.0),
    (2, 4, 12.0, 0.6, 5.0, 6.0), (3, 5, 8.0, 1.0, 5.8, 4.0),
]


def _changes_opt(rows=CHANGE_ROWS):
    opt = FoodOptimizer(f"slim_changes_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Sugar", 0, 2)
    opt.add_ingredient("Starch", 1, 6)
    opt.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    for r, no, oil, sugar, starch, firm in rows:
        opt.tell({"Oil": oil, "Sugar": sugar, "Starch": starch}, {"Firmness": firm},
                 formulation_no=no, batch_no=r)
    opt._var_by_name("Starch")['bounds'] = (1.0, 5.0)
    return opt


def test_what_changed_at_an_earlier_round_judges_limits_up_to_it():
    opt = _changes_opt()
    early = U.what_changed(opt, 2)
    assert (early.start_no, early.best_no, early.round_no) == (2, 4, 2)
    assert early.limits == "Best so far sits at the Highest of Starch (5 g); nothing tested beyond."
    latest = U.what_changed(opt, 3)
    assert latest.best_no == 4
    assert latest.limits == "Best so far sits at the Highest of Starch (5 g)."
    assert "Formulation 5 lies beyond Starch's Highest (5 g)." in latest.limits_help
    # Round 1: Best so far is still the start.
    first = U.what_changed(opt, 1)
    assert first.same and U.changes_answer(opt, first) == "Best so far is still Round 1's best."
    # The rows are Best so far at that Round minus the Start.
    row = next(r for r in early.rows if r.name == "Starch")
    assert (row.start, row.best, row.change) == ("2 g", "5 g", "+3 g")


def test_a_formulation_with_no_round_always_counts_as_recorded():
    opt = _changes_opt()
    opt.tell({"Oil": 9.0, "Sugar": 1.0, "Starch": 5.9}, {"Firmness": 2.0}, formulation_no=6)
    opt.batch_history[-1] = None
    assert "nothing tested beyond" not in U.what_changed(opt, 2).limits


def test_the_open_rounds_rows_count_only_at_the_latest_round():
    rows = CHANGE_ROWS[:4] + [(3, 5, 8.0, 1.0, 4.0, 4.0)]
    opt = _changes_opt(rows)
    opt.pending_batch = [{'formulation': 6, 'recipe': {"Oil": 8.0, "Sugar": 1.0, "Starch": 5.8}}]
    assert "nothing tested beyond" not in U.what_changed(opt, 3).limits    # today
    assert "nothing tested beyond" in U.what_changed(opt, 2).limits        # by Round 2


def test_the_table_and_title_name_the_round():
    opt = _changes_opt()
    ch = U.what_changed(opt, 2)
    table = U.changes_table(opt, ch)
    assert table.startswith("| Ingredient | Start (F2) | Best so far (F4, Round 2) | Change |")
    assert wording.changes_title(2) == "What changed by Round 2"
    assert wording.changes_title(None) == wording.CHANGES_TITLE
    # No slider: today's table, as before.
    assert U.changes_table(opt, U.what_changed(opt)).startswith(
        "| Ingredient | Start (F2) | Best so far (F4) | Change |")


# ------------------------------------------------------------------ #
#  2 · The model's verdicts in the Forecast check
# ------------------------------------------------------------------ #

def _lf(**verdicts):
    """LearnedFacts with hand-made orderings: name -> (right, pairs, verdict)."""
    orderings = {}
    for name, (right, pairs, verdict) in verdicts.items():
        name = insights.SCORE if name == "score" else name.replace("_", " ")
        orderings[name] = learned.Ordering(name, right=right, pairs=pairs, verdict=verdict,
                                           tested=len(verdicts))
    return U.LearnedFacts(orderings, {}, (None, []), None, (0, 0, 0), "")


def test_the_verdict_line_merges_the_so_far_count():
    opt = _spider_opt()
    P, H, N = learned.PREDICTABLE, learned.PARTLY, learned.NOT_YET
    ten = _lf(score=(7, 9, H), Consistency=(11, 11, P), Creaminess=(5, 12, N),
              Tangy=(11, 14, H), Thickness_uniformity=(5, 10, N))
    line, help_ = U.learned_verdict_line(opt, ten)
    assert line == "So far: Overall score 7 of 9 pairs ordered right. Predicts: Consistency."
    six = _lf(score=(5, 5, H), Consistency=(3, 3, N), Creaminess=(2, 4, N), Tangy=(5, 6, H),
              Thickness_uniformity=(2, 3, N))
    line6, _ = U.learned_verdict_line(opt, six)
    assert line6 == "So far: Overall score 5 of 5 pairs ordered right. Partly: Overall score, Tangy."
    for text in (line, line6):
        assert _words(text) <= 14 == wording.LEARNED_LINE_WORDS
    # The "?" says the three words plainly and the adjustment.
    for word in ("Predicts:", "Partly:", "Not yet:", "Adjusted for checking 5 results"):
        assert word in help_
    # Fewer counted pairs than a verdict needs: said in the same line.
    few = _lf(score=(4, 4, N), Consistency=(3, 3, N), Creaminess=(1, 2, N),
              Tangy=(5, 6, H), Thickness_uniformity=(2, 3, N))
    line4 = U.learned_verdict_line(opt, few)[0]
    assert line4 == "So far: Overall score 4 of 4 pairs (too few to judge). Partly: Tangy."
    assert _words(line4) <= 14
    assert learned.MIN_PAIRS == 5
    # No Overall-score pairs: the groups alone, no "So far:".
    none = _lf(score=(0, 0, N), Tangy=(5, 6, H))
    assert U.learned_verdict_line(opt, none)[0] == "Partly: Tangy. Not yet: Overall score."
    assert U.learned_verdict_line(opt, _lf()) is None


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """The Line A demo with Round 6 dropped: Round 5 is the latest scored
    (the Forecast check drawn), synthetic."""
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-slim-scored"))
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
        yield at, opt
    mp.undo()


def _headings(at):
    return [_text(el)[5:] for el, _, _ in _leaves(at.tabs[3])
            if el.type == "markdown" and _text(el).startswith("#### ")]


def test_the_model_section_is_gone_and_its_line_sits_in_the_forecast_check(scored):
    at, opt = scored
    assert wording.LEARNED_TITLE not in _headings(at)
    assert _headings(at) == [wording.Q2_TITLE, wording.Q5_TITLE, wording.FORECAST_CHECK_TITLE]
    texts = [_text(el) for el, _, _ in _leaves(at.tabs[3])]
    head = texts.index(f"#### {wording.FORECAST_CHECK_TITLE}")
    lf = at.session_state[U.LEARNED_KEY][1]          # the screen's own verdicts
    line = U.learned_verdict_line(opt, lf)[0]
    assert texts.count(line) == 1 and texts.index(line) > head
    # Merged: no "So far:" line but the verdict line (here the groups run
    # past 14 words beside the count, so the line is the groups alone).
    assert [t for t in texts if t.startswith("So far:")] in ([], [line])
    assert _words(line) <= 14
    # The strip's Model item is unchanged: the verdicts' words.
    word, names = wording.learned_strip(lf.model_names(opt, learned.PREDICTABLE),
                                        lf.model_names(opt, learned.PARTLY),
                                        lf.model_names(opt, learned.NOT_YET))
    assert any(t.startswith(f"**{word}**") for t in texts)


def test_the_ordering_chart_sits_in_a_closed_details_fold(scored):
    at, _opt = scored
    folds = [e for e in at.tabs[3].expander if e.label == wording.LEARNED_DETAILS_TITLE]
    assert len(folds) == 1 and not folds[0].proto.expanded
    assert len(wording.LEARNED_DETAILS_TITLE.split()) <= 5
    inside = [_text(el) for el, _, _ in _leaves(folds[0])]
    assert any(t.startswith(wording.LEARNED_CAPTION) for t in inside)
    assert folds[0].get("vega_lite_chart")
    # Order: the Forecast check, its fold, then What have we tried?.
    labels = [e.label for e in at.tabs[3].expander]
    assert labels.index(wording.LEARNED_DETAILS_TITLE) < labels.index(wording.Q4_TITLE)


# ------------------------------------------------------------------ #
#  3 · What have we tried? closed
# ------------------------------------------------------------------ #

def test_what_have_we_tried_is_a_closed_fold_worked_out_only_when_open(scored, monkeypatch):
    at, _opt = scored
    folds = [e for e in at.tabs[3].expander if e.label == wording.Q4_TITLE]
    assert len(folds) == 1 and not folds[0].proto.expanded
    assert not folds[0].get("vega_lite_chart")           # nothing worked out
    assert U.FOLDS == {"tried": U.TRIED_FOLD_KEY, "drives": U.DRIVES_FOLD_KEY}
    src = inspect.getsource(U._tried_section)
    assert src.index('getattr(slot, "open", True)') < src.index("search_path_chart(")


def test_what_have_we_tried_opens_with_its_answer_first():
    opt = _spider_opt()
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.session_state[U.TRIED_FOLD_KEY] = True
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    fold = next(e for e in at.tabs[3].expander if e.label == wording.Q4_TITLE)
    inside = [_text(el) for el, _, _ in _leaves(fold) if _text(el)]
    assert inside[0].startswith(f"**{len(ROWS)} tested.**")
    assert fold.get("vega_lite_chart")


def test_the_explored_link_opens_the_tried_fold():
    src = inspect.getsource(U._land)
    assert "EXPLORED_PLACE" in src and "TRIED_FOLD_KEY" in src


# ------------------------------------------------------------------ #
#  4 · The profiler hidden on stub projects
# ------------------------------------------------------------------ #

def _stub_opt(tight=True):
    """Three ingredients 0–10; with `tight`, a batch total of 15 ± 0.1 g, so
    from Best so far each can move about 0.2 g (2 % of its span)."""
    opt = FoodOptimizer(f"slim_stub_{next(_n)}")
    opt.set_amount_unit("g")
    for n in ("A", "B", "C"):
        opt.add_ingredient(n, 0, 10)
    if tight:
        opt.add_quantity_constraint(["A", "B", "C"], min_val=14.9, max_val=15.1)
    opt.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    for no, (a, b, f) in enumerate(((5, 5, 3), (4, 6, 5), (6, 4, 4), (3, 7, 6)), 1):
        opt.tell({"A": a, "B": b, "C": 15.0 - a - b}, {"Firmness": f},
                 formulation_no=no, batch_no=1 + (no > 3))
    return opt


def test_the_profiler_is_hidden_when_most_rows_have_too_little_room():
    stub, roomy = _stub_opt(), _stub_opt(tight=False)
    tight, n = U.room_counts(stub)
    assert n == 3 and tight * 2 > n and U.profiler_hidden(stub)
    assert U.room_counts(roomy)[0] == 0 and not U.profiler_hidden(roomy)
    assert U.ROOM_MIN == 0.05            # the profiler's own rule
    q3 = wording.Q3_TITLE
    at = _review(stub)
    assert q3 not in [e.label for e in at.tabs[3].expander]
    at = _review(roomy)
    folds = [e for e in at.tabs[3].expander if e.label == q3]
    assert len(folds) == 1 and not folds[0].proto.expanded      # the fold as it was


def test_half_is_not_more_than_half(monkeypatch):
    opt = _stub_opt(tight=False)
    monkeypatch.setattr(U, "room_counts", lambda o: (2, 4))
    assert not U.profiler_hidden(opt)
    monkeypatch.setattr(U, "room_counts", lambda o: (3, 4))
    assert U.profiler_hidden(opt)
    monkeypatch.setattr(U, "room_counts", lambda o: (0, 0))
    assert not U.profiler_hidden(opt)


def test_method_says_when_the_profiler_is_hidden():
    assert wording.profiler_hidden_rule() == (
        "What drives each result? Hidden when most ingredients have too little room to move.")
    assert "settings" in wording.profiler_hidden_rule(settings_only=True)
    assert "profiler_hidden_rule" in inspect.getsource(U._how_learned)


# ------------------------------------------------------------------ #
#  Word limits and banned words
# ------------------------------------------------------------------ #

def test_word_limits():
    assert _words(wording.replay_label(10, 10)) <= 5      # a label
    for label in (wording.REPLAY_PLAY, wording.REPLAY_STOP, wording.REPLAY_BACK,
                  wording.REPLAY_EARLIER, wording.REPLAY_LATEST):
        assert _words(label) <= 5
    assert _words(wording.replay_play_help(1, 10)) <= 8
    assert _words(wording.changes_title(10)) <= 5
    assert _words(wording.LEARNED_DETAILS_TITLE) <= 5
    assert _words(wording.STORY_IMPROVING_CAPTION) <= 8
    assert _words(wording.REPLAY_HELP) <= wording.TOOLTIP_WORDS
    assert _words(wording.learned_verdicts_help(5)) <= wording.TOOLTIP_WORDS
    assert _words(wording.profiler_hidden_rule()) <= 16


def test_no_banned_word():
    opt = _changes_opt()
    ch = U.what_changed(opt, 2)
    lf = _lf(score=(5, 5, learned.PARTLY), Tangy=(5, 6, learned.PARTLY))
    texts = [wording.replay_label(4, 10), wording.REPLAY_HELP, wording.REPLAY_PLAY,
             wording.REPLAY_STOP, wording.REPLAY_BACK, wording.REPLAY_EARLIER,
             wording.REPLAY_LATEST, wording.replay_to(3), wording.replay_play_help(1, 6),
             wording.changes_title(4),
             wording.changes_best_column(10, 4), wording.LEARNED_DETAILS_TITLE,
             wording.learned_verdicts_help(5), wording.learned_verdicts_help(1),
             wording.profiler_hidden_rule(), wording.profiler_hidden_rule(True),
             *U.learned_verdict_line(_spider_opt(), lf), U.changes_table(opt, ch),
             ch.limits, ch.limits_help]
    assert [(t, _banned(t)) for t in texts if _banned(t)] == []
