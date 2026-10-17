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
from types import SimpleNamespace as NS
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


def test_the_section_reruns_as_one_fragment():
    src = inspect.getsource(U)
    assert "@st.fragment\ndef _progress_fragment(" in src
    body = inspect.getsource(U._progress_fragment)
    assert "_replay_bar(" in body and "shown_round(" in body
    # Review v3: the headline and the three charts all step with the Round.
    for part in ("headline_text(", "_improving_part(", "_targets_part(", "_amounts_part("):
        assert part in body
    # No slider, no fragment of its own for any part.
    assert "select_slider" not in inspect.getsource(U._replay_bar)
    assert not hasattr(U, "_spider_fragment")


# ------------------------------------------------------------------ #
#  1a · The score chart's highlight
# ------------------------------------------------------------------ #


def test_round_bests_follow_best_index_rule():
    opt = _spider_opt()
    rounds, at = U.round_bests(opt)
    assert rounds == [1, 2, 3, 4]
    assert {r: int(opt.formulation_ids[i]) for r, i in at.items()} == {1: 2, 2: 4, 3: 7, 4: 7}
    assert at[4] == opt.best_index()


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


# ------------------------------------------------------------------ #
#  3 · What have we tried? closed
# ------------------------------------------------------------------ #

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


