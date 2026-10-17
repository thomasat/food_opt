"""Sensory profile against target (review, 2026-10-08): from three
measurements, Start vs best's dot chart becomes a spider — one spoke per
measurement, the targets in ink, Best so far at the chosen Round filled
blue, the Start (as What changed) dashed grey, earlier Rounds' bests faint —
with a Round slider in a fragment. Measured values only. Synthetic projects
only."""
import inspect
import itertools
import json
import math
import re

import pytest
from streamlit.testing.v1 import AppTest

import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED

_n = itertools.count(1)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _banned(text):
    return [p.pattern for p in _BANNED if p.search(text)]


# Four measurements on 0–15, targets 12 / 5 / 5 / 13 (a synthetic panel).
# Round 1's best is F2 (score order F2 > F3 > F1); Round 2 brings F4; Round 3
# two formulations tie for the top score (F7 and F8, F7 made first); Round 4
# beats nobody.
MEASURES = [("Firmness", 12), ("Sweetness", 5), ("Sourness", 5), ("Smoothness", 13)]
ROWS = [  # (Round, number, Firmness, Sweetness, Sourness, Smoothness)
    (1, 1, 5, 3, 9, 12),
    (1, 2, 8, 4, 5, 13),
    (1, 3, 6, 4, 8, 13),
    (2, 4, 10, 4.5, 6, 12),
    (2, 5, 7, 5, 7, 11),
    (2, 6, 9, 3, 8, 12),
    (3, 7, 12, 5, 4, 13),
    (3, 8, 12, 5, 6, 13),
    (3, 9, 11, 6, 6, 12),
    (4, 10, 11, 5, 7, 12),
]


def _opt(rows=ROWS, measures=MEASURES, goals=None, ranges=None):
    opt = FoodOptimizer(f"spider_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Starch", 0, 5)
    for k, (name, target) in enumerate(measures):
        goal = (goals or {}).get(name, "target")
        lo, hi = (ranges or {}).get(name, (0, 15))
        opt.add_objective(name, 25, goal=goal, target=target if goal == "target" else None,
                          min_val=lo, max_val=hi)
    for k, row in enumerate(rows):
        r, no, *vals = row
        opt.tell({"Oil": 2.0 + k, "Starch": 1.0 + 0.1 * k},
                 {name: v for (name, _t), v in zip(measures, vals)},
                 formulation_no=no, batch_no=r)
    return opt


def _independent_bests(rows, opt):
    """Best so far by Round from the rows alone: the highest Overall score
    among Rounds up to r, the first made on a tie."""
    score = {int(opt.formulation_ids[i]): float(opt.Y_history[i])
             for i in range(len(opt.Y_history))}
    out = {}
    for r in sorted({row[0] for row in rows}):
        mine = sorted(row[1] for row in rows if row[0] <= r)
        top = max(score[n] for n in mine)
        out[r] = next(n for n in mine if abs(score[n] - top) < 1e-9)
    return out


# ------------------------------------------------------------------ #
#  Geometry
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Spider or dot chart
# ------------------------------------------------------------------ #


def _review(opt):
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


# ------------------------------------------------------------------ #
#  Targets: point, range, higher or lower is better
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Best so far per Round, the Start
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Slider and fragment
# ------------------------------------------------------------------ #

def test_control_runs_in_a_fragment():
    # Layout study (re-pinned): the Replay panel's control, in the
    # section's fragment (_progress_fragment), drives the profile too.
    src = inspect.getsource(U)
    assert "@st.fragment\ndef _progress_fragment(" in src
    body = inspect.getsource(U._progress_fragment)
    assert "shown_round(opt, rounds)" in body and "_targets_part(opt, view)" in body


def test_one_scored_round_draws_no_control():
    rows = [r for r in ROWS if r[0] == 1]
    at = _review(_opt(rows))
    assert not [b for b in at.tabs[3].button if (b.key or "").startswith("insights_replay")]
    texts = [m.value for m in at.tabs[3].markdown]
    # Review v3: the headline names Round 1's best.
    assert any(t.startswith("**Best ") and "F2 (Round 1)" in t for t in texts), texts


# ------------------------------------------------------------------ #
#  The answer line: the Stop line's tolerance
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Words
# ------------------------------------------------------------------ #

def _all_texts():
    opt = _opt()
    data = U.spider_data(opt)
    texts = [wording.SPIDER_TITLE, wording.SPIDER_CAPTION, wording.SPIDER_KEY_TARGET,
             wording.SPIDER_KEY_EARLIER, wording.spider_key_best(6, 16),
             wording.spider_key_start(3), wording.spider_help(3, 1),
             wording.spider_help(3, 1, repeats=True), wording.spider_same_help(1, 3),
             wording.spider_moves_help(3, 16, ["Tangy moved 1 away"]),
             wording.spider_off("Tangy", "1.0", "under")]
    for r in data.rounds:
        texts += [U.spider_round_answer(opt, data, r), U.spider_round_help(opt, data, r)]
    return texts


