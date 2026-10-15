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

def test_polar_geometry_first_spoke_up_then_clockwise():
    pts = [U.spider_xy(k, 4, 1.0) for k in range(4)]
    want = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for (x, y), (wx, wy) in zip(pts, want):
        assert math.isclose(x, wx, abs_tol=1e-12) and math.isclose(y, wy, abs_tol=1e-12)
    x, y = U.spider_xy(1, 3, 0.5)
    assert math.isclose(math.hypot(x, y), 0.5)
    assert math.isclose(math.degrees(math.atan2(y, x)), 90 - 120)


def test_radius_runs_linearly_from_range_low_to_high():
    sp = {"low": 0.0, "high": 15.0}
    assert U.spider_frac(sp, 0) == 0.0 and U.spider_frac(sp, 15) == 1.0
    assert math.isclose(U.spider_frac(sp, 12), 0.8)
    assert U.spider_frac(sp, 16) == 1.0 and U.spider_frac(sp, -1) == 0.0   # drawn at the end
    shifted = {"low": 2.0, "high": 7.0}
    assert math.isclose(U.spider_frac(shifted, 4.5), 0.5)


def test_gridline_rings_at_round_values_labelled_once():
    assert U.spider_ring_values(0, 15) == [5, 10, 15]
    assert U.spider_ring_values(0, 10) == [2.5, 5, 7.5, 10]
    assert U.spider_ring_values(0, 100) == [25, 50, 75, 100]
    opt = _opt()
    f = U.spider_frames(opt, U.spider_data(opt), 4)
    assert [r["text"] for r in f["ring_labels"]] == ["5", "10", "15"]
    # Each ring label sits on its ring.
    for lab, v in zip(f["ring_labels"], (5, 10, 15)):
        assert math.isclose(math.hypot(lab["x"], lab["y"]), v / 15)
    # Mixed Ranges: rings without numbers (each spoke its own Range).
    mixed = _opt(ranges={"Smoothness": (0, 20)})
    assert U.spider_data(mixed).shared is None
    assert U.spider_frames(mixed, U.spider_data(mixed), 4)["ring_labels"] == []


def test_vertices_sit_at_the_measured_values():
    opt = _opt()
    data = U.spider_data(opt)
    f = U.spider_frames(opt, data, 3)
    best = f["best"]
    assert best[0] == {**best[-1], "j": best[0]["j"]} or (best[0]["x"], best[0]["y"]) == (
        best[-1]["x"], best[-1]["y"])                             # closed
    for p in best[:-1]:
        sp = data.spokes[p["k"]]
        x, y = U.spider_xy(p["k"], 4, U.spider_frac(sp, p["raw"]))
        assert math.isclose(p["x"], x) and math.isclose(p["y"], y)
    # F7 at Round 3: 12 / 5 / 4 / 13, labelled as entered.
    assert [p["value"] for p in best[:-1]] == ["12", "5", "4", "13"]


def test_labels_never_sit_on_a_target_diamond():
    opt = _opt()
    data = U.spider_data(opt)
    f = U.spider_frames(opt, data, 3)
    for lab, p in zip(f["labels"], f["best"]):
        sp = data.spokes[p["k"]]
        r_lab = math.hypot(lab["x"], lab["y"])
        r_tgt = U.spider_frac(sp, sp["target"])
        assert abs(r_lab - r_tgt) >= U._SPIDER_NUDGE - 1e-9 or p["raw"] < sp["target"]
        if p["raw"] < sp["target"]:              # short of target: labelled inside
            assert r_lab < U.spider_frac(sp, p["raw"])


# ------------------------------------------------------------------ #
#  Spider or dot chart
# ------------------------------------------------------------------ #

def test_three_or_more_measurements_draw_the_spider_fewer_the_dot_chart():
    assert U.SPIDER_MIN == 3
    assert U.spider_data(_opt()) is not None
    three = _opt([r[:5] for r in ROWS], MEASURES[:3])
    assert U.spider_data(three) is not None and len(U.spider_data(three).spokes) == 3
    two = _opt([r[:4] for r in ROWS], MEASURES[:2])
    assert U.spider_data(two) is None
    assert U.target_chart(two) is not None       # the dot chart stays


def _review(opt):
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_the_review_draws_the_spider_above_what_changed():
    opt = _opt()
    at = _review(opt)
    texts = [m.value for m in at.tabs[3].markdown]
    title = f"**{wording.SPIDER_TITLE}**"
    assert title in texts and f"**{wording.STORY_TARGET_TITLE}**" not in texts
    k = texts.index(title)
    # Layout study (re-pinned): What changed names the panel's Round and
    # sits beside the profile (the left column, under the score chart).
    assert f"**{wording.changes_title(4)}**" in texts
    # Final demo (A1), re-pinned: the best so far by then, and its own Round.
    assert texts[k + 1] == "Best so far, F7 (Round 3): on target 3 of 4; Sourness 1.0 under."
    captions = [c.value for c in at.tabs[3].caption]
    assert wording.SPIDER_CAPTION in captions
    # Layout study (re-pinned): the Replay panel's control, Round 1 to the
    # latest scored Round, the latest first; Back to latest shown only on
    # an earlier Round.
    assert wording.replay_label(4, 4) in texts
    assert not [b for b in at.tabs[3].button if b.key == "insights_replay_latest"]
    # Stepping back redraws the answer for that Round.
    # (AppTest reruns the whole script and drops the tab widget's state, so
    # the tab is set again; the browser reruns only the fragment.)
    for _ in range(3):
        at.button(key="insights_replay_prev").click()
        at.session_state["main_tab"] = wording.TAB_INSIGHTS
        at.run()
        assert not at.exception, [e.message for e in at.exception]
    texts = [m.value for m in at.markdown]
    assert "Best so far, F2 (Round 1): on target 2 of 4; Firmness 4.0 under." in texts
    assert "Best so far, F7 (Round 3): on target 3 of 4; Sourness 1.0 under." not in texts


def test_two_measurements_keep_start_vs_best():
    opt = _opt([r[:4] for r in ROWS], MEASURES[:2])
    at = _review(opt)
    texts = [m.value for m in at.tabs[3].markdown]
    assert f"**{wording.STORY_TARGET_TITLE}**" in texts
    assert f"**{wording.SPIDER_TITLE}**" not in texts
    # Layout study (re-pinned): the Replay panel's control is drawn for the
    # score chart and What changed; the dot chart follows it too.
    assert wording.replay_label(4, 4) in texts
    assert at.button(key="insights_replay_prev") is not None
    _rounds, best = U.round_bests(opt)
    rows = U.target_rows(opt, best=best[1])
    assert [r["best"] for r in rows] == [8, 4]          # Round 1's best, F2
    assert [r["best"] for r in U.target_rows(opt)] == [12, 5]   # today's, F7


# ------------------------------------------------------------------ #
#  Targets: point, range, higher or lower is better
# ------------------------------------------------------------------ #

def test_max_and_min_draw_an_arrow_and_stay_out_of_the_target_outline():
    opt = _opt(goals={"Sweetness": "max", "Smoothness": "min"})
    data = U.spider_data(opt)
    f = U.spider_frames(opt, data, 4)
    names = {p["name"] for p in f["target"]}
    assert names == {"Firmness", "Sourness"}                      # two targets: a line
    assert len(f["target"]) == 2 and len(f["diamonds"]) == 2
    arrows = {a["name"]: a for a in f["arrows"]}
    assert set(arrows) == {"Sweetness", "Smoothness"}
    # Higher is better: at the rim, pointing out along its spoke (spoke 1:
    # 90°); lower is better: near the centre, pointing in.
    sweet, smooth = arrows["Sweetness"], arrows["Smoothness"]
    assert math.isclose(math.hypot(sweet["x"], sweet["y"]), 1.0) and sweet["angle"] == 90
    assert math.hypot(smooth["x"], smooth["y"]) < 0.1 and smooth["angle"] == (270 + 180) % 360
    spokes = {s["text"] for s in f["spokes"]}
    assert {"Sweetness · higher", "Smoothness · lower", "Firmness · target 12"} <= spokes
    spec = json.dumps(U.spider_chart(opt, data, 4).to_dict())
    assert '"triangle-up"' in spec


def test_a_range_target_is_a_band_on_its_spoke():
    opt = _opt()
    data = U.spider_data(opt)
    spokes = [dict(sp) for sp in data.spokes]
    spokes[2].update(target=None, band=(4.0, 6.0))                # Sourness 4–6
    banded = U.Spider(spokes, data.rounds, data.bests, data.start, data.shared)
    f = U.spider_frames(opt, banded, 4)
    assert len(f["bands"]) == 1
    b = f["bands"][0]
    assert math.isclose(math.hypot(b["x"], b["y"]), 4 / 15)
    assert math.isclose(math.hypot(b["x2"], b["y2"]), 6 / 15)
    assert b["value"] == "4–6"
    assert "Sourness" not in {p["name"] for p in f["target"]}
    assert not f["arrows"]
    U.spider_chart(opt, banded, 4).to_dict()                      # draws


# ------------------------------------------------------------------ #
#  Best so far per Round, the Start
# ------------------------------------------------------------------ #

def test_best_so_far_per_round_with_ties_matches_the_apps_rule():
    opt = _opt()
    data = U.spider_data(opt)
    assert data.rounds == [1, 2, 3, 4]
    got = {r: data.bests[r][1] for r in data.rounds}
    assert got == _independent_bests(ROWS, opt)
    # F7 and F8 tie at Round 3: the first made, as best_index.
    i7 = opt.index_of_formulation(7)
    i8 = opt.index_of_formulation(8)
    assert math.isclose(opt.Y_history[i7], opt.Y_history[i8])
    assert got[3] == got[4] == 7
    # The latest Round agrees with the strip and What changed.
    assert got[4] == opt.best_formulation_no() == U.what_changed(opt).best_no
    # Measured values, as entered.
    assert data.bests[3][2] == {"Firmness": 12.0, "Sweetness": 5.0, "Sourness": 4.0,
                                "Smoothness": 13.0}


def test_waiting_and_not_scored_formulations_never_compete():
    opt = _opt()
    opt.tell({"Oil": 3.0, "Starch": 1.0}, {"Firmness": 12, "Sweetness": 5, "Sourness": 5},
             formulation_no=11, batch_no=5)          # Smoothness still to come
    data = U.spider_data(opt)
    assert data.rounds == [1, 2, 3, 4]               # Round 5 has nothing scored yet
    assert data.bests[4][1] == opt.best_formulation_no() == 7


def test_start_is_what_changeds_start():
    opt = _opt()
    data = U.spider_data(opt)
    ch = U.what_changed(opt)
    assert data.start[1] == ch.start_no == 2 and data.start[2] == ch.start_round == 1
    assert data.bests[1][1] == data.start[1]          # Round 1's best is the Start
    f = U.spider_frames(opt, data, 4)
    assert f["names"]["start"] == "Start (F2)"
    # Final demo (A1), re-pinned: as of Round 4, made in Round 3.
    assert f["names"]["best"] == "Best after Round 4: F7 (Round 3)"
    # Earlier Rounds: Round 2's best (F4) once; neither the Start nor Round 4's.
    assert {p["no"] for p in f["earlier"]} == {4}
    assert U.spider_frames(opt, data, 2)["earlier"] == []
    # Final demo (A2), re-pinned: "on target" as rated, and within noise.
    assert U.spider_round_help(opt, data, 1) == (
        "On target as rated: 2 of 4 (the rating equals the target). Within noise of target "
        "(±0.6, where 95 % of single tastings land): 2 of 4. Beyond its noise: "
        "Firmness 4.0 under (±0.6), Sweetness 1.0 under (±0.6).\n\n"
        "Round 1's best is the Start (F2).")


def test_cached_by_project_state():
    opt = _opt()
    a = U.spider_data(opt)
    assert U.spider_data(opt) is a
    opt.tell({"Oil": 3.0, "Starch": 1.0},
             {"Firmness": 12, "Sweetness": 5, "Sourness": 5, "Smoothness": 13},
             formulation_no=11, batch_no=5)
    b = U.spider_data(opt)
    assert b is not a and b.rounds[-1] == 5 and b.bests[5][1] == 11


# ------------------------------------------------------------------ #
#  Slider and fragment
# ------------------------------------------------------------------ #

def test_control_runs_in_a_fragment():
    # Layout study (re-pinned): the Replay panel's control, in the
    # section's fragment (_progress_fragment), drives the profile too.
    src = inspect.getsource(U)
    assert "@st.fragment\ndef _progress_fragment(" in src
    body = inspect.getsource(U._progress_fragment)
    assert "shown_round(opt, rounds)" in body and "_profile_part(opt, r)" in body


def test_one_scored_round_draws_no_control():
    rows = [r for r in ROWS if r[0] == 1]
    at = _review(_opt(rows))
    assert not [b for b in at.tabs[3].button if (b.key or "").startswith("insights_replay")]
    texts = [m.value for m in at.tabs[3].markdown]
    assert "Best so far, F2 (Round 1): on target 2 of 4; Firmness 4.0 under." in texts


# ------------------------------------------------------------------ #
#  The answer line: the Stop line's tolerance
# ------------------------------------------------------------------ #

def test_answer_line_per_round():
    opt = _opt()
    data = U.spider_data(opt)
    got = [U.spider_round_answer(opt, data, r) for r in data.rounds]
    # Final demo (A1), re-pinned: Best so far as of each Round, named by
    # its Formulation and the Round it was made in (F7 holds through Round 4).
    assert got == ["Best so far, F2 (Round 1): on target 2 of 4; Firmness 4.0 under.",
                   "Best so far, F4 (Round 2): on target 0 of 4; Firmness 2.0 under.",
                   "Best so far, F7 (Round 3): on target 3 of 4; Sourness 1.0 under.",
                   "Best so far, F7 (Round 3): on target 3 of 4; Sourness 1.0 under."]
    assert all(U.limit_words(t) <= U.SPIDER_ANSWER_WORDS for t in got)


@pytest.mark.parametrize("value, on", [(12.04, True), (12.06, False), (11.96, True),
                                       (11.94, False)])
def test_answer_line_counts_on_target_as_the_stop_line(value, on):
    rows = ROWS[:3] + [(2, 4, value, 5, 5, 13)]
    opt = _opt(rows)
    data = U.spider_data(opt)
    assert data.bests[2][1] == 4 == opt.best_formulation_no()
    answer = U.spider_round_answer(opt, data, 2)
    stop = U.ceiling_reason_text(opt)
    n_on = int(re.search(r"on target (\d) of 4", answer).group(1))
    assert stop.startswith(f"On target: {n_on} of 4.")
    assert n_on == (4 if on else 3)
    if not on:
        assert "Firmness 0.1" in answer and "Firmness 0.1" in stop
    # The "?" moves by the same tolerance.
    helped = U.spider_round_help(opt, data, 2)
    assert ("Firmness moved onto target" in helped) == on


def test_answer_line_keeps_to_twelve_words():
    long = [("Thickness uniformity", 12), ("Overall creaminess", 5), ("Lingering sourness", 5),
            ("Smoothness", 13)]
    opt = _opt(measures=long)
    data = U.spider_data(opt)
    # Final demo (A1), re-pinned: the app's own count (a bracket's Round not
    # counted, limit_words); a long name keeps its miss with 'Best F7 (…)'.
    for r in data.rounds:
        line = U.spider_round_answer(opt, data, r)
        assert U.limit_words(line) <= 12 and " under." in line, line


def test_max_and_min_answer_says_short():
    opt = _opt(goals={"Sweetness": "max"})
    line = U.spider_round_answer(opt, U.spider_data(opt), 4)
    # Not every measurement has a target: the Stop line's "at their best".
    # Final demo (A1), re-pinned.
    assert line == "Best F7 (Round 3): at their best 2 of 4; Sweetness 10.0 short."
    assert U.ceiling_reason_text(opt).startswith("At their best: 2 of 4.")


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


def test_word_limits():
    assert _words(wording.SPIDER_TITLE) <= 5
    assert _words(wording.SPIDER_CAPTION) <= 8
    assert wording.SPIDER_TITLE == "Sensory profile against target"
    assert wording.SPIDER_CAPTION == "Measured profile of the best so far."
    for label in (wording.SPIDER_KEY_TARGET, wording.SPIDER_KEY_EARLIER,
                  wording.spider_key_start(3)):
        assert _words(label) <= 5, label
    for text in (wording.spider_help(3, 1), wording.spider_help(3, 1, repeats=True)):
        assert _words(text) <= wording.TOOLTIP_WORDS
    help_ = wording.spider_help(3, 1)
    assert "not per measurement" in help_ and "no model" in help_
    # Final review: true beside the Round line's "?", which gives the ± band.
    assert "No tasting noise is drawn; the Round line's \"?\" gives its ± band, assumed" in help_
    assert "from your Repeats." in wording.spider_help(3, 1, repeats=True)
    # Only a disagreeing Repeat may point at the panel (test_review_diag's BENCH).
    assert not [t for t in _all_texts() if re.search(r"panel|supplier|equipment", t, re.I)]


def test_banned_words():
    assert [(t, _banned(t)) for t in _all_texts() if _banned(t)] == []
