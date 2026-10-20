"""Final demo, stages A and B (owner-approved, 2026-10-13): the Round
panel's labels say "best after Round r"; "on target" is as rated, with
"within noise of target" in its "?"; the Forecast check counts the pairs it
could call; the stop's "?" no longer reads as beating perfect; the score
is match to targets; the stop is a callout with no primary button; Play
draws only the Rounds up to the one on view at one panel height, every
2 s, with double-click guards; the Latest chart marks the stop signal; and
the project's own tasting noise (0.5-3 points, the assumed 1 by default)
drives every Review-side noise reading, the Repeats' measured value taking
over from 3 df.

Synthetic projects only: a yogurt-shaped panel of four 0-15 targets, the
numbers made up so the stop moves with the noise as the pilot's does."""
import contextlib
import inspect
import json
import math
import re
from types import SimpleNamespace as NS

import pytest
from streamlit.testing.v1 import AppTest

import ui_insights as U
import ui_setup
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED

TARGETS = [("Consistency", 12), ("Creaminess", 5), ("Tanginess", 5), ("Thickness", 13)]
# (Round, Formulation, Consistency, Creaminess, Tanginess, Thickness): Round
# bests 91.7 (R1, F3), 92.5 (R3, F7), 95.0 (R4, F10), 98.3 (R6, F16).
ROWS = [
    (1, 1, 6, 3, 7, 13), (1, 2, 8, 3, 6, 12), (1, 3, 8, 4, 5, 13),
    (2, 4, 9, 4, 6, 12), (2, 5, 8, 4, 6, 12), (2, 6, 7, 3, 6, 12),
    (3, 7, 10, 4.5, 6, 12), (3, 8, 9, 3, 6, 12), (3, 9, 9, 4, 6, 12),
    (4, 10, 12, 5, 7, 12), (4, 11, 10, 5, 6, 11), (4, 12, 9, 4, 6, 12),
    (5, 13, 11, 5, 7, 12), (5, 14, 9, 4, 6, 12), (5, 15, 10, 5, 6, 11),
    (6, 16, 12, 5, 4, 13), (6, 17, 12, 5, 4, 12.5), (6, 18, 11, 5, 4, 13),
]


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _recipe(k):
    return {"Oil": 2.0 + 0.5 * k, "Starch": 1.0 + 0.1 * k}


def _opt(name="final_demo", rows=ROWS, noise=None):
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Starch", 0, 5)
    for n, t in TARGETS:
        opt.add_objective(n, 25, goal="target", target=t, min_val=0, max_val=15)
    for k, (r, no, *vals) in enumerate(rows):
        opt.tell(_recipe(k), dict(zip([n for n, _ in TARGETS], vals)),
                 formulation_no=no, batch_no=r)
    if noise is not None:
        opt.tasting_noise = float(noise)
        opt.save()
    return opt


def _score(opt, no):
    return float(opt.Y_history[[int(f) for f in opt.formulation_ids].index(no)])


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _clean(text):
    assert not [p.pattern for p in _BANNED if p.search(text)], text
    for word in ("panel", "replay", "recipe", "held", "scale", "importance", "interval",
                 "batches", "spread"):
        assert word not in text.lower(), (word, text)


class _FakeSt:
    """Records what a builder draws."""

    def __init__(self):
        self.calls, self.session_state = [], {}

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "button":
                return False
            if name in ("container", "empty", "columns"):
                return contextlib.nullcontext(self)
            return None
        return call


# ------------------------------------------------------------------ #
#  The synthetic project mirrors the pilot's shape
# ------------------------------------------------------------------ #

def test_the_project_has_the_pilots_round_bests():
    opt = _opt()
    rounds, at = U.round_bests(opt)
    assert rounds == [1, 2, 3, 4, 5, 6]
    bests = {r: (int(opt.formulation_ids[i]), round(float(opt.Y_history[i]), 2))
             for r, i in at.items()}
    assert bests == {1: (3, 91.67), 2: (3, 91.67), 3: (7, 92.5), 4: (10, 95.0),
                     5: (10, 95.0), 6: (16, 98.33)}


# ------------------------------------------------------------------ #
#  B5 · the tasting noise: its effects at 1, 2 and 3 points
# ------------------------------------------------------------------ #


def test_set_up_has_no_tasting_noise_control():
    """v3: Tasting noise is no longer shown in Set up; it stays internal at
    its default (None), and Set up says nothing of noise or of when to
    stop."""
    assert not hasattr(ui_setup, "_tasting_noise")
    assert not hasattr(ui_setup, "TASTING_NOISE_KEY")
    src = inspect.getsource(ui_setup)
    assert "TASTING_NOISE_LABEL" not in src and "SETUP_NOISE_HELP" not in src
    assert _opt().tasting_noise is None


def test_the_review_has_no_noise_control():
    assert not hasattr(U, "_noise_control") and not hasattr(U, "NOISE_KEY")
    src = inspect.getsource(U)
    assert "selectbox(_say(opt, wording.TASTING_NOISE_LABEL)" not in src
    assert "insights_tasting_noise" not in src


# ------------------------------------------------------------------ #
#  B5 · persistence and backward-compatible loading
# ------------------------------------------------------------------ #

def test_the_noise_is_saved_with_the_project_and_read_back():
    opt = _opt("saved")
    raw = json.load(open("saved.pkl"))
    assert "tasting_noise" not in raw                     # the default writes nothing new
    opt.tasting_noise = 2.0
    opt.save()
    assert json.load(open("saved.pkl"))["tasting_noise"] == 2.0
    again = FoodOptimizer("saved")
    assert again.tasting_noise == 2.0
    # Back to the default (None): nothing on disk.
    again.tasting_noise = None
    again.save()
    assert "tasting_noise" not in json.load(open("saved.pkl"))
    assert FoodOptimizer("saved").tasting_noise is None


def test_an_older_file_loads_with_the_assumed_noise():
    _opt("older")
    raw = json.load(open("older.pkl"))
    raw.pop("tasting_noise", None)
    assert raw["CLASS_VERSION"] == FoodOptimizer.CLASS_VERSION == 17     # nothing bumped
    json.dump(raw, open("older.pkl", "w"))
    before = open("older.pkl").read()
    opt = FoodOptimizer("older")
    assert opt.load_error is None if hasattr(opt, "load_error") else True
    assert opt.tasting_noise is None
    assert open("older.pkl").read() == before            # not rewritten on open
    assert FoodOptimizer.validate_state(raw)


@pytest.mark.parametrize("bad", ["two", -1, 0, True, float("nan"), [2]])
def test_a_damaged_noise_reads_as_the_assumed_one(bad):
    _opt("damaged")
    raw = json.load(open("damaged.pkl"))
    raw["tasting_noise"] = bad
    with open("damaged.pkl", "w") as f:
        f.write(json.dumps(raw).replace("NaN", "NaN"))
    opt = FoodOptimizer("damaged")
    assert opt.tasting_noise is None


def test_start_over_and_a_new_project_keep_the_default():
    fresh = FoodOptimizer("fresh")
    assert fresh.tasting_noise is None
    # app.py's Start this project over writes a fresh FoodOptimizer.
    src = open(APP_PATH).read()
    i = src.index("wording.START_OVER_LABEL")
    assert "_fresh = FoodOptimizer(_target, storage=STORAGE)" in src[i:i + 2000]
    # Suggestions never read it: food_bo has no use of it outside its state.
    import food_bo
    body = inspect.getsource(food_bo.FoodOptimizer.ask)
    assert "tasting_noise" not in body and "assumed_noise" not in body


# ------------------------------------------------------------------ #
#  B2 · progressive Play at one panel height
# ------------------------------------------------------------------ #

def _dataset_rows(spec):
    return [row for rows in spec.get("datasets", {}).values() for row in rows]


def test_an_earlier_round_draws_only_the_rounds_up_to_it():
    opt = _opt(noise=3)
    full = U.improving_chart(opt).to_dict()
    early = U.improving_chart(opt, upto=3).to_dict()
    score = wording.OVERALL_SCORE_COLUMN
    dots = [r for r in _dataset_rows(early) if r.get(score) is not None and "full" in r]
    assert dots and max(r[U._say(opt, wording.FORMULATION_CAP)] for r in dots) == 9
    texts = json.dumps(early, ensure_ascii=False)
    # The best is the headline's to name, not the chart's (N12).
    assert "Best: " not in texts
    assert "Stop signal" not in texts and "after the stop" not in texts
    # Round averages and the best line only to Round 3.
    rounds = {r.get("round") for r in _dataset_rows(early) if "avg" in r}
    assert rounds == {wording.round_tick(r) for r in (1, 2, 3)}
    # The axes as at the latest: a step never moves them.
    def domains(d):
        # Positions only: the key's own labels count what is drawn (v3).
        return sorted({v for v in re.findall(r'"domain": (\[[^\]]*\])', json.dumps(d))
                       if '"' not in v})
    assert domains(early) == domains(full)
    # The kept spec: one per Round on view, the latest the full chart.
    a = U.improving_spec(opt, 6, height=U.PANEL_CHART_H)
    b = U.improving_spec(opt, 3, height=U.PANEL_CHART_H)
    assert str(a) != str(b)


# ------------------------------------------------------------------ #
#  B3 · the stop marker on the Latest chart
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  B4 · Play / Stop debounce
# ------------------------------------------------------------------ #

@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(U.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(U.st, "session_state", {})
    return now


def test_a_stop_right_after_play_is_ignored(clock):
    a, rounds = NS(project_name="A"), [1, 2, 3, 4]
    U._on_play(a, rounds)
    clock[0] += 0.5                                   # a slow double-click's second click
    U._on_stop(a, rounds, 1)
    assert U.replaying(a)
    clock[0] += 0.4                                   # 0.9 s after Play: a real Stop
    U._on_stop(a, rounds, 1)
    assert not U.replaying(a) and U.shown_round(a, rounds) == 1


def test_a_play_right_after_stop_or_the_end_is_ignored(clock, monkeypatch):
    a, rounds = NS(project_name="A"), [1, 2, 3]
    U._on_play(a, rounds)
    clock[0] += 1.0
    U._on_stop(a, rounds, 2)
    clock[0] += 0.3                                   # Stop clicked twice lands on Play
    U._on_play(a, rounds)
    assert not U.replaying(a) and U.shown_round(a, rounds) == 2
    clock[0] += 1.0
    U._on_play(a, rounds)
    assert U.replaying(a)
    # The run ends on the latest: a Play within 1 s of it is ignored too.
    monkeypatch.setattr(U, "_fragment_rerun", lambda: True)
    monkeypatch.setattr(U, "_click_pending", lambda: False)
    monkeypatch.setattr(U.time, "sleep", lambda s: None)

    class Rerun(Exception):
        pass

    def rerun(scope="app"):
        raise Rerun
    monkeypatch.setattr(U.st, "rerun", rerun)
    r = U.shown_round(a, rounds)
    while U.replaying(a):
        with pytest.raises(Rerun):
            U._replay_tick(a, rounds, r)
        r = U.shown_round(a, rounds)
    assert r == 3
    clock[0] += 0.5
    U._on_play(a, rounds)
    assert not U.replaying(a)
    clock[0] += 0.6
    U._on_play(a, rounds)
    assert U.replaying(a) and U.shown_round(a, rounds) == 1
    assert (U.STOP_GUARD_SECS, U.PLAY_GUARD_SECS) == (0.8, 1.0)


def test_the_play_state_is_not_the_play_buttons_widget_state():
    """Streamlit drops an undrawn widget's state after a click's run: with
    the Play button's own key as the playing flag, a second click (an
    ignored Stop) ended Play a Round later."""
    assert U.REPLAY_PLAY != "insights_replay_play"
    assert 'key="insights_replay_play"' in inspect.getsource(U._replay_bar)


# ------------------------------------------------------------------ #
#  B1 · the stop callout; no primary button at the stop
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  A1 · the labels say "best after Round r"
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  A3 · the Forecast check's pairs: called of all
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  A4 · the stop's "?" — one tasting, regression to the mean
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  A5 · the score caption; the wording rules
# ------------------------------------------------------------------ #


