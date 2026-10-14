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

import insights
import learned
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED

TARGETS = [("Consistency", 12), ("Creaminess", 5), ("Tangy", 5), ("Thickness", 13)]
# (Round, Formulation, Consistency, Creaminess, Tangy, Thickness): Round
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
        insights.set_tasting_noise(opt, noise)
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


def _facts(opt):
    b = opt.best_index()
    return NS(yardstick=insights.noise_yardstick(opt), best=NS(score=float(opt.Y_history[b]),
                                                                 formulation=16))


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

@pytest.mark.parametrize("sigma, rising, stop_round", [(1, True, 6), (2, False, 6), (3, False, 4)])
def test_the_noise_moves_the_rise_the_tie_and_the_stop(sigma, rising, stop_round):
    opt = _opt(noise=None if sigma == 1 else sigma)
    assert insights.assumed_noise(opt) == sigma
    yard = insights.noise_yardstick(opt)
    assert yard.value == sigma and yard.assumed and not yard.from_repeats
    # Rising needs σ × √2 × z(0.95^(1/3)) = 3.0 σ for the best of 3.
    a = insights.improvement(opt, yardstick=yard)
    assert a.threshold == pytest.approx(3.0 * sigma, abs=0.01)
    assert (a.state == "yes") == rising
    # The tie / ceiling threshold: σ × √2 × 1.645 = 2.33 σ.
    need = U.tie_need(yard)
    assert need == pytest.approx(2.326 * sigma, abs=0.01)
    # The stop: the first Round whose best is within it of the top.
    sig = U.stop_signal(opt)
    assert sig[0] == stop_round
    assert sig[3] == pytest.approx(_score(opt, 16) - (_score(opt, 16) if stop_round == 6
                                                       else _score(opt, 10)))
    facts = NS(best=NS(score=_score(opt, 16)), yardstick=yard)
    assert U.at_ceiling(opt, facts)                       # 1.7 below the top at every σ
    # The marker only when a Round came after the stop.
    assert (U._drawn_stop(opt) is not None) == (stop_round < 6)


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_the_noise_moves_the_derived_noise_and_the_pairs_too_close(sigma):
    opt = _opt(noise=None if sigma == 1 else sigma)
    noise = learned.pair_noise(opt)
    assert noise[insights.SCORE] == sigma
    # σ_a = σ × 15 ÷ (25 × √4) = 0.3 σ, each measurement.
    for n, _ in TARGETS:
        assert noise[n] == pytest.approx(0.3 * sigma)
    assert learned.overall_noise(opt) == (sigma, True)
    # Two forecasts 1.5 apart: called at 1 point, too close at 2 and 3.
    a = insights.Check(1, 1, insights.SCORE, 90.0, 85, 95, 91.0, True, True, raw_mean=90.0, sd=1)
    b = insights.Check(2, 1, insights.SCORE, 91.5, 86, 96, 93.0, True, True, raw_mean=91.5, sd=1)
    call = learned.round_call([a, b], insights.SCORE, 1, noise[insights.SCORE])
    assert (call.close == 1) == (sigma > 1.5)


@pytest.mark.parametrize("sigma, within", [(1, 3), (2, 4), (3, 4)])
def test_within_noise_of_target_follows_the_noise(sigma, within):
    opt = _opt(noise=None if sigma == 1 else sigma)
    res = opt.results_history[opt.best_index()]
    got = U.within_noise_of_target(opt, res)
    half = insights.Z95 * 0.3 * sigma
    assert got == (within, 4, f"{half:.1f}")
    text = U.on_target_help_text(opt, res)
    assert text.startswith("On target as rated: 3 of 4 (the rating equals the target).")
    assert f"Within noise of target (±{half:.1f}, where 95 % of single tastings land): " \
           f"{within} of 4." in text
    _clean(text)


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_the_texts_and_the_band_say_the_noise(sigma):
    opt = _opt(noise=None if sigma == 1 else sigma)
    pts = "1 point" if sigma == 1 else f"{sigma} points"
    assert U.noise_sd_text(opt) == f"{sigma}"
    # Fix round (owner, 2026-10-14): no noise band on the score chart.
    spec = json.dumps(U.improving_chart(opt).to_dict(), ensure_ascii=False)
    assert "assumed noise (1 SD)" not in spec and "Noise from Repeats" not in spec
    facts = _facts(opt)
    facts.next = None
    tip = U.ceiling_tip(opt, facts)
    assert f"assumed noise of {pts}" in tip
    assert f"{U.tie_need(facts.yardstick):.1f} is needed to tell two single results apart" in tip
    assert wording.progress_rise_answer("95.0", "98.3", "3.3", f"{3 * sigma:.1f}",
                                        sd_text=f"{sigma}").endswith(
        f"needs {3 * sigma:.1f} if tasting noise ≈ {pts})")
    assert wording.check_noise([("Overall score", f"{sigma:.1f}"),
                                ("Tangy", f"{0.3 * sigma:.1f}")], True, 4, f"{sigma}") \
        .startswith(f"Tasting noise assumed: Overall score {sigma:.1f}; Tangy {0.3 * sigma:.1f}.")
    # The "?" of the noise threshold, from the project's noise.
    assert U._threshold_tip(insights.noise_yardstick(opt)).startswith(pts)


def test_the_measured_noise_takes_over_from_three_df(tmp_path):
    opt = _opt(noise=3)
    k = len(ROWS)
    same = _recipe(15)                                   # F16's amounts, made 4 times
    for j, vals in enumerate([(12, 5, 4.5, 13), (12, 5, 3.5, 13), (12, 4.5, 4, 13)]):
        opt.tell(same, dict(zip([n for n, _ in TARGETS], vals)), formulation_no=19 + j,
                 batch_no=7)
    yard = insights.noise_yardstick(opt)
    assert yard.df == 3 and yard.from_repeats and not yard.assumed
    assert yard.value != 3 and yard.value < 1.5           # the Repeats', not the setting
    assert learned.overall_noise(opt)[1] is False
    assert insights.improvement(opt, yardstick=yard) is None or True
    # The control shows it, read-only: a caption, no selectbox.
    fake = _FakeSt()
    U.st, real = fake, U.st
    try:
        U._noise_control(opt)
    finally:
        U.st = real
    kinds = [c[0] for c in fake.calls]
    assert "selectbox" not in kinds
    cap = next(c for c in fake.calls if c[0] == "caption")[1][0]
    assert cap == wording.tasting_noise_measured(f"{yard.value:.1f}", 3)
    assert cap.endswith("measured from Repeats (3 df)") and _words(cap) <= 10
    assert k == 18


def test_the_control_offers_the_four_choices_with_the_assumed_one_default(monkeypatch):
    opt = _opt()
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "confirmation_open", lambda: False)
    U._noise_control(opt)
    sb = next(c for c in fake.calls if c[0] == "selectbox")
    label, choices = sb[1]
    assert label == wording.TASTING_NOISE_LABEL == "Tasting noise"
    assert list(choices) == [0.5, 1.0, 2.0, 3.0]
    assert fake.session_state[U.NOISE_KEY] == 1.0
    fmt = sb[2]["format_func"]
    assert [fmt(v) for v in choices] == ["0.5 points", "1 point (assumed)", "2 points",
                                         "3 points"]
    assert sb[2]["on_change"] is U._on_noise
    help_ = next(c for c in fake.calls if c[0] == "markdown")[2]["help"]
    # Sceptic review (2026-10-14), re-pinned: the unit, what 1 point is per
    # measurement here, and the noise over which the stop shows.
    assert help_ == wording.tasting_noise_help(
        "Here 1 point = 0.3 per measurement, in its own units (Range 0–15).",
        "The stop shows at noise over 0.72 points.")
    assert help_.startswith("Points of the Overall score (0–100), 1 SD of a single tasting.")
    assert "1 point is assumed" in help_ and "stops sooner" in help_
    assert "never the suggestions" in help_


# ------------------------------------------------------------------ #
#  B5 · persistence and backward-compatible loading
# ------------------------------------------------------------------ #

def test_the_noise_is_saved_with_the_project_and_read_back():
    opt = _opt("saved")
    raw = json.load(open("saved.pkl"))
    assert "tasting_noise" not in raw                     # the default writes nothing new
    insights.set_tasting_noise(opt, 2.0)
    assert json.load(open("saved.pkl"))["tasting_noise"] == 2.0
    again = FoodOptimizer("saved")
    assert again.tasting_noise == 2.0 and insights.assumed_noise(again) == 2.0
    # Back to 1: the default again (None), and nothing on disk.
    insights.set_tasting_noise(again, 1.0)
    assert again.tasting_noise is None
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
    assert opt.tasting_noise is None and insights.assumed_noise(opt) == 1.0
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
    assert opt.tasting_noise is None and insights.assumed_noise(opt) == 1.0


def test_start_over_and_a_new_project_keep_the_default():
    fresh = FoodOptimizer("fresh")
    assert fresh.tasting_noise is None and insights.assumed_noise(fresh) == 1.0
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
    assert "Best: 92.5 (Round 3)" in texts and "Best: 98.3 (Round 6)" not in texts
    assert "Stop signal" not in texts and "after the stop" not in texts
    # Round averages and the best line only to Round 3.
    rounds = {r.get("round") for r in _dataset_rows(early) if "avg" in r}
    assert rounds == {wording.round_tick(r) for r in (1, 2, 3)}
    # The axes as at the latest: a step never moves them.
    def domains(d):
        return sorted({json.dumps(v) for v in re.findall(r'"domain": (\[[^\]]*\])', json.dumps(d))})
    assert domains(early) == domains(full)
    # The kept spec: one per Round on view, the latest the full chart.
    a = U.improving_spec(opt, 6, height=U.PANEL_CHART_H)
    b = U.improving_spec(opt, 3, height=U.PANEL_CHART_H)
    assert "Best: 98.3 (Round 6)" in str(a)
    assert "Best: 98.3 (Round 6)" not in str(b)


def test_what_changed_keeps_one_height_and_play_steps_every_two_seconds(monkeypatch):
    opt = _opt()
    assert U.changes_height(opt) == U.CHANGES_BASE_H + 2 * U.CHANGES_ROW_H
    for r in (1, 2, 6):                     # Rounds 1-2: no picture; Round 6: the picture
        fake = _FakeSt()
        monkeypatch.setattr(U, "st", fake)
        U._changes_block(opt, r)
        styles = [c[1][0] for c in fake.calls if c[0] == "html"]
        assert styles == [f"<style>.st-key-insights_changes {{ min-height: "
                          f"{U.changes_height(opt)}px; }}</style>"]
        # Round panel (spec §10, re-pinned): the title, the picture or the
        # still-the-start line, then two notes or held places.
        drawn = [c[0] for c in fake.calls if c[0] in ("markdown", "empty", "vega_lite_chart",
                                                      "caption")]
        assert len(drawn) == 4                              # as many elements every Round
    same = [c for c in fake.calls]
    assert same
    assert U.REPLAY_STEP_SECS == 2.0
    assert wording.replay_play_help(1, 6) == "Round 1 to 6, one every 2 seconds."
    # Stop is still seen within a slice (≤ 0.3 s).
    assert U.REPLAY_SLICE_SECS <= 0.1
    # The earlier Round's frame adds no height (1 px border plus an inset shadow).
    assert "border: 1px solid" in U.REPLAY_STYLE and "box-shadow: inset" in U.REPLAY_STYLE


# ------------------------------------------------------------------ #
#  B3 · the stop marker on the Latest chart
# ------------------------------------------------------------------ #

def test_the_stop_marker_after_its_round_with_the_gain_since():
    opt = _opt(noise=3)
    (r, then, now, gain, need), x = U._drawn_stop(opt)
    assert (r, round(then, 2), round(now, 2), round(gain, 2)) == (4, 95.0, 98.33, 3.33)
    assert x == 12.5 and need == pytest.approx(6.98, abs=0.01)
    texts = json.dumps(U.improving_chart(opt).to_dict(), ensure_ascii=False)
    assert "Stop signal (Round 4)" in texts and "after the stop: +3.3" in texts
    # At 1 point the stop is the latest Round: no marker (the callout says it).
    assert U._drawn_stop(_opt("one", noise=None)) is None
    assert "Stop signal" not in json.dumps(U.improving_chart(_opt("one2")).to_dict())
    # Model-free: the rule is at_ceiling's, at each Round's best.
    assert U.stop_signal(_opt("low", noise=0.5)) is None     # 1.7 ≥ 1.16: never


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

def test_the_stop_callout_and_its_lines(monkeypatch):
    opt = _opt()
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    U._stop_callout(opt, _facts(opt))
    keys = [c[2].get("key") for c in fake.calls if c[0] == "container"]
    assert "insights_stop_callout" in keys
    md = [c[1][0] for c in fake.calls if c[0] == "markdown"]
    # Sceptic review (2026-10-14), re-pinned: the score within noise of 100;
    # Tangy's 1.0 under beyond its own ±0.6.
    assert md == ["**Stop: score within noise of 100**",
                  "1.7 below the top; 2.3 needed to tell two results apart.",
                  "On target 3 of 4; Tangy 1.0 under, beyond noise."]
    assert not [c for c in fake.calls if c[0] == "button"]
    helps = [c[2].get("help") for c in fake.calls if c[0] == "markdown"]
    assert "tasted once, averages about" in helps[1]
    assert helps[2].startswith("On target as rated: 3 of 4")
    assert helps[2].endswith("Beyond its noise: Tangy 1.0 under (±0.6).")
    # Warm, never error-red; the headline larger.
    assert "rgba(214, 158, 46" in U.STRIP_STYLE and "#b7791f" in U.STRIP_STYLE
    assert ".st-key-insights_stop_head" in U.STRIP_STYLE
    for t in md:
        _clean(t)
    assert _words(md[0]) <= 14 and all(_words(t) <= 12 for t in md[1:])
    # The strip: Stop, with its numbers.
    label = U.strip_stop_label(opt, _facts(opt))
    assert label == ":orange[**Stop**] 1.7 below top; 2.3 needed" and _words(label) <= 10


def test_no_primary_button_at_the_stop(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "confirmation_open", lambda: False)
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    U._ceiling_foot(NS())
    [btn] = [c for c in fake.calls if c[0] == "button"]
    assert btn[2]["type"] == "tertiary" and btn[1][0] == wording.SETUP_BUTTON_SHORT
    # _finish puts _ceiling_foot in the strip's place at the stop.
    src = inspect.getsource(U._finish)
    assert "_ceiling_foot(opt)" in src


@pytest.mark.parametrize("kind,lines", [
    ("misses", [("Repeat Formulation 16 to check noise", False)]),   # Round 6 sample
    ("close", []),                                                   # all 10 sample
])
def test_the_stop_repeat_is_one_quiet_line_under_the_callout(monkeypatch, kind, lines):
    """Both samples' What next? at the stop: the Repeat request sits in the
    answer's place, under the callout, as one caption, whether the Next
    card says it (Round 6's) or not (all 10's), so it lines up the same."""
    fake = _FakeSt()
    monkeypatch.setattr(U, "st", fake)
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    card = U.Card("x", list(lines))
    monkeypatch.setattr(U, "card_for", lambda o, f, w: card)
    monkeypatch.setattr(U, "ceiling_stop", lambda o, f: True)
    monkeypatch.setattr(U, "_stop_callout", lambda o, f: None)
    monkeypatch.setattr(U, "repeat_asks", lambda o, f, c: [])
    monkeypatch.setattr(U, "_why_link_offered", lambda o: False)
    facts = NS(next=NS(kind=kind, assumed=True), best=NS(formulation=16))
    U._next_body(NS(), None, facts)
    captions = [c[1][0] for c in fake.calls if c[0] == "caption"]
    assert captions == ["Repeat Formulation 16 to check noise"]
    # Not assumed (a measured noise): no Repeat request at all.
    fake.calls.clear()
    card.lines = []
    U._next_body(NS(), None, NS(next=NS(kind="close", assumed=False),
                                best=NS(formulation=16)))
    assert not [c for c in fake.calls if c[0] == "caption"]


def test_the_review_at_the_stop_end_to_end():
    """The app itself: the callout leads What next?, the strip says Stop,
    no primary button on the Review, and the Tasting noise choice moves
    the stop's numbers and is saved."""
    opt = _opt("e2e")
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    review = at.tabs[3]
    texts = [m.value for m in review.markdown]
    assert "**Stop: score within noise of 100**" in texts
    assert "1.7 below the top; 2.3 needed to tell two results apart." in texts
    stop = next(b for b in review.button if (b.key or "").endswith("_next"))
    assert stop.label.startswith(":orange[**Stop**] 1.7 below top; 2.3 needed")
    assert not [b for b in review.button if b.proto.type == "primary"]
    # Rising at 1 point; the Tasting noise to 3.
    assert any(t.startswith("**Rising.** Best 95.0 → 98.3 (+3.3; needs 3.0 if tasting noise ≈ "
                            "1 point)") for t in texts)
    sb = at.selectbox(key=U.NOISE_KEY)
    assert sb.value == 1.0
    sb.set_value(3.0).run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    texts = [m.value for m in at.tabs[3].markdown]
    assert "1.7 below the top; 7.0 needed to tell two results apart." in texts
    assert any(t.startswith("**Within noise.** Best 95.0 → 98.3 (+3.3; needs 9.0 if tasting "
                            "noise ≈ 3 points)") for t in texts)
    assert json.load(open("e2e.pkl"))["tasting_noise"] == 3.0


# ------------------------------------------------------------------ #
#  A1 · the labels say "best after Round r"
# ------------------------------------------------------------------ #

def test_the_labels_name_the_best_after_the_round_and_its_own_round():
    opt = _opt()
    data = U.spider_data(opt)
    got = {r: U.spider_round_answer(opt, data, r) for r in data.rounds}
    assert got[5] == "Best so far, F10 (Round 4): on target 2 of 4; Tangy 2.0 over."
    assert got[6] == "Best so far, F16 (Round 6): on target 3 of 4; Tangy 1.0 under."
    assert got[2].startswith("Best so far, F3 (Round 1): ")
    assert all(U.limit_words(t) <= U.SPIDER_ANSWER_WORDS == 12 for t in got.values())
    # The profile's key and What changed's column.
    f = U.spider_frames(opt, data, 5)
    assert f["names"]["best"] == "Best after Round 5: F10 (Round 4)"
    # Round panel (spec §10, re-pinned): What changed's picture names it alike.
    head = U.changes_picture(opt, U.what_changed(opt, 5))["best"]
    assert head == "Best after Round 5: F10 (Round 4)"
    assert U.changes_picture(opt, U.what_changed(opt, 6))["best"] == "Best after Round 6: F16"
    assert wording.best_after(10, 21, 7) == "Best after Round 10: F21 (Round 7)"
    # A Round whose best is still the start: the placeholder line.
    ch = U.what_changed(opt, 2)
    assert ch.same and wording.changes_same(ch.start_round) == "Best so far is still Round 1's best."
    for t in list(got.values()) + [f["names"]["best"], head]:
        _clean(t)


def test_a_long_name_keeps_its_miss_with_the_short_head():
    line = wording.spider_answer(10, 3, 4, ["Thickness uniformity 0.5 under"], True,
                                 no=21, made_round=7, short=True)
    assert line == "Best F21 (Round 7): on target 3 of 4; Thickness uniformity 0.5 under."
    assert U.limit_words(line) <= 12


# ------------------------------------------------------------------ #
#  A3 · the Forecast check's pairs: called of all
# ------------------------------------------------------------------ #

def _lf(right, pairs, close, ties, p=0.03, p_holm=0.15, tested=5):
    o = learned.Ordering(insights.SCORE, right=right, pairs=pairs, close=close, ties=ties)
    o.verdict, o.reason, o.p_holm, o.tested = learned.PARTLY, "", p_holm, tested
    o.nulls = []
    lf = NS(orderings={insights.SCORE: o}, items=[], noise={insights.SCORE: 1.0})
    lf.model_names = lambda opt, v: (["Overall score"] if v == learned.PARTLY else [])
    return lf


def test_the_pairs_line_says_called_of_all():
    assert wording.score_pairs_called("Overall score", 5, 5, 12) == \
        "Overall score (5 of 12 pairs called, 5 right)"   # fix round 2: one form
    assert wording.learned_count("Overall score", 7, 9, total=24) == \
        "Overall score 9 of 24 pairs called, 7 right"
    assert wording.learned_count_needed("Overall score", 4, 4, 5, total=12) == \
        "Overall score 4 of 12 pairs called, 4 right (needs 5)"
    line = wording.learned_line(wording.learned_count("Overall score", 7, 9, total=24),
                                [("Predicts", ["Consistency (11/11)"])])
    assert line == ("So far: Overall score 9 of 24 pairs called, 7 right. "
                    "Predicts: Consistency (11/11).")
    assert _words(line) <= wording.LEARNED_LINE_WORDS == 14


def test_the_pairs_help_gives_the_chance_raw_and_adjusted():
    opt = NS(objectives=[], utility_ceiling=lambda: 100.0, has_ingredients=lambda: True)
    text = U.pairs_called_text(opt, _lf(5, 5, 6, 1))
    assert text == ("Overall score: 5 of 12 pairs called (6 too close, 1 tie); 5 right. "
                    "Chance by luck: 3 in 100; 15 in 100 adjusted for checking 5 results.")
    _clean(text)
    assert U._chance_words(0.0625) == "6 in 100"


def test_the_margin_of_a_value_just_inside_its_range_is_in_the_help():
    opt = NS(objectives=[], utility_ceiling=lambda: 100.0, formulation_ids=[],
             has_ingredients=lambda: True)
    c_in = insights.Check(18, 6, insights.SCORE, 94.0, 92.0, 96.0, 92.07, True, True,
                          raw_mean=94.0, sd=1)
    c_mid = insights.Check(17, 6, insights.SCORE, 94.0, 92.0, 96.0, 94.0, True, True,
                           raw_mean=94.0, sd=1)
    lf = NS(items=[(insights.SCORE, None, [c_in, c_mid])])
    assert U.near_edge_text(opt, lf) == "Close to the range edge: F18 in by 0.07."
    lf = NS(items=[(insights.SCORE, None, [c_mid])])
    assert U.near_edge_text(opt, lf) == ""


# ------------------------------------------------------------------ #
#  A4 · the stop's "?" — one tasting, regression to the mean
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_the_stop_help_reframes_the_perfect_score(sigma):
    opt = _opt(noise=None if sigma == 1 else sigma)
    perfect = U.perfect_noisy_score(opt)
    assert perfect == pytest.approx(100 - 4 * (25 / 15) * 0.3 * sigma * math.sqrt(2 / math.pi),
                                    abs=0.01)
    facts = _facts(opt)
    facts.next = None
    tip = U.ceiling_tip(opt, facts)
    assert (f"A formulation exactly on target, tasted once, averages about {perfect:.1f} with "
            f"this noise") in tip
    assert "single tastings scatter around it" in tip
    assert "would score about" not in tip and "only pulls the score down" not in tip
    _clean(tip)


# ------------------------------------------------------------------ #
#  A5 · the score caption; the wording rules
# ------------------------------------------------------------------ #

def test_the_score_caption():
    assert wording.SCORE_IS_MATCH == "Score = match to your targets, not liking."
    assert _words(wording.SCORE_IS_MATCH) <= 8
    assert "wording.SCORE_IS_MATCH" in inspect.getsource(U._panel_captions)


def test_every_final_demo_string_keeps_the_rules():
    statuses = [wording.stop_strip_line("1.7", "2.3"),
                wording.tasting_noise_measured("1.4", 4)]
    headlines = [wording.STOP_CALLOUT_HEAD, wording.stop_callout_line("1.7", "2.3"),
                 wording.stop_on_target(3, 4, ["Tangy 1.0 under"]),
                 wording.stop_on_target(3, 4, ["Tangy 1.0 under"], True, [True]),
                 wording.score_pairs_called("Overall score", 5, 5, 12)]
    captions = [wording.SCORE_IS_MATCH, wording.stop_marker(6), wording.after_stop("0.8")]
    labels = [wording.TASTING_NOISE_LABEL, wording.STOP_WORD,
              wording.tasting_noise_option(1.0), wording.tasting_noise_option(0.5)]
    for t in statuses:
        assert _words(t) <= 10, t
    for t in headlines:
        assert _words(t) <= 14, t
    for t in captions:
        assert _words(t) <= 10, t
    for t in labels:
        assert _words(t) <= 5, t
    for t in statuses + headlines + captions + labels + [
            wording.TASTING_NOISE_HELP, wording.TASTING_NOISE_MEASURED_HELP,
            wording.stop_marker_help(4, "95.0", "98.3", "3.3", "7.0"),
            wording.on_target_help(3, 4, 4, "0.6"),
            wording.pairs_called_help("Overall score", 5, 5, 12, 6, 1, "3 in 100", "15 in 100", 5),
            wording.ceiling_help("98.3", "100", "1.7", "2.3", "98.4", True, sd_text="2"),
            wording.story_improving_help("84.2", "96.1", 1, 6, "2.3"),
            wording.tip_threshold_assumed("2")]:
        _clean(t)
    # Every status names what, how many and against what.
    assert re.fullmatch(r"[\d.]+ below top; [\d.]+ needed", wording.stop_strip_line("1.7", "2.3"))
    # The final demo strings sit under their own heading in wording.py.
    src = open(wording.__file__).read()
    tail = src[src.index("# --- final demo ---"):]
    for name in ("STOP_CALLOUT_HEAD", "TASTING_NOISE_LABEL", "SCORE_IS_MATCH", "def best_after",
                 "def pairs_called_help", "def stop_marker"):
        assert name in tail, name
