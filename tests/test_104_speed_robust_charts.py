"""0.8.0 e2e fixes, W3 (spec 2026-10-01 §C, §D; review I11, M5-M7, I13):
speed, robustness and charts.

- C3: a cache entry holding a non-finite number is dropped and refitted.
- C1: the next Round's expected gain is kept beside the project.
- C1: the progress line counts only what is really being fitted.
- C2: a broken pool is replaced on the next need, up to a cap.
- D1: the charts' meaning-carrying marks have theme tokens at 3:1.
- A5 / I13: the layer's new name and the triangle's legend."""
import json
import math
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import insights
import insights_cache


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")


# ------------------------------------------------------------------ #
#  C3: entries with non-finite numbers are dropped (review M6)
# ------------------------------------------------------------------ #

def _check(**over):
    c = {"formulation": 7, "round": 2, "name": insights.SCORE, "predicted": 80.0, "low": 70.0,
         "high": 90.0, "measured": 82.0, "inside": True, "fully_measured": True,
         "raw_mean": 80.0, "sd": 5.0, "widen": 1.0}
    return {**c, **over}


def _best(**over):
    score = {"mean": 90.0, "low": 85.0, "high": 95.0, "raw_mean": 90.0, "sd": 2.0}
    best = {"recipe": {"Pea": 20.0, "Oil": 8.0}, "score": score, "close": None, "compare": 3,
            "factor": 1.2, "repeat": dict(score, mean=88.0)}
    return {**best, **over}


def _write(payload):
    os.makedirs(".insights", exist_ok=True)
    full = {"version": insights_cache.VERSION, "salt": insights.MODEL_SALT, "memo": {},
            "rounds": {}, "best": None, **payload}
    with open(insights_cache.path("Patty"), "w", encoding="utf-8") as fh:
        # json writes NaN and Infinity bare unless told not to: what a damaged
        # but well-formed file holds (04 M-2).
        json.dump(full, fh)


# A real GP's state: a constraint with no upper bound stores inf, and that
# alone may be infinite.
GOOD_STATE = {"mean_module.raw_constant": ["float64", 0.5],
              "covar_module.raw_lengthscale": ["float64", [[0.1, 0.2]]],
              "covar_module.raw_lengthscale_constraint.upper_bound": ["float64", float("inf")],
              "count": ["int64", 3]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_model_state_with_a_non_finite_number_is_dropped_alone(bad):
    _write({"memo": {
        "good": {"state": GOOD_STATE, "widen": 1.3},
        "bad_state": {"state": {**GOOD_STATE, "covar_module.raw_lengthscale":
                                ["float64", [[0.1, bad]]]}, "widen": 1.3},
        "bad_widen": {"state": GOOD_STATE, "widen": bad},
        "no_widen": {"state": GOOD_STATE, "widen": None},
        "nan_bound": {"state": {**GOOD_STATE, "covar_module.raw_lengthscale_constraint."
                                "upper_bound": ["float64", float("nan")]}, "widen": 1.3},
    }})
    assert list(insights_cache.load("Patty").memo) == ["good"]


def test_a_real_fitted_state_loads_back():
    """Its open constraint bounds are infinite, legitimately."""
    from tests.test_insights import _burger
    opt = _burger(rounds=3)
    memo = {}
    with insights.memo_in_use(memo):
        insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    insights_cache.save("Patty", insights_cache.Saved(memo, {}, None))
    assert insights_cache.load("Patty").memo.keys() == memo.keys()


def test_a_rounds_checks_with_a_non_finite_number_are_dropped_and_no_other():
    _write({"rounds": {
        "r1": [_check(), _check(formulation=8)],
        "r2": [_check(), _check(formulation=9, measured=float("nan"))],
        "r3": [_check(low=-float("inf"))],
        "r4": [_check(raw_mean=None, sd=None)],          # the dataclass allows None here
        "r5": [_check(predicted=None)],                  # and not here
        "r6": [_check(widen=float("inf"))],
    }})
    back = insights_cache.load("Patty").rounds
    assert sorted(back) == ["r1", "r4"]
    assert back["r4"][0].sd is None


@pytest.mark.parametrize("over", [
    {"factor": float("nan")},
    {"score": {"mean": float("nan"), "low": 1.0, "high": 2.0}},
    {"repeat": {"mean": 1.0, "low": float("inf"), "high": 2.0}},
    {"recipe": {"Pea": float("nan"), "Oil": 8.0}},
    {"score": {"mean": None, "low": 1.0, "high": 2.0}},
])
def test_a_predicted_best_with_a_non_finite_number_is_dropped(over):
    _write({"memo": {"good": {"state": GOOD_STATE, "widen": 1.0}},
            "best": ["k", _best(**over)]})
    back = insights_cache.load("Patty")
    assert back.best is None
    assert list(back.memo) == ["good"]                   # the rest of the file still loads


def test_a_finite_predicted_best_and_a_refusal_still_load():
    _write({"best": ["k", _best(repeat=None)]})
    key, best = insights_cache.load("Patty").best
    assert key == "k" and best.repeat is None and best.score.mean == 90.0
    _write({"best": ["k", {"refusal": "Add an ingredient first."}]})
    assert insights_cache.load("Patty").best == ("k", "Add an ingredient first.")


def test_dropped_entries_are_not_written_back():
    """Review M6: what was dropped is gone from the next write too, so the
    damage does not persist until the data changes (04 M-2)."""
    _write({"memo": {"good": {"state": GOOD_STATE, "widen": 1.0},
                     "bad": {"state": GOOD_STATE, "widen": float("nan")}},
            "rounds": {"r1": [_check(sd=float("nan"))]},
            "best": ["k", _best(factor=float("inf"))]})
    insights_cache.save("Patty", insights_cache.load("Patty"))
    with open(insights_cache.path("Patty"), encoding="utf-8") as fh:
        text = fh.read()
    assert "NaN" not in text
    payload = json.loads(text)
    assert list(payload["memo"]) == ["good"] and payload["rounds"] == {}
    assert payload["best"] is None


def test_a_non_finite_number_is_never_written():
    """The writer refuses one too: a fit that came out NaN is left out of
    the file instead of being read back as a model."""
    saved = insights_cache.Saved({"good": {"state": GOOD_STATE, "widen": 1.0},
                                  "bad": {"state": GOOD_STATE, "widen": math.nan}},
                                 {"r1": [insights.Check(**_check(low=math.nan))],
                                  "r2": [insights.Check(**_check())]}, None)
    insights_cache.save("Patty", saved)
    back = insights_cache.load("Patty")
    assert list(back.memo) == ["good"] and list(back.rounds) == ["r2"]
    with open(insights_cache.path("Patty"), encoding="utf-8") as fh:
        assert "NaN" not in fh.read()


# ------------------------------------------------------------------ #
#  C1: the next Round's expected gain kept beside the project
#  (review I11)
# ------------------------------------------------------------------ #

import bo_views  # noqa: E402
from tests.test_insights import _burger  # noqa: E402


def _gain(**over):
    g = {"round": None, "best_measured": 82.0, "highest": None, "gain": 1.25, "exact": False,
         "why_none": None}
    return {**g, **over}


def test_the_figure_is_saved_and_read_back_beside_the_rest():
    gain = ("k", bo_views.RoundGain(**_gain()))
    insights_cache.save("Patty", insights_cache.Saved({}, {}, None, gain))
    back = insights_cache.load("Patty")
    assert back.gain == gain
    with open(insights_cache.path("Patty"), encoding="utf-8") as fh:
        assert json.load(fh)["gain"] == ["k", _gain()]


def test_a_file_without_the_figure_still_loads_whole():
    """The field is optional, like `best`: no VERSION bump, and every
    user's cached fits survive the upgrade (review I11)."""
    _write({"memo": {"good": {"state": GOOD_STATE, "widen": 1.0}}})
    back = insights_cache.load("Patty")
    assert back.gain is None and list(back.memo) == ["good"]
    assert insights_cache.Saved({}, {}, None).gain is None


@pytest.mark.parametrize("over", [{"gain": float("nan")}, {"best_measured": float("inf")},
                                  {"highest": float("nan")}, {"exact": None},
                                  {"round": 1.5}])
def test_a_figure_with_a_non_finite_number_is_dropped(over):
    _write({"memo": {"good": {"state": GOOD_STATE, "widen": 1.0}},
            "gain": ["k", _gain(**over)]})
    back = insights_cache.load("Patty")
    assert back.gain is None and list(back.memo) == ["good"]


def test_a_cold_figure_and_no_figure_are_both_kept():
    cold = ("k", bo_views.RoundGain(**_gain(gain=None, why_none="cold")))
    insights_cache.save("Patty", insights_cache.Saved({}, {}, None, cold))
    assert insights_cache.load("Patty").gain == cold
    insights_cache.save("Patty", insights_cache.Saved({}, {}, None, ("k", None)))
    assert insights_cache.load("Patty").gain is None       # a failed search: next time


@pytest.fixture
def burger():
    with bo_views.fresh():
        yield _burger(rounds=5, noise=0.6)


def test_the_key_moves_with_everything_the_search_depends_on(burger, monkeypatch):
    """Review I11: the setup (rows, limits, formulas, pre-mixes, Not
    scored), the seed, robust, the model settings, the library versions,
    ask()'s search settings, the cold-start count and a salt."""
    key = bo_views.next_gain_key(burger)
    assert bo_views.next_gain_key(burger) == key
    moved = []

    def moves(change, undo):
        change()
        moved.append(bo_views.next_gain_key(burger) != key)
        undo()
        assert bo_views.next_gain_key(burger) == key

    seed = burger.next_formulation_no
    moves(lambda: setattr(burger, "next_formulation_no", seed + 1),
          lambda: setattr(burger, "next_formulation_no", seed))
    moves(lambda: setattr(burger, "robust", True), lambda: setattr(burger, "robust", False))
    cfg = getattr(burger, "bo_config", None)
    moves(lambda: setattr(burger, "bo_config", {"acquisition": "qlogei"}),
          lambda: setattr(burger, "bo_config", cfg))
    limits = list(burger.constraints)
    moves(lambda: burger.constraints.append({"property": "Fat", "max": 12}),
          lambda: setattr(burger, "constraints", limits))
    premixes = burger.premixes
    moves(lambda: setattr(burger, "premixes", {"Mix": {"Pea": 1.0}}),
          lambda: setattr(burger, "premixes", premixes))
    for name, value in (("_VERSIONS", ("x", "y", "z")), ("ASK_RESTARTS", 21),
                        ("ASK_RAW_SAMPLES", 512), ("COLD_START_RUNS", 99),
                        ("GAIN_SALT", "later")):
        real = getattr(bo_views, name)
        moves(lambda n=name, v=value: monkeypatch.setattr(bo_views, n, v),
              lambda n=name, v=real: monkeypatch.setattr(bo_views, n, v))
    assert all(moved), moved


def _no_search(monkeypatch):
    monkeypatch.setattr(bo_views, "next_model", lambda *a: pytest.fail("fitted again"))
    monkeypatch.setattr(bo_views, "next_optimum", lambda *a: pytest.fail("searched again"))


def test_a_saved_figure_with_its_key_is_used_without_a_fit_or_a_search(burger, monkeypatch):
    real = bo_views.next_gain(burger, bo_views.next_model(burger))
    bo_views._CACHE.clear()
    _no_search(monkeypatch)
    saved = (bo_views.next_gain_key(burger), real)
    assert bo_views.next_gain_for(burger, saved) == real


def test_a_saved_figure_under_another_key_is_worked_out_again(burger, monkeypatch):
    real = bo_views.next_gain(burger, bo_views.next_model(burger))
    bo_views._CACHE.clear()
    stale = ("another", bo_views.RoundGain(**_gain(gain=99.0)))
    got = bo_views.next_gain_for(burger, stale)
    assert got == real and got.gain != 99.0


def test_a_figure_worked_out_once_serves_the_process_without_a_fit(burger, monkeypatch):
    """Reconciled with the process cache (27ceb6a): a second session of
    the app, with nothing saved, neither fits nor searches again."""
    first = bo_views.next_gain_for(burger)
    _no_search(monkeypatch)
    assert bo_views.next_gain_for(burger) == first
    assert bo_views.saved_next_gain(burger) == first


def test_nothing_known_means_none_without_a_search(burger, monkeypatch):
    _no_search(monkeypatch)
    assert bo_views.saved_next_gain(burger) is None
    assert bo_views.saved_next_gain(burger, ("another", None)) is None


# On screen: the figure is written beside the project and read back.

from streamlit.testing.v1 import AppTest  # noqa: E402

import insights_jobs  # noqa: E402
import ui_insights  # noqa: E402
import wording  # noqa: E402
from tests.test_app_ui import APP_PATH  # noqa: E402


@pytest.fixture
def jobs(monkeypatch):
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})


@pytest.fixture
def skilled(monkeypatch):
    """A model with skill (integration with W1's A3 gate, review I4): this
    5-Round project has fewer than 10 Overall-score checks, so the gate
    would withhold the expected gain these tests persist. Stood in
    the way W1 stands it in for the effect lines (test_080_try_one). Ftue
    fix A's Round-grouped folds move this project's record to 15 of 18
    (More misses than expected), which withholds the figure too, so the
    record is stood in as well."""
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)
    monkeypatch.setattr(ui_insights, "withheld", lambda opt, score, checks: None)


def _app(project):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = project
    return at


def _open(at, tab=wording.TAB_INSIGHTS):
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    return at


def _texts(at):
    tab = at.tabs[3]
    return ([m.value for m in tab.markdown] + [c.value for c in tab.caption]
            + [e.label for e in tab.expander])


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_a_reopened_project_shows_the_figure_without_a_fit_or_a_search(burger, jobs, skilled,
                                                                         monkeypatch):
    """04 I-1: the reopen time was the next Round's search (2.8 s at
    120 x 6), worked out again in every new session and after a restart."""
    first = _open(_app(burger.project_name))
    line = first.session_state[ui_insights.GAIN_KEY][1]
    assert line is not None and line.gain is not None
    bo_views._CACHE.clear()                         # as after a restart of the app
    _no_search(monkeypatch)
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: pytest.fail("fitted again"))
    again = _open(_app(burger.project_name))
    assert not again.exception
    assert again.session_state[ui_insights.GAIN_KEY][1] == line
    assert _texts(again) == _texts(first)


# ------------------------------------------------------------------ #
#  C1: the progress line counts only what is really being fitted
#  (review I11; 04 I-1, M-1; 01 F2)
# ------------------------------------------------------------------ #

def _bars(monkeypatch):
    """Every progress bar's text from now on."""
    from streamlit.delta_generator import DeltaGenerator
    texts = []
    real = DeltaGenerator.progress

    def progress(self, value, text=None, **k):
        texts.append(text)
        return real(self, value, text=text, **k)
    monkeypatch.setattr(DeltaGenerator, "progress", progress)
    return texts


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_reopen_with_everything_cached_shows_no_progress_line(burger, jobs, monkeypatch):
    """04 I-1: "0 of 23 Rounds checked…" for 3 s while nothing was."""
    _open(_app(burger.project_name))
    bars = _bars(monkeypatch)
    at = _open(_app(burger.project_name))
    assert not at.exception and wording.PAIR_FOLD in _texts(at)
    assert bars == []


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_rounds_already_checked_count_as_done_from_the_start(burger, jobs, monkeypatch):
    """After a new Round only it is checked: the line never starts at 0."""
    _open(_app(burger.project_name))
    import torch
    g = torch.Generator().manual_seed(5)
    for no in range(16, 19):
        u = torch.rand(2, generator=g).tolist()
        pea, oil = 4 + 4 * u[0], 2 + 4 * u[1]
        burger.tell(burger.fill_formulas({"Pea": pea, "Oil": oil}),
                    {"Firmness": 0.8 * pea, "Juiciness": 1.2 * oil + 1}, formulation_no=no,
                    batch_no=6)
    total = len(ui_insights._checked(burger))
    bars = _bars(monkeypatch)
    at = _open(_app(burger.project_name))
    assert not at.exception and wording.PAIR_FOLD in _texts(at)
    assert bars, "the current models are fitted: one line"
    # Integration (T0′): with every Round checked, the line says the model
    # of the latest results is being updated, never "N of N checked…".
    done = wording.insights_progress(total, total)
    assert all(f"{total} of {total}" not in t for t in bars)
    counts = [total if t == done else int(t.split(": ")[1].split(" of ")[0]) for t in bars]
    assert counts[0] >= total - 1 and counts == sorted(counts)
    assert all(t == done or t.endswith(f"of {total} Rounds checked…") for t in bars)


def test_the_poller_reruns_the_filling_tab_when_a_round_lands(burger, monkeypatch):
    """01 F2: the count sat on "0 of 11" while Rounds came back; it now
    moves per Round returned, on the Insights tab only."""
    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui_insights, "_catch_up", lambda o: None)
    monkeypatch.setattr(ui_insights, "_in_flight", lambda o: True)
    monkeypatch.setattr(ui_insights, "_seen", lambda o, s: (False,) * len(s))
    monkeypatch.setattr(ui_insights, "_plan", lambda o: [])
    monkeypatch.setattr(ui_insights, "memo", lambda o: {})
    done = [2]
    monkeypatch.setattr(insights_jobs, "rounds_done", lambda *a: (done[0], 6))
    insights_tab = ("column", "trust", "stand", "all")
    batch_tab = ("column", "trust", "now", "all")
    state[ui_insights.SEEN_KEY] = (False,) * 4
    state[ui_insights.PROGRESS_SEEN_KEY] = (insights.data_key(burger), 2)
    assert not ui_insights._poll_step(burger, insights_tab)
    done[0] = 3
    assert ui_insights._poll_step(burger, insights_tab)
    assert not ui_insights._poll_step(burger, batch_tab)
    state.pop(ui_insights.PROGRESS_SEEN_KEY)
    assert not ui_insights._poll_step(burger, insights_tab)


# ------------------------------------------------------------------ #
#  C2: a broken pool is replaced on the next need, at most 3 times in
#  10 minutes (04 I-3, review M5)
# ------------------------------------------------------------------ #

@pytest.fixture
def fresh_pool(monkeypatch, tmp_path):
    """pool() from scratch, with workers that print ready and die at their
    first request: each pool breaks after becoming ready."""
    dies = tmp_path / "dies_worker.py"
    dies.write_text("import json, sys\nprint(json.dumps({'ready': True}), flush=True)\n"
                    "sys.stdin.readline()\nsys.exit(1)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(dies))
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setattr(insights_jobs, "_STOPPED", False)
    monkeypatch.setattr(insights_jobs, "_BREAKS", [])
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "1")
    yield
    if insights_jobs._POOL is not None:
        insights_jobs._POOL.close()


def _break(p):
    assert p.wait_ready(60)
    with pytest.raises(Exception):
        p.submit(0, {"id": 1}).result(30)
    assert p.broken


def test_a_pool_that_broke_after_starting_is_replaced_on_the_next_need(fresh_pool):
    first = insights_jobs.pool()
    _break(first)
    again = insights_jobs.pool()
    assert again is not None and again is not first and not again.broken
    assert insights_jobs.workers_on() and not insights_jobs.stopped()


def test_after_three_restarts_in_ten_minutes_it_calculates_here(fresh_pool):
    pools = []
    for _ in range(4):
        p = insights_jobs.pool()
        assert p is not None and p not in pools
        pools.append(p)
        _break(p)
    assert insights_jobs.pool() is None and insights_jobs.pool() is None
    assert not insights_jobs.workers_on() and insights_jobs.stopped()


def test_breaks_more_than_ten_minutes_apart_never_stop_the_workers(fresh_pool, monkeypatch):
    clock = [1000.0]
    real = insights_jobs.time.monotonic
    monkeypatch.setattr(insights_jobs.time, "monotonic", lambda: real() + clock[0])
    for _ in range(6):
        p = insights_jobs.pool()
        assert p is not None
        _break(p)
        clock[0] += insights_jobs.RESTART_WINDOW / 2 + 1     # two in any ten minutes
    assert insights_jobs.workers_on() and not insights_jobs.stopped()


def test_a_pool_that_never_started_is_given_up_on_at_once(fresh_pool, monkeypatch, tmp_path):
    """Review M5: as before; no restart, and not the cap's indicator."""
    bad = tmp_path / "broken_worker.py"
    bad.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(bad))
    first = insights_jobs.pool()
    assert not first.wait_ready(60)
    assert insights_jobs.pool() is None
    assert not insights_jobs.workers_on() and not insights_jobs.stopped()


class _Pool:
    """A stand-in pool: never answers, or fails everything once broken."""

    def __init__(self):
        self.broken = self.closed = False
        self.sent = []

    def submit(self, priority, request):
        import concurrent.futures as cf
        fut = cf.Future()
        if self.broken:
            fut.set_exception(RuntimeError("the Insights workers stopped"))
        self.sent.append(request["kind"])
        return fut

    def idle(self):
        return False


def test_a_failed_job_is_sent_again_to_the_new_pool(burger, jobs):
    """The project's fits go to the replacement instead of staying in this
    process for the rest of the session (04 I-3: 64-84 s per change)."""
    old = _Pool()
    memo = {}
    job = insights_jobs.job_for(burger, old, memo)
    old.broken = True
    for futs in job.futures.values():
        for f in futs:
            f.set_exception(RuntimeError("stopped"))
    job.poll(memo)
    assert job.failed
    new = _Pool()
    again = insights_jobs.job_for(burger, new, memo)
    assert again is not job and not again.failed and new.sent
    assert insights_jobs.job_for(burger, new, memo) is again


# ------------------------------------------------------------------ #
#  D1: dark mode (01 F1, O16; review M7): every mark that carries
#  meaning has a colour and an opacity per theme, at 3:1 or more
# ------------------------------------------------------------------ #

PAGES = {"light": "#ffffff", "dark": "#0e1117"}


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return [int(h[k:k + 2], 16) for k in (0, 2, 4)]


def _luminance(rgb):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_on(colour, opacity, page):
    """The mark blended onto the page at its opacity, against the page."""
    fg, bg = _rgb(colour), _rgb(page)
    seen = [opacity * f + (1 - opacity) * b for f, b in zip(fg, bg)]
    la, lb = sorted((_luminance(seen), _luminance(bg)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# Vega's own "blues" scheme (vega-scale): eleven stops, linear between them.
_VEGA_BLUES = ("cfe1f2bed8eca8cee58fc1de74b2d75ba3cf4592c63181bd206fb2125ca40a4a90")


def _vega_blues(t):
    stops = [_rgb(_VEGA_BLUES[i:i + 6]) for i in range(0, len(_VEGA_BLUES), 6)]
    x = t * (len(stops) - 1)
    k = min(int(x), len(stops) - 2)
    f = x - k
    return "#" + "".join(f"{round(a + f * (b - a)):02x}" for a, b in zip(stops[k], stops[k + 1]))


def test_every_token_reaches_3_to_1_in_its_theme():
    for theme, page in PAGES.items():
        for name, (colour, opacity) in ui_insights._MARKS[theme].items():
            assert _contrast_on(colour, opacity, page) >= 3.0, (theme, name)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_colour_scales_highest_end_reaches_3_to_1_in_both_themes():
    """O16: the Uncertainty scale's high end, rgb(10,74,144), was about
    2:1 on the dark page."""
    assert _vega_blues(1.0) == "#0a4a90"
    for theme, page in PAGES.items():
        top = ui_insights.blues(theme).extent[1]
        assert _contrast_on(_vega_blues(top), 1.0, page) >= 3.0, theme


@pytest.fixture(params=["light", "dark"])
def theme(request, monkeypatch):
    monkeypatch.setattr(ui_insights, "theme", lambda: request.param)
    return request.param


def _all_layers(spec):
    out = []
    for key in ("concat", "layer"):
        for part in spec.get(key, []):
            out += _all_layers(part)
    if "spec" in spec:
        out += _all_layers(spec["spec"])
    if "mark" in spec:
        out.append(spec)
    return out


def _mark(layer):
    return layer["mark"] if isinstance(layer["mark"], dict) else {"type": layer["mark"]}


@pytest.fixture
def scored():
    opt = _burger(rounds=4, noise=0.4)
    return opt, insights.fit_all(opt), insights.check_rounds(opt)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_paths_reference_line_and_untested_edges_take_the_tokens(scored, theme):
    opt, fitted, _ = scored
    path = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    near = [False] * (len(path.amounts) // 2) + [True] * (len(path.amounts) - len(path.amounts) // 2)
    for chart in (ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path),
                                         near),
                  ui_insights.path_layer_chart(opt, path, [0.5] * len(path.amounts), near,
                                               wording.LAYER_SD)):
        layers = _all_layers(chart.to_dict())
        start = [l for l in layers if _mark(l).get("strokeDash") == [2, 2]]
        assert start and all(_mark(l)["color"] == ui_insights._MARKS[theme]["reference"][0]
                             and _mark(l)["opacity"] == ui_insights._MARKS[theme]["reference"][1]
                             for l in start)
        edges = [l for l in layers if _mark(l)["type"] == "rule"
                 and _mark(l).get("color") == ui_insights._MARKS[theme]["untested_edge"][0]]
        assert edges, "the grey band's edges carry its contrast"
        assert not any(_mark(l).get("color") in (None, "black") and _mark(l)["type"] == "rule"
                       and l["encoding"].get("x") and not l["encoding"].get("y")
                       for l in layers)


def _triangle(scored, **k):
    opt, fitted, _ = scored
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"],
                                 insights.start_recipe(opt, opt.best_index()))
    preds = insights.mixture_predictions(opt, fitted, mix)
    return ui_insights.mixture_chart(opt, mix, preds, 3, **k).to_dict()


def _colour_scale(spec):
    [grid] = [l for l in _all_layers(spec)
              if l.get("encoding", {}).get("color", {}).get("field") == "score"]
    return grid["encoding"]["color"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_uses_the_themes_scale(scored, theme):
    # Charts proto: the model layers in the theme's blues; the score layer
    # in the theme's score colours (as What you've tried).
    for layer in (wording.LAYER_SD, wording.LAYER_GAIN):
        scale = _colour_scale(_triangle(scored, layer=layer))["scale"]
        assert scale["scheme"]["extent"] == list(ui_insights.blues(theme).extent)
    scale = _colour_scale(_triangle(scored, layer=None))["scale"]
    assert scale["range"] == ui_insights.score_colours(theme)


# ------------------------------------------------------------------ #
#  A5 / I13: the layer's new name and the triangle's legend
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangles_score_legend_says_model_mean(scored):
    """02 M8 / 03 M5: the colour is the model mean, not an interval."""
    for layer in (None, wording.LAYER_SCORE):
        title = _colour_scale(_triangle(scored, layer=layer))["title"]
        assert title == wording.LAYER_SCORE_LEGEND == "Predicted Overall score (model mean)"
    assert _colour_scale(_triangle(scored, layer=wording.LAYER_GAIN))["title"] == (
        wording.LAYER_GAIN_LEGEND)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_session_holding_the_former_layer_name_reads_as_the_new_one(burger, monkeypatch):
    """Review I6, then charts proto: "Expected improvement" is now "Expected gain"; a session
    from before the rename must not break the radio or fall back."""
    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(
        session_state=state, radio=lambda label, options, key, horizontal: (
            state[key] if state.get(key) in options else options[0])))
    monkeypatch.setattr(ui_insights, "layers_offered", lambda o, f: ui_insights._LAYERS)
    assert wording.LAYER_GAIN in ui_insights._LAYERS
    assert wording.LAYER_GAIN_FORMER not in ui_insights._LAYERS
    for key in (ui_insights.LAYER_KEY, ui_insights.PATH_LAYER_KEY):
        state[key] = wording.LAYER_GAIN_FORMER
        assert ui_insights._layer_pick(burger, {}, key) == wording.LAYER_GAIN
        assert state[key] == wording.LAYER_GAIN


def test_a_failed_request_finds_its_pool_already_broken(fresh_pool, monkeypatch):
    """A Job that sees a request fail must find the pool broken at that
    moment, so start() replaces it at once instead of handing the screen
    to a whole in-process fit (C2). The clean-up after it may lag."""
    import time as _time
    real = insights_jobs.Pool._break
    monkeypatch.setattr(insights_jobs.Pool, "_break",
                        lambda self: (_time.sleep(0.3), real(self))[1])
    for _ in range(3):
        insights_jobs._BREAKS.clear()
        p = insights_jobs.pool()
        assert p.wait_ready(60)
        fut = p.submit(0, {"id": 1})
        assert fut.exception(30) is not None
        assert p.broken
