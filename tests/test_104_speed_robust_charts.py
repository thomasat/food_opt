"""0.8.0 e2e fixes, W3 (spec 2026-10-01 §C, §D; review I11, M5-M7, I13):
speed, robustness and charts.

- C3: a cache entry holding a non-finite number is dropped and refitted.
- C1: the next Round's expected improvement is kept beside the project.
- C1: the progress line counts only what is really being fitted.
- C2: a broken pool is replaced on the next need, up to a cap.
- D1: the charts' meaning-carrying marks have theme tokens at 3:1.
- A5 / I13: the layer's new name and the triangle's legend."""
import json
import math
import os

import pytest

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
#  C1: the next Round's expected improvement kept beside the project
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


def test_insights_writes_the_figure_beside_the_project(burger, jobs):
    at = _open(_app(burger.project_name))
    assert not at.exception
    saved = insights_cache.load(burger.project_name)
    assert saved.gain is not None and saved.gain[0] == bo_views.next_gain_key(burger)
    assert saved.gain[1] == at.session_state[ui_insights.GAIN_KEY][1]


def test_a_reopened_project_shows_the_figure_without_a_fit_or_a_search(burger, jobs,
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


def test_a_new_seed_is_never_served_the_old_figure(burger, jobs, monkeypatch):
    """A discarded Round moves the next seed with the data unchanged."""
    _open(_app(burger.project_name))
    burger.next_formulation_no += 1
    burger.save()
    bo_views._CACHE.clear()
    searched = []
    real = bo_views.next_optimum
    monkeypatch.setattr(bo_views, "next_optimum", lambda *a: searched.append(1) or real(*a))
    at = _open(_app(burger.project_name))
    assert not at.exception and searched == [1]
    assert insights_cache.load(burger.project_name).gain[0] == bo_views.next_gain_key(
        at.session_state["optimizer"])


def test_make_a_round_reads_a_saved_figure_without_working_it_out(burger, jobs, monkeypatch):
    """E3 (fit=False) shows only what is worked out: a saved figure is."""
    _open(_app(burger.project_name))
    bo_views._CACHE.clear()
    _no_search(monkeypatch)
    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(session_state=state))
    got = ui_insights.latest_gain(burger, fit=False)
    assert got is not None and got.gain is not None
