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
