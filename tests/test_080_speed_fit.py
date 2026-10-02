"""0.8.0 speed, Plan A Task 1: Insights' display fits use ftol 1e-6, one
fit function serves this process and the workers, and a fitted model can be
kept as JSON numbers and rebuilt without optimising (spec 2026-09-30 §10)."""
import json

import pytest
import torch

import food_bo
import insights
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _spy_fits(monkeypatch, module):
    """Every fit_gpytorch_mll call `module` makes, as its keyword arguments."""
    seen = []
    real = module.fit_gpytorch_mll

    def spy(mll, **kwargs):
        seen.append(kwargs)
        return real(mll, **kwargs)
    monkeypatch.setattr(module, "fit_gpytorch_mll", spy)
    return seen


def _rows(opt):
    return list(range(len(opt.Y_history)))


def test_display_fits_use_the_looser_tolerance(monkeypatch):
    opt = _burger(rounds=3)
    seen = _spy_fits(monkeypatch, insights)
    insights.fit(opt, "Firmness", _rows(opt))
    assert seen, "no display fit was made"
    assert all(k == {"optimizer_kwargs": {"options": {"ftol": 1e-6}}} for k in seen)


def test_the_suggestion_model_keeps_botorchs_defaults(monkeypatch):
    opt = _burger(rounds=3)
    seen = _spy_fits(monkeypatch, food_bo)
    opt.ask(1)
    assert seen == [{}]


def test_a_worker_fit_equals_the_fit_made_here():
    opt = _burger(rounds=3, noise=0.3)
    rows = _rows(opt)
    here = insights._fit_gp(opt, "Juiciness", rows)
    X, Y = insights.xy(opt, "Juiciness", rows)
    there = insights.fit_xy(insights.builder(getattr(opt, "bo_config", None)), X, Y)
    assert insights.hyper_state(here) == insights.hyper_state(there)


def test_a_fold_answer_equals_the_folds_made_here():
    opt = _burger(rounds=3, noise=0.3)
    use = _rows(opt)
    X, Y = insights.xy(opt, "Firmness", use)
    build = insights.builder(None)
    z = [v for kept, held in insights.cv_folds(opt, use)
         for v in insights.fold_z(build, X[kept], Y[kept], X[held], Y[held])]
    assert z == pytest.approx(insights._cv_z(opt, "Firmness", use), abs=1e-12)
    assert insights.widen_from(z) == pytest.approx(insights._widen(opt, "Firmness", use))


def test_a_model_from_the_memo_is_rebuilt_not_optimised(monkeypatch):
    opt = _burger(rounds=3, noise=0.3)
    memo = {}
    with insights.memo_in_use(memo):
        first = insights.fit(opt, "Firmness", _rows(opt))
    assert len(memo) == 1
    seen = _spy_fits(monkeypatch, insights)
    with insights.memo_in_use(memo):
        again = insights.fit(opt, "Firmness", _rows(opt))
    assert seen == []
    assert again.widen == first.widen
    X = [opt.X_history[i] for i in range(4)]
    for a, b in zip(insights.predict(opt, first, X), insights.predict(opt, again, X)):
        assert (a.mean, a.low, a.high) == pytest.approx((b.mean, b.low, b.high), abs=1e-9)


def test_the_memo_is_json_numbers_and_survives_a_round_trip():
    opt = _burger(rounds=3, noise=0.3)
    memo = {}
    with insights.memo_in_use(memo):
        first = insights.fit(opt, "Juiciness", _rows(opt))
    loaded = json.loads(json.dumps(memo))
    with insights.memo_in_use(loaded):
        again = insights.fit(opt, "Juiciness", _rows(opt))
    [a] = insights.predict(opt, first, [opt.X_history[0]])
    [b] = insights.predict(opt, again, [opt.X_history[0]])
    assert (a.mean, a.low, a.high) == pytest.approx((b.mean, b.low, b.high), abs=1e-9)


def test_without_a_memo_nothing_is_kept_and_every_fit_is_fresh(monkeypatch):
    opt = _burger(rounds=3)
    seen = _spy_fits(monkeypatch, insights)
    insights.fit(opt, "Firmness", _rows(opt))
    n = len(seen)
    insights.fit(opt, "Firmness", _rows(opt))
    assert len(seen) == 2 * n


def test_a_failed_fit_is_kept_as_its_error(monkeypatch):
    opt = _burger(rounds=3)
    memo = {}

    def boom(X, Y, dim, tf):
        raise RuntimeError("no fit")
    monkeypatch.setattr(opt, "_build_gp", boom)
    with insights.memo_in_use(memo):
        f = insights.fit(opt, "Firmness", _rows(opt))
    assert f.model is None and "no fit" in f.error
    assert list(memo.values()) == [{"error": "no fit"}]


def test_model_key_follows_rows_settings_and_salt(monkeypatch):
    opt = _burger(rounds=3)
    rows = _rows(opt)
    key = insights.model_key(opt, "Firmness", rows)
    assert insights.model_key(opt, "Firmness", rows) == key
    assert insights.model_key(opt, "Juiciness", rows) != key
    assert insights.model_key(opt, "Firmness", rows[:-1]) != key
    opt.results_history[0]["Firmness"] += 0.5
    assert insights.model_key(opt, "Firmness", rows) != key
    opt.results_history[0]["Firmness"] -= 0.5
    monkeypatch.setattr(insights, "MODEL_SALT", "another")
    assert insights.model_key(opt, "Firmness", rows) != key


def test_model_key_follows_each_setting_a_fit_depends_on(monkeypatch):
    """Review A M-2: the model settings, a row's Lowest/Highest, the fit
    options and the folds each change the key."""
    opt = _burger(rounds=3)
    rows = _rows(opt)
    key = insights.model_key(opt, "Firmness", rows)
    opt.bo_config = {"noise": "low"}
    assert insights.model_key(opt, "Firmness", rows) != key
    opt.bo_config = None
    assert insights.model_key(opt, "Firmness", rows) == key
    var = next(v for v in opt.variables if v["type"] == "continuous")
    lo, hi = var["bounds"]
    var["bounds"] = (lo, hi + 1.0)
    assert insights.model_key(opt, "Firmness", rows) != key
    var["bounds"] = (lo, hi)
    monkeypatch.setattr(insights, "FIT_OPTIONS", {"ftol": 1e-7})
    assert insights.model_key(opt, "Firmness", rows) != key
    monkeypatch.undo()
    monkeypatch.setattr(insights, "FOLDS", 4)
    assert insights.model_key(opt, "Firmness", rows) != key


def test_expert_model_settings_rebuild_from_the_memo():
    opt = _burger(rounds=3, noise=0.3)
    opt.bo_config = {"noise": "fixed_tiny"}
    memo = {}
    with insights.memo_in_use(memo):
        first = insights.fit(opt, "Firmness", _rows(opt))
        again = insights.fit(opt, "Firmness", _rows(opt))
    assert first.model is not None and again.model is not None
    [a] = insights.predict(opt, first, [opt.X_history[1]])
    [b] = insights.predict(opt, again, [opt.X_history[1]])
    assert a.mean == pytest.approx(b.mean, abs=1e-9)


def test_the_memo_is_this_threads_alone():
    import threading
    opt = _burger(rounds=3)
    memo, seen = {}, []
    with insights.memo_in_use(memo):
        t = threading.Thread(target=lambda: seen.append(insights._MEMO.get()))
        t.start()
        t.join()
    assert seen == [None]


def test_hyper_state_is_plain_numbers():
    opt = _burger(rounds=3)
    state = insights.hyper_state(insights._fit_gp(opt, "Firmness", _rows(opt)))
    blob = json.dumps(state)
    assert "tensor" not in blob
    assert all(isinstance(d, str) and hasattr(torch, d) for d, _ in state.values())
