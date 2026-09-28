import itertools
import math
import copy
import torch
import pytest
from food_bo import FoodOptimizer
import insights


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # FoodOptimizer.tell()/set_amount_unit() call self.save(), and the default
    # LocalStorage backend writes "<project_name>.pkl" into the current
    # working directory (see tests/test_food_bo.py's own `opt` fixture for the
    # same pattern). Isolate every test here into its own tmp_path so this
    # file does not litter the worktree root with test .pkl files.
    monkeypatch.chdir(tmp_path)


_burger_no = itertools.count()


def _burger(rounds=4, per_round=3, noise=0.0, seed=0, fill=True, shift=0.0):
    """A small mixture project with a known smooth truth.
    Ingredients Pea 4–8 g, Oil 2–6 g, Water fills to 100 g (fill=True) or is
    an ordinary ingredient 80–94 g (fill=False, total 100 g, Piepel paths).
    Firmness (target 6 on 0–10) rises with Pea; Juiciness (max, 0–10) rises
    with Oil. `shift` is added to Firmness in the LAST round only (a truth
    that moves, for miss tests). Rounds are numbered 1..rounds."""
    g = torch.Generator().manual_seed(seed)
    # A fresh, distinct project name per call: FoodOptimizer.__init__ loads
    # any existing file of the same name from storage (LocalStorage writes
    # "<project_name>.pkl" into the cwd on every tell()/set_amount_unit()),
    # so two calls sharing one name within a test would have the second
    # silently resume the first's saved history instead of starting empty.
    opt = FoodOptimizer(f"insights_test_{next(_burger_no)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    if fill:
        opt.add_ingredient("Water", 0, 100)
        opt.set_formulation_total(100)
        opt.set_balance("Water")
    else:
        opt.add_ingredient("Water", 80, 94)
        opt.set_formulation_total(100)
    opt.add_objective("Firmness", 60, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 40, goal="max", min_val=0, max_val=10)
    no = 0
    for r in range(1, rounds + 1):
        for k in range(per_round):
            u = torch.rand(2, generator=g).tolist()
            pea, oil = 4 + 4 * u[0], 2 + 4 * u[1]
            recipe = {"Pea": pea, "Oil": oil}
            if not fill:
                recipe["Water"] = 100 - pea - oil
            eps = (torch.randn(2, generator=g) * noise).tolist()
            firm = 2 + 0.9 * pea + eps[0] + (shift if r == rounds else 0.0)
            juicy = 1 + 1.2 * oil + eps[1]
            no += 1
            opt.tell(opt.fill_formulas(recipe) if fill else recipe,
                     {"Firmness": min(10, max(0, firm)),
                      "Juiciness": min(10, max(0, juicy))},
                     formulation_no=no, batch_no=r)
    return opt


def test_measurement_models_use_only_rows_with_that_measurement():
    opt = _burger(rounds=3)
    # a partial row: Juiciness only
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Juiciness": 5.0},
             formulation_no=99, batch_no=4)
    rows = list(range(len(opt.Y_history)))
    firm = insights.fit(opt, "Firmness", rows)
    juicy = insights.fit(opt, "Juiciness", rows)
    assert firm.n_rows == len(rows) - 1
    assert juicy.n_rows == len(rows)
    assert firm.model is not None and firm.error is None


def test_fewer_than_five_rows_gives_no_model():
    opt = _burger(rounds=1, per_round=4)
    f = insights.fit(opt, "Firmness", list(range(4)))
    assert f.model is None and f.error is None and f.n_rows == 4


def test_prediction_band_is_clipped_to_the_range():
    opt = _burger(rounds=3, noise=0.5)
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    preds = insights.predict(opt, f, [opt.X_history[0], opt.X_history[1]])
    for p in preds:
        assert 0.0 <= p.low <= p.mean <= p.high <= 10.0


def test_score_output_uses_utility_ceiling():
    opt = _burger(rounds=2)
    names = [n for n, _, _ in insights.outputs(opt)]
    assert names == ["Firmness", "Juiciness", insights.SCORE]
    assert insights.outputs(opt)[-1][2] == pytest.approx(opt.utility_ceiling())


def test_orphan_measurement_keys_are_ignored():
    opt = _burger(rounds=2)
    opt.results_history[0]["Old taste"] = 3.0   # a deleted measurement's key
    assert [n for n, _, _ in insights.outputs(opt)] == ["Firmness", "Juiciness", insights.SCORE]
    assert set(insights.fit_all(opt)) == {"Firmness", "Juiciness", insights.SCORE}


def test_expert_settings_still_fit():
    opt = _burger(rounds=3)
    opt.bo_config = dict(opt.DEFAULT_BO_CONFIG) if hasattr(opt, "DEFAULT_BO_CONFIG") else {}
    opt.bo_config.update({"noise": "fixed_tiny"})
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    assert f.model is not None, f.error


def test_a_failed_fit_is_isolated(monkeypatch):
    opt = _burger(rounds=3)
    real = opt._build_gp
    def boom(X, Y, dim, tf):
        if Y.shape[0] and float(Y.max()) <= 10.0:   # measurement fits only
            raise RuntimeError("no fit")
        return real(X, Y, dim, tf)
    monkeypatch.setattr(opt, "_build_gp", boom)
    fitted = insights.fit_all(opt)
    assert fitted["Firmness"].model is None and "no fit" in fitted["Firmness"].error
    assert fitted[insights.SCORE].model is not None


def test_insights_leave_the_project_and_rng_untouched():
    opt = _burger(rounds=3)
    before = opt.export_json()
    rng = torch.get_rng_state().clone()
    fitted = insights.fit_all(opt)
    insights.predict(opt, fitted["Firmness"], [opt.X_history[0]])
    assert opt.export_json() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_insights_do_not_change_the_next_suggestion():
    a, b = _burger(rounds=3), _burger(rounds=3)
    insights.fit_all(b)
    assert a.ask(2) == b.ask(2)


def test_data_key_changes_with_results_and_settings():
    opt = _burger(rounds=2)
    k1 = insights.data_key(opt)
    opt.results_history[0]["Firmness"] = 1.0
    assert insights.data_key(opt) != k1
