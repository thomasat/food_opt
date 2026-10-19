import itertools
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


def test_score_output_uses_utility_ceiling():
    opt = _burger(rounds=2)
    names = [n for n, _, _ in insights.outputs(opt)]
    assert names == ["Firmness", "Juiciness", insights.SCORE]
    assert insights.outputs(opt)[-1][2] == pytest.approx(opt.utility_ceiling())


def test_data_key_changes_with_results_and_settings():
    opt = _burger(rounds=2)
    k1 = insights.data_key(opt)
    opt.results_history[0]["Firmness"] = 1.0
    assert insights.data_key(opt) != k1


def test_resized_round_is_brought_back_to_the_total():
    opt = _burger(rounds=2)
    big = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    big = {k: v * 5 for k, v in big.items()}          # made at 500 g
    opt.tell(big, {"Firmness": 6.0, "Juiciness": 5.0}, formulation_no=70, batch_no=3)
    s = insights.start_recipe(opt, len(opt.Y_history) - 1)
    assert s["Pea"] == pytest.approx(6.0) and s["Oil"] == pytest.approx(4.0)


def test_same_fixed_row_requires_exact_match():
    # RULING (fix round 1, finding 3): a fixed row's tolerance is 0 (its
    # Lowest equals its Highest, so the usual "0.5 % of the span" rule would
    # divide by zero) -- decided via opt.is_fixed rather than re-deriving
    # "hi == lo" locally. `insights._same` is exercised directly because
    # `repeat_groups`' own pipeline (via `start_recipe` -> `_recorded_recipe`
    # -> decode) always snaps a fixed row back to its one allowed amount, so
    # two real formulations could never actually disagree on it -- there is
    # no way to reach this branch with a "slightly different" value through
    # the public entry point.
    opt = _burger(rounds=2)
    # See test_path_choices_premixes_and_process for why Salt is added with
    # room first and folded into the total's own limit before being fixed.
    opt.add_ingredient("Salt", 0, 5)
    opt.set_formulation_total(101)
    opt.set_formulation_total(100)
    opt.add_ingredient("Salt", 1, 1)               # fixed: Lowest == Highest
    a = {**opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), "Salt": 1.0}
    b = {**opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), "Salt": 1.0001}
    assert not insights._same(opt, a, b)


# ------------------------------------------------------------------ #
#  Final fix wave D, E, F
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Final fix wave A (part 2): the likely range is calibrated by 5-fold
#  cross-validation, refitting the model on each fold
# ------------------------------------------------------------------ #


