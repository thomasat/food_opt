"""Round 1 under limits so tight that too few points of the space-filling
sequence meet them (the web app's round-1 fallback, e2752b7
`_sample_feasible`).

The opening Round tries 2048 then 8192 Sobol points, each projected onto the
Formulation total; under tight amount and per-100 g limits too few of them
pass, and Generate used to stop with an error (none passed) or hand back a
short Round (some did). The Round is now topped up with points sampled
inside the region the limits leave (FoodOptimizer._sample_feasible), by the
web app's own chain. A Round the sequence fills is exactly what it was.

The chain is the web app's exactly (2000 steps of burn-in, every 64th
point, in chain order), and under limits this tight it moves slowly: the
points are valid and distinct, but can sit close together — in the second
seed measured, every row of the five within 3 % of its own range
(project name "other name"; "tight mozz" spreads up to 17 %)."""
import os

import pytest

import wording
import torch

import sample_projects as sp
import storage
import wording
from food_bo import FoodOptimizer

# name: (Lowest g, Highest g, sodium mg per 100 g, cost $ per 100 g)
MOZZ = {
    "Coconut oil": (150, 280, 0.0, 0.4),
    "Tapioca starch": (70, 150, 0.0, 0.2),
    "Soy protein isolate": (20, 60, 1000.0, 0.8),
    "Potato starch": (10, 40, 0.0, 0.2),
    "Salt": (8, 16, 39300.0, 0.05),
    "Kappa carrageenan": (4, 12, 1000.0, 2.0),
    "Guar gum": (2, 6, 0.0, 1.0),
    "Lactic acid": (2, 5, 0.0, 0.5),
}
TOTAL = 1002.0
WINDOWS = ((("Coconut oil", "Tapioca starch"), 330.0, 331.0),
           (("Soy protein isolate", "Potato starch"), 60.0, 61.0),
           (("Salt", "Kappa carrageenan"), 18.0, 18.5))
SODIUM_MAX, COST_MAX = 600.0, 0.32
FIXED = ("Soy lecithin", 5.0)
PREMIX_PARTS = {"Yeast": (1.0, 100.0, 2.0), "Garlic": (1.0, 50.0, 1.0)}


def tight(name="tight mozz", windows=WINDOWS):
    """A plant mozzarella at 1002 g: water fills to the total, one fixed
    row, one portioned pre-mix, two stepped settings, three 1 g sum
    windows, and sodium and cost limits per 100 g."""
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    for n, (lo, hi, _, _) in MOZZ.items():
        opt.add_ingredient(n, lo, hi)
    opt.add_ingredient(FIXED[0], FIXED[1], FIXED[1])
    opt.add_premix("Flavour mix", "portioned")
    opt.set_premix_parts("Flavour mix", [{"name": p, "share": s}
                                         for p, (s, _, _) in PREMIX_PARTS.items()])
    opt.add_ingredient("Flavour mix", 1, 5)
    opt.add_ingredient("Tap water", 0, TOTAL)
    for metric in ("Sodium", "Cost"):
        opt.add_property(metric)
    for n, (_, _, na, cost) in MOZZ.items():
        opt.set_property_value(n, "Sodium", na)
        opt.set_property_value(n, "Cost", cost)
    for n in (FIXED[0], "Tap water"):
        opt.set_property_value(n, "Sodium", 0.0)
        opt.set_property_value(n, "Cost", 0.0 if n == "Tap water" else 1.0)
    for p, (_, na, cost) in PREMIX_PARTS.items():
        opt.set_property_value(p, "Sodium", na)
        opt.set_property_value(p, "Cost", cost)
    opt.add_process_parameter("Cook temperature", 78, 92, unit="°C", step=1)
    opt.add_process_parameter("Cook time", 6, 12, unit="min", step=0.5)
    opt.add_objective("Stretch", 50, goal="max", min_val=0, max_val=15)
    opt.set_formulation_total(TOTAL)
    opt.set_formula("Tap water", "= rest")
    opt.add_constraint("Sodium", max_val=SODIUM_MAX)
    opt.add_constraint("Cost", max_val=COST_MAX)
    for names, lo, hi in windows:
        opt.add_quantity_constraint(list(names), min_val=lo, max_val=hi)
    return opt


def _per_100(recipe, column):
    figures = {n: v[column] for n, v in MOZZ.items()}
    figures[FIXED[0]] = 0.0 if column == 2 else 1.0
    figures["Tap water"] = 0.0
    share = sum(s for s, _, _ in PREMIX_PARTS.values())
    figures["Flavour mix"] = sum(
        s * (na if column == 2 else cost)
        for s, na, cost in PREMIX_PARTS.values()) / share
    mass = sum(recipe[n] for n in figures)
    return sum(recipe[n] * f for n, f in figures.items()) / mass


def assert_meets_every_limit(recipe, windows=WINDOWS):
    """Checked by hand, not through the app's own _check_constraints."""
    eps = 1e-6
    ingredients = list(MOZZ) + [FIXED[0], "Flavour mix", "Tap water"]
    total = sum(recipe[n] for n in ingredients)
    assert abs(total - TOTAL) <= 0.005 * TOTAL + eps
    for n, (lo, hi, _, _) in MOZZ.items():
        assert lo - eps <= recipe[n] <= hi + eps, (n, recipe[n])
    assert 1 - eps <= recipe["Flavour mix"] <= 5 + eps
    assert recipe[FIXED[0]] == pytest.approx(FIXED[1], abs=1e-9)
    assert recipe["Tap water"] >= -eps
    for names, lo, hi in windows:
        assert lo - eps <= sum(recipe[n] for n in names) <= hi + eps
    assert _per_100(recipe, 2) <= SODIUM_MAX + eps
    assert _per_100(recipe, 3) <= COST_MAX + eps
    assert 78 <= recipe["Cook temperature"] <= 92
    assert float(recipe["Cook temperature"]).is_integer()
    assert 6 <= recipe["Cook time"] <= 12
    assert (2 * recipe["Cook time"]).is_integer()


def _key(recipe):
    return tuple(round(float(v), 9) for _, v in sorted(recipe.items()))


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_sobol_point_meets_the_tight_limits(here):
    """The setup really is past the space-filling design: neither pool has a
    single point that meets every limit, so before this sampler Generate
    stopped with an error."""
    opt = tight()
    bounds = opt._get_bounds()
    for size in opt.COLD_START_POOLS:
        pool = opt._cold_start_pool(size, bounds, bounds.shape[1])
        assert not any(opt._check_constraints(rec) for rec in pool)


def test_without_the_sampler_round_1_fails_as_before(here, monkeypatch):
    """With the sampler switched off, round 1 is refused: the total is
    reachable, so the sentence blames the limits. Tap water has no range
    of its own typed, so it is not named."""
    opt = tight()
    monkeypatch.setattr(FoodOptimizer, "_sample_feasible",
                        lambda self, *a, **k: [])
    with pytest.raises(ValueError) as err:
        opt.ask(5)
    assert str(err.value) == wording.no_formulation_meets_limits("1002 g")


def test_tight_limits_get_a_full_round_inside_every_limit(here):
    opt = tight()
    recipes = opt.ask(5)
    assert len(recipes) == 5
    for recipe in recipes:
        assert opt._check_constraints(recipe)
        assert_meets_every_limit(recipe)
    assert len({_key(r) for r in recipes}) == 5
    # Brought onto the total wherever that kept every limit.
    assert sum(abs(sum(v for k, v in r.items() if k not in
                       ("Cook temperature", "Cook time")) - TOTAL) < 1e-6
               for r in recipes) >= 1


PARTIAL = ((("Coconut oil", "Tapioca starch"), 330.0, 332.0),
           (("Soy protein isolate", "Potato starch"), 60.0, 61.0))


def test_a_short_round_is_topped_up_and_keeps_its_sobol_points(
        tmp_path, monkeypatch):
    """With two windows of 2 g and 1 g, one point of the 8192 pool meets
    every limit: origin/main handed back a Round of 1 when 5 were asked
    for. That one stays first, exactly as it was, and four sampled points
    follow."""
    rounds = {}
    for sub, off in (("before", True), ("after", False)):
        os.makedirs(tmp_path / sub)
        monkeypatch.chdir(tmp_path / sub)
        with monkeypatch.context() as m:
            if off:
                m.setattr(FoodOptimizer, "_sample_feasible",
                          lambda self, *a, **k: [])
            rounds[sub] = tight("partial mozz", windows=PARTIAL).ask(5)
    assert len(rounds["before"]) == 1
    assert len(rounds["after"]) == 5
    assert rounds["after"][:1] == rounds["before"]
    assert len({_key(r) for r in rounds["after"]}) == 5
    for recipe in rounds["after"]:
        assert_meets_every_limit(recipe, windows=PARTIAL)


def test_the_round_is_reproducible(tmp_path, monkeypatch):
    """The same project in the same state samples the same Round, on a
    fresh copy in another folder too; and the global torch generator is
    left exactly where it was."""
    rounds = []
    for sub in ("a", "b"):
        os.makedirs(tmp_path / sub)
        monkeypatch.chdir(tmp_path / sub)
        opt = tight()
        before = torch.random.get_rng_state().clone()
        rounds.append(opt.ask(5))
        assert torch.equal(torch.random.get_rng_state(), before)
    assert rounds[0] == rounds[1]


def test_a_regenerate_samples_new_formulations(here):
    opt = tight()
    first = opt.ask(5)
    second = opt.ask(5, batch_no=opt.pending_batch_no,
                     discarded=[r['formulation'] for r in opt.pending_batch])
    assert len(second) == 5
    for recipe in second:
        assert_meets_every_limit(recipe)
    assert not ({_key(r) for r in first} & {_key(r) for r in second})


def test_a_truly_infeasible_setup_still_gets_the_clear_error(here):
    """Each window can be met on its own, but the four together cannot:
    the first two put those four rows at 390-392 g and the fourth asks for
    400 g. Nothing meets every limit, and the refusal says so at the
    total, which the amounts do reach."""
    four = ("Coconut oil", "Tapioca starch", "Soy protein isolate",
            "Potato starch")
    opt = tight("infeasible mozz", windows=WINDOWS + ((four, 400.0, 401.0),))
    with pytest.raises(ValueError) as err:
        opt.ask(5)
    assert str(err.value) == wording.no_formulation_meets_limits("1002 g")
    assert not opt.pending_batch


def test_ordinary_setups_never_reach_the_sampler(tmp_path, monkeypatch):
    """A project the Sobol pool can fill never calls the sampler, so its
    Round is the one it always was (the golden tests pin the numbers)."""
    monkeypatch.chdir(tmp_path)

    def refuse(self, *a, **k):
        raise AssertionError("the feasible sampler was reached")

    monkeypatch.setattr(FoodOptimizer, "_sample_feasible", refuse)
    store = storage.LocalStorage()
    sp.ensure(store, (sp.BURGER,))
    burger = FoodOptimizer(sp.BURGER, storage=store)
    assert len(burger.ask(5)) == 5

    plain = FoodOptimizer("plain")
    plain.set_amount_unit("g")
    plain.add_ingredient("Pea", 4, 8)
    plain.add_ingredient("Oil", 2, 6)
    plain.add_ingredient("Water", 0, 100)
    plain.add_objective("Firmness", 50, goal="max", min_val=0, max_val=10)
    plain.set_formulation_total(100)
    plain.set_formula("Water", "= rest")
    assert len(plain.ask(5)) == 5


# ------------------------------------------------------------------ #
#  The 0.9.1 units together: the round-1 sampler with the Fill to total
#  row's enforced range (algo-balance-range) and a row fixed after it
#  varied (algo-fixed-scale).
# ------------------------------------------------------------------ #

FILL_RANGE = (40.0, 40.1)
FLOUR_SUGAR = (30.0, 30.5)


def fill_tight(name="fill tight", water=FILL_RANGE, salt=None):
    """Water fills a 100 g formulation and is held inside 40-40.1 g, and
    Flour + Sugar sit inside a 0.5 g window. The other rows leave Water
    anywhere from 0 to 80 g (0 to 76 g with Salt), so both ends of its range
    bind. `salt` adds a Salt row of that (Lowest, Highest)."""
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", *water)
    opt.add_ingredient("Flour", 10, 40)
    opt.add_ingredient("Sugar", 5, 40)
    opt.add_ingredient("Fat", 5, 40)
    if salt is not None:
        opt.add_ingredient("Salt", *salt)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    opt.add_quantity_constraint(["Flour", "Sugar"], min_val=FLOUR_SUGAR[0],
                                max_val=FLOUR_SUGAR[1])
    return opt


def assert_fill_inside_its_range(recipe, low=FILL_RANGE[0], high=FILL_RANGE[1]):
    """Checked by hand: Water is what the other rows leave of 100 g, and
    that lands inside its own Lowest and Highest."""
    eps = 1e-6
    others = sum(v for k, v in recipe.items() if k != "Water")
    assert recipe["Water"] == pytest.approx(100.0 - others, abs=1e-6)
    assert low - eps <= recipe["Water"] <= high + eps, recipe["Water"]
    assert abs(sum(recipe.values()) - 100.0) <= 0.5 + eps
    assert (FLOUR_SUGAR[0] - eps <= recipe["Flour"] + recipe["Sugar"]
            <= FLOUR_SUGAR[1] + eps)
    assert 10 - eps <= recipe["Flour"] <= 40 + eps
    assert 5 - eps <= recipe["Sugar"] <= 40 + eps
    assert 5 - eps <= recipe["Fat"] <= 40 + eps


def _spy_sampler(monkeypatch):
    """[(asked for, handed back)] per call of _sample_feasible."""
    calls = []
    real = FoodOptimizer._sample_feasible

    def spy(self, n, *args, **kwargs):
        out = real(self, n, *args, **kwargs)
        calls.append((n, len(out)))
        return out

    monkeypatch.setattr(FoodOptimizer, "_sample_feasible", spy)
    return calls


def test_round_1_samples_inside_an_enforced_fill_range(here, monkeypatch):
    """No point of either Sobol pool keeps Water inside 40-40.1 g with the
    window met, so the whole Round is the sampler's, and every point keeps
    the Fill to total row inside its own range."""
    opt = fill_tight()
    bounds = opt._get_bounds()
    for size in opt.COLD_START_POOLS:
        pool = opt._cold_start_pool(size, bounds, bounds.shape[1])
        assert not any(opt._check_constraints(rec) for rec in pool)
    calls = _spy_sampler(monkeypatch)
    recipes = opt.ask(5)
    assert calls == [(5, 5)]
    assert len(recipes) == 5
    assert len({_key(r) for r in recipes}) == 5
    for recipe in recipes:
        assert opt._check_constraints(recipe)
        assert_fill_inside_its_range(recipe)


def test_without_its_range_the_sampler_leaves_the_fill_row_anywhere(
        here, monkeypatch):
    """The range really is what holds Water in the test above: given its
    default 0-100 g instead, the same sampler under the same window puts
    Water outside 40-40.1 g."""
    opt = fill_tight("fill wide", water=(0, 100))
    bounds = opt._get_bounds()
    recipes = opt._sample_feasible(5, bounds, bounds.shape[1])
    assert len(recipes) == 5
    assert any(not FILL_RANGE[0] <= r["Water"] <= FILL_RANGE[1]
               for r in recipes)


def test_limits_that_shut_out_the_fill_range_name_it(here, monkeypatch):
    """Fat at least 35 g leaves Water at most 35 g once the window is met.
    The other rows on their own still reach 40-40.1 g, so Generate gets as
    far as the sampler, which finds nothing, and the refusal names the
    Fill to total row's range."""
    opt = fill_tight("fill shut")
    opt.add_quantity_constraint(["Fat"], min_val=35.0, max_val=40.0)
    assert opt._fill_range_refusal() == ""
    calls = _spy_sampler(monkeypatch)
    with pytest.raises(ValueError) as refused:
        opt.ask(5)
    assert calls == [(5, 0)]
    assert str(refused.value) == wording.no_formulation_with_fill_range(
        "100 g", "Water", "40–40.1 g")
    assert not opt.pending_batch


def test_a_fill_range_that_cannot_bind_is_not_blamed(here, monkeypatch):
    """The tight mozzarella's Tap water keeps its 0-1002 g, which the other
    rows never take it past (they leave it 423-730 g). When the windows
    together cannot be met, the refusal blames the limits, not a range
    that was never in play."""
    four = ("Coconut oil", "Tapioca starch", "Soy protein isolate",
            "Potato starch")
    opt = tight("infeasible mozz", windows=WINDOWS + ((four, 400.0, 401.0),))
    assert opt._fill_range() == ("Tap water", 0.0, TOTAL)
    calls = _spy_sampler(monkeypatch)
    with pytest.raises(ValueError) as refused:
        opt.ask(5)
    assert calls == [(5, 0)]
    assert str(refused.value) == wording.no_formulation_meets_limits("1002 g")


def test_limits_nothing_meets_without_a_fill_row_are_said_as_limits(
        here, monkeypatch):
    """No Fill to total row, a total the allowed amounts reach (15-140 g),
    and limits that leave at most 85 g (Flour 79.9-80 g, Sugar at most
    5 g): the sampler finds nothing and the sentence blames the limits."""
    opt = FoodOptimizer("limits only")
    opt.set_amount_unit("g")
    opt.add_ingredient("Flour", 10, 80)
    opt.add_ingredient("Sugar", 5, 60)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.add_quantity_constraint(["Flour"], min_val=79.9, max_val=80.0)
    opt.add_quantity_constraint(["Sugar"], max_val=5.0)
    calls = _spy_sampler(monkeypatch)
    with pytest.raises(ValueError) as refused:
        opt.ask(3)
    assert calls == [(3, 0)]
    assert str(refused.value) == wording.no_formulation_meets_limits("100 g")


def test_a_total_out_of_reach_still_gets_the_adds_up_to_sentence(
        here, monkeypatch):
    """A stored project whose amounts no longer reach its 25 g total (Flour
    1-2 g and Sugar 5-10 g make 6-12 g): the sampler finds nothing, and the
    refusal is the old one about the size."""
    opt = FoodOptimizer("out of reach")
    opt.set_amount_unit("g")
    opt.add_ingredient("Flour", 10, 20)
    opt.add_ingredient("Sugar", 5, 10)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(25)
    opt._var_by_name("Flour")['bounds'] = (1.0, 2.0)
    assert opt.total_reach() == (6.0, 12.0)
    calls = _spy_sampler(monkeypatch)
    with pytest.raises(ValueError) as refused:
        opt.ask(3)
    assert calls == [(3, 0)]
    assert str(refused.value) == wording.no_formulation_reaches_total("25 g")
    assert "adds up to 25 g" in str(refused.value)


@pytest.mark.parametrize("fixed_at, frame", [(1.5, (0.0, 4.0)),
                                             (6.0, (0.0, 6.0))])
def test_a_row_fixed_after_it_varied_samples_with_the_fill_range(
        here, monkeypatch, fixed_at, frame):
    """Salt varies over 0-4 g in Round 1, then is fixed on the grid (inside
    and outside the range it varied over). Round 2 is still a cold start;
    no Sobol point meets the limits and the model is made to pick nothing,
    so the sampler fills the Round: in the frame Salt varied over (widened
    to its fixed amount), Salt pinned where it is fixed, and Water inside
    its own range."""
    opt = fill_tight("fixed fill", salt=(0, 4))
    first = opt.ask(3)
    for i, recipe in enumerate(first):
        assert_fill_inside_its_range(recipe)
        opt.tell(recipe, {"Taste": 3.0 + i})
    frame_grid = opt.ingredient_grid_frame()
    row = next(i for i in frame_grid.index
               if frame_grid.loc[i, wording.NAME_LABEL] == "Salt")
    frame_grid.loc[row, wording.LOWEST_LABEL] = fixed_at
    frame_grid.loc[row, wording.HIGHEST_LABEL] = fixed_at
    errors, _ = opt.apply_ingredient_grid(frame_grid)
    assert errors == [], errors
    salt = opt._var_by_name("Salt")
    assert salt['search_range'] == [0.0, 4.0]
    assert opt._search_bounds()[opt._search_columns()["Salt"]] == frame
    bounds = opt._get_bounds()
    for size in opt.COLD_START_POOLS:
        pool = opt._cold_start_pool(size, bounds, bounds.shape[1])
        assert not any(opt._check_constraints(rec) for rec in pool)
    monkeypatch.setattr(FoodOptimizer, "_ask_optimize",
                        lambda self, *a, **k: [])
    calls = _spy_sampler(monkeypatch)
    recipes = opt.ask(3)
    assert calls == [(3, 3)]
    assert len({_key(r) for r in recipes}) == 3
    for recipe in recipes:
        assert recipe["Salt"] == pytest.approx(fixed_at, abs=1e-9)
        assert opt._check_constraints(recipe)
        assert_fill_inside_its_range(recipe)
