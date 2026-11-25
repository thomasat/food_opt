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
