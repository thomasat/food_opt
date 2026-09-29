"""Insights wave C, computation for "What happens when three ingredients
trade places?": which rows are offered, the triangle's geometry (implied
bounds, L-pseudo components), the limit check at every grid point, the
tested formulations drawn on it, and its predictions. Nothing here may
write to the project or move the RNG."""
import itertools
import pickle

import pytest
import torch

import insights
from food_bo import CLOSE_TO_THE_BEST, FoodOptimizer
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # tell() saves "<project_name>.pkl" into the cwd.
    monkeypatch.chdir(tmp_path)


_no = itertools.count()


def _four(rows=()):
    """Pea 4–8, Oil 2–6, Starch 1–21 and Water 60–93 g, no Fill to total,
    Formulation total 100 g, one measurement. `rows` are (Pea, Oil, Starch)
    recorded fully measured; Water is what is left of 100 g."""
    opt = FoodOptimizer(f"four_{next(_no)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Starch", 1, 21)
    opt.add_ingredient("Water", 60, 93)
    opt.set_formulation_total(100)
    opt.add_objective("Taste", 100, goal="max", min_val=0, max_val=30)
    for no, (pea, oil, starch) in enumerate(rows, start=1):
        opt.tell({"Pea": pea, "Oil": oil, "Starch": starch, "Water": 100 - pea - oil - starch},
                 {"Taste": 2 * pea + oil}, formulation_no=no, batch_no=1)
    return opt


def _start(opt, **amounts):
    return opt.fill_formulas(dict(amounts))


def _allowed(mix):
    return [p for p in mix.points if p.recipe is not None]


def _state(opt):
    return pickle.dumps({k: v for k, v in opt.__dict__.items() if k != "_forms_cache"})


# ------------------------------------------------------------------ #
#  Which rows are offered
# ------------------------------------------------------------------ #

def test_the_fill_row_is_offered_with_the_varying_ingredients():
    assert insights.mixture_choices(_burger(rounds=0)) == ["Pea", "Oil", "Water"]
    assert insights.mixture_choices(_burger(rounds=0, fill=False)) == ["Pea", "Oil", "Water"]


def test_no_formulation_total_nothing_is_offered():
    opt = FoodOptimizer("free")
    for name in ("Pea", "Oil", "Starch"):
        opt.add_ingredient(name, 0, 10)
    assert insights.mixture_choices(opt) == []


def test_process_settings_and_fixed_rows_are_not_offered():
    opt = _four()
    opt.add_process_parameter("Temp", 150, 200, baseline=170, unit="°C")
    opt.add_ingredient("Starch", 3, 3)
    assert insights.mixture_choices(opt) == ["Pea", "Oil", "Water"]


def test_without_a_fill_row_an_ingredient_a_calculation_reads_is_not_offered():
    opt = _four()
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 0.5 * Pea")
    assert insights.mixture_choices(opt) == ["Oil", "Starch", "Water"]


def test_with_a_fill_row_an_ingredient_a_calculation_reads_is_offered():
    opt = _burger(rounds=0)
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 2 * Pea")
    assert insights.mixture_choices(opt) == ["Pea", "Oil", "Water"]


def test_a_portioned_premix_is_offered_and_draws():
    opt = _burger(rounds=0)
    opt.add_premix("Spice mix", "portioned")
    opt.set_premix_parts("Spice mix", [{"name": "Pepper", "share": 50},
                                       {"name": "Cumin", "share": 50}])
    opt.add_ingredient("Spice mix", 1, 5)
    assert "Spice mix" in insights.mixture_choices(opt)
    s = _start(opt, Pea=6.0, Oil=4.0, **{"Spice mix": 3.0})
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Spice mix"], s)
    assert mix.refusal is None and mix.start_breaks is None
    assert _allowed(mix)
    for p in _allowed(mix):
        assert p.recipe["Pea"] + p.recipe["Oil"] + p.recipe["Spice mix"] == pytest.approx(13.0)


# ------------------------------------------------------------------ #
#  Geometry: implied bounds and the L-pseudo triangle (B1)
# ------------------------------------------------------------------ #

def test_implied_bounds_and_the_span():
    opt = _burger(rounds=0, fill=False)                  # Pea 4–8, Oil 2–6, Water 80–94
    s = _start(opt, Pea=6.0, Oil=4.0, Water=90.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert mix.total == pytest.approx(100.0)
    assert mix.lower == pytest.approx([4.0, 2.0, 86.0])   # Water: 100 − 8 − 6
    assert mix.upper == pytest.approx([8.0, 6.0, 94.0])
    assert mix.span == pytest.approx(8.0)


def test_the_grid_covers_only_what_each_can_take_and_keeps_its_density():
    """Review fixes §5.1 / §9 I10: the grid is built over the region where
    each of the three stays within its implied Highest (U*), not the whole
    L-pseudo triangle, whose corner (12 g of Pea) lies past Pea's 8 g. Half
    the triangle is left, so the grid is twice as fine: 1/60."""
    opt = _burger(rounds=0, fill=False)
    s = _start(opt, Pea=6.0, Oil=4.0, Water=90.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert mix.steps == 60
    assert len(mix.points) > 496
    for p in mix.points:
        assert sum(p.x) == pytest.approx(1.0)
        assert sum(p.amounts) == pytest.approx(100.0)
        for k in range(3):
            assert p.amounts[k] == pytest.approx(mix.lower[k] + p.x[k] * mix.span)
            assert p.amounts[k] <= mix.upper[k] + 1e-9
    assert mix.top == pytest.approx([8.0, 6.0, 94.0])
    for k in range(3):
        assert max(p.amounts[k] for p in mix.points) == pytest.approx(mix.top[k])


def test_every_coloured_point_keeps_every_limit_and_the_sum():
    opt = _four()
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    names = ["Pea", "Oil", "Water"]
    mix = insights.mixture_slice(opt, names, s)
    allowed = _allowed(mix)
    assert 0 < len(allowed) < len(mix.points)
    for p in allowed:
        assert opt._check_constraints(p.recipe)
        assert sum(p.recipe[n] for n in names) == pytest.approx(mix.total)
        assert p.recipe["Starch"] == pytest.approx(5.0)       # every other row stays
        assert p.broken is None
    for p in mix.points:
        if p.recipe is None:
            r = dict(s, **dict(zip(names, p.amounts)))
            assert p.broken == insights._limits_broken(opt, opt.fill_formulas(r), names, s)
            assert p.broken is not None


def test_a_limit_between_the_three_shows_as_not_allowed():
    opt = _four()
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    qc = opt.quantity_constraints[-1]
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    over = [p for p in mix.points if p.amounts[0] + p.amounts[1] > 11.0 + 1e-6
            and p.amounts[0] <= 8 and p.amounts[1] <= 6]
    assert over and all(p.broken == opt.limit_text(qc) for p in over)


def test_the_fill_as_one_of_the_three_is_set_by_the_triangle():
    opt = _burger(rounds=0)                              # Water fills to 100 g
    s = _start(opt, Pea=6.0, Oil=4.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert mix.lower == pytest.approx([4.0, 2.0, 86.0])  # the fill's bounds: 0 to 100 g
    assert mix.follows == [] and mix.fill_moves is None
    for p in _allowed(mix):
        assert p.recipe["Water"] == pytest.approx(p.amounts[2])
        assert opt.ingredient_total(p.recipe) == pytest.approx(100.0)


def test_a_calculation_follows_and_the_fill_takes_up_the_change():
    opt = _four()
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 0.5 * Pea")
    opt.add_ingredient("Brine", 0, 100)
    opt.set_formula("Brine", "= rest")
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=70.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Starch"], s)
    assert mix.follows == ["Hydration"] and mix.fill_moves == "Brine"
    allowed = _allowed(mix)
    assert allowed
    for p in allowed:
        r = p.recipe
        assert r["Hydration"] == pytest.approx(0.5 * r["Pea"])
        assert r["Pea"] + r["Oil"] + r["Starch"] == pytest.approx(mix.total)
        assert opt.ingredient_total(r) == pytest.approx(100.0)
        assert r["Water"] == pytest.approx(70.0)


def test_the_fill_cannot_trade_with_a_row_a_calculation_reads():
    opt = _burger(rounds=0)
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 2 * Pea")
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], _start(opt, Pea=6.0, Oil=4.0))
    assert mix.refusal == insights.FILL_FOLLOWS
    assert mix.follows == ["Hydration"] and mix.points == []


def test_no_room_when_the_three_already_sit_at_their_lowest():
    opt = _four()
    s = _start(opt, Pea=4.0, Oil=2.0, Starch=1.0, Water=93.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Starch"], s)
    assert mix.span == pytest.approx(0.0, abs=1e-9)
    assert mix.refusal == insights.NO_ROOM and mix.points == []


def test_a_start_outside_its_own_range_still_draws():
    opt = _burger(rounds=0, fill=False)
    s = _start(opt, Pea=9.0, Oil=4.0, Water=87.0)       # Pea's Highest is 8
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert mix.start_breaks is None
    assert mix.upper[0] == pytest.approx(9.0)
    assert any(p.recipe is not None and p.amounts[0] > 8.0 + 1e-9 for p in mix.points)


def test_a_start_that_breaks_a_limit_says_so_and_draws_nothing():
    opt = _four()
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=8.0)
    qc = opt.quantity_constraints[-1]
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"],
                                 _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0))
    assert mix.start_breaks == opt.limit_text(qc)
    assert mix.points == []


def test_a_property_limit_with_missing_values_is_listed_as_not_checked():
    opt = _four()
    opt.add_property("Fat")
    opt.set_property_value("Oil", "Fat", 100.0)
    opt.add_constraint("Fat", max_val=5.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"],
                                 _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0))
    assert [prop for _, prop in mix.skipped] == ["Fat"]


# ------------------------------------------------------------------ #
#  Tested formulations on the triangle (I1)
# ------------------------------------------------------------------ #

def test_dots_are_fully_measured_formulations_near_the_slice():
    # Start: Pea 6, Oil 4, Starch 5, Water 85. S (Pea+Oil+Water) = 95, R = 8.
    opt = _four([(6.0, 4.0, 5.0),        # 1: the start itself
                 (5.0, 5.0, 6.0),        # 2: Starch +1 (0.05 of 20; S' off by 1/8)
                 (5.0, 5.0, 7.0)])       # 3: Starch +2 (0.1 of 20, but S' off by 2/8)
    opt.add_objective("Colour", 50, goal="max", min_val=0, max_val=10)
    opt.tell({"Pea": 7.0, "Oil": 3.0, "Starch": 5.0, "Water": 85.0}, {"Taste": 17.0},
             formulation_no=4, batch_no=1)                     # 4: waiting for Colour
    assert not opt.fully_measured(3)
    s = insights.start_recipe(opt, 0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert [d.formulation for d in mix.dots] == [1, 2]
    d2 = mix.dots[1]
    # Formulation 2's three amounts (5, 5, 84) sum to 94: drawn scaled to 95.
    assert d2.amounts == pytest.approx(tuple(v * 95 / 94 for v in (5.0, 5.0, 84.0)))
    assert sum(d2.amounts) == pytest.approx(95.0)
    for k in range(3):
        assert d2.x[k] == pytest.approx((d2.amounts[k] - mix.lower[k]) / mix.span)


def test_a_process_setting_farther_than_fifteen_hundredths_is_not_a_dot():
    opt = _four()
    opt.add_process_parameter("Temp", 150, 200, baseline=170, unit="°C")
    for no, temp in ((1, 170.0), (2, 177.0), (3, 178.0)):
        opt.tell({"Pea": 6.0, "Oil": 4.0, "Starch": 5.0, "Water": 85.0, "Temp": temp},
                 {"Taste": 16.0}, formulation_no=no, batch_no=1)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, 0))
    assert [d.formulation for d in mix.dots] == [1, 2]       # 7/50 = 0.14 in, 8/50 = 0.16 out


def test_a_choice_that_differs_is_never_a_dot():
    opt = _four()
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
    for no, pack in ((1, "Tray"), (2, "Pouch")):
        opt.tell({"Pea": 6.0, "Oil": 4.0, "Starch": 5.0, "Water": 85.0, "Package": pack},
                 {"Taste": 16.0}, formulation_no=no, batch_no=1)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, 0))
    assert [d.formulation for d in mix.dots] == [1]


def test_tested_near_here_is_within_fifteen_hundredths_of_a_dot():
    # One dot, at the start. With 40 steps a grid step is 0.2 g of Pea (span
    # 4 g): 6.6 g is 0.15 away (tested), 6.8 g is 0.2 away (not).
    opt = _four([(6.0, 4.0, 5.0)])
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, 0),
                                 steps=40)
    at = {(round(p.amounts[0], 6), round(p.amounts[1], 6)): p for p in mix.points}
    assert at[(6.6, 4.0)].tested
    assert not at[(6.8, 4.0)].tested
    assert CLOSE_TO_THE_BEST == 0.15


# ------------------------------------------------------------------ #
#  Predictions
# ------------------------------------------------------------------ #

def test_predictions_only_where_allowed_and_equal_to_predict():
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, opt.best_index()))
    preds = insights.mixture_predictions(opt, fitted, mix)
    assert len(preds) == len(mix.points)
    allowed = [(p, q) for p, q in zip(mix.points, preds) if p.recipe is not None]
    assert all(q is None for p, q in zip(mix.points, preds) if p.recipe is None)
    direct = insights.predict(opt, fitted[insights.SCORE], [opt._encode(p.recipe) for p, _ in allowed])
    for (_, q), d in zip(allowed, direct):
        assert (q.mean, q.low, q.high) == pytest.approx((d.mean, d.low, d.high), rel=1e-9, abs=1e-9)


def test_no_overall_score_model_no_predictions():
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    fitted[insights.SCORE] = insights.Fitted(insights.SCORE, None, None, 4, 0.0, 100.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, 0))
    assert insights.mixture_predictions(opt, fitted, mix) == []


def test_the_triangle_leaves_the_project_and_rng_untouched():
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    state, rng = _state(opt), torch.get_rng_state().clone()
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], insights.start_recipe(opt, 0))
    insights.mixture_predictions(opt, fitted, mix)
    assert _state(opt) == state and torch.equal(torch.get_rng_state(), rng)


# ------------------------------------------------------------------ #
#  Review fixes (2026-09-29) §5: only the allowed region, and the default
# ------------------------------------------------------------------ #

def test_a_whole_triangle_keeps_the_base_grid():
    """Three rows whose implied Highest is the L-pseudo corner itself: the
    whole triangle is the region, drawn at 1/30 as before."""
    opt = _four()
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Starch"], s)
    # Pea 4-8, Oil 2-6, Starch 1-21 around S = 15: R = 8, every U* = L* + R
    # only for Starch; Pea and Oil cap at half the span.
    assert mix.top[2] == pytest.approx(mix.lower[2] + mix.span)
    opt2 = FoodOptimizer(f"wide_{next(_no)}")
    opt2.set_amount_unit("g")
    for n in ("A", "B", "C"):
        opt2.add_ingredient(n, 0, 10)
    opt2.set_formulation_total(10)
    opt2.add_objective("Taste", 100, goal="max", min_val=0, max_val=30)
    wide = insights.mixture_slice(opt2, ["A", "B", "C"], _start(opt2, A=4.0, B=3.0, C=3.0))
    assert wide.steps == insights.GRID_STEPS and len(wide.points) == 496
    assert wide.top == pytest.approx([10.0, 10.0, 10.0])


def test_a_narrow_region_is_refined_but_capped():
    opt = _four()
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert insights.GRID_STEPS <= mix.steps <= insights.GRID_STEPS * insights.MAX_REFINE
    assert all(all(p.amounts[k] <= mix.upper[k] + 1e-9 for k in range(3)) for p in mix.points)


def test_a_tested_formulation_outside_the_region_is_not_a_dot():
    opt = _four([(6.0, 4.0, 5.0)])
    s = insights.start_recipe(opt, 0)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    for d in mix.dots:
        assert all(d.amounts[k] <= mix.upper[k] + 1e-6 for k in range(3))


def test_allowed_share_counts_the_trade_space_every_limit_keeps():
    opt = _four()
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    free = insights.allowed_share(opt, ["Pea", "Oil", "Water"], s)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    tighter = insights.allowed_share(opt, ["Pea", "Oil", "Water"], s)
    assert 0 < tighter < free <= 1
    stuck = _start(opt, Pea=4.0, Oil=2.0, Starch=1.0, Water=93.0)
    assert insights.allowed_share(opt, ["Pea", "Oil", "Starch"], stuck) == 0.0
