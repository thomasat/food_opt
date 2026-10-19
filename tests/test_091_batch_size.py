"""0.8.0 BO views, Task 1 (spec 2026-09-30-bo-views §1, §8; review B1, B2,
I11, M1, M2, M16): a formulation made off the Formulation total — a Round
re-sized with the Batch size box — is learned at the total by every model,
the suggestion GP and Insights alike. Stored amounts never change."""
import pytest
import torch

import insights
from food_bo import FoodOptimizer


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _truth(r):
    return {"Firmness": 2 + 0.9 * r["Pea"], "Juiciness": 1 + 1.2 * r["Oil"]}


def _project(name, formula=False, fixed=False):
    """Pea 4-8, Oil 2-6, Water; a Formulation total of 100 g; with
    `formula` Salt is worked out as `= 0.5 + 0.05 × Pea` and Water varies
    (the case where total ÷ batch size would miss the total); else Water
    fills to the total. Six formulations told as Rounds of three."""
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    if formula:
        opt.add_ingredient("Salt", 0, 2)
    elif fixed:
        opt.add_ingredient("Salt", 1.5, 1.5)            # a fixed row
    opt.add_ingredient("Water", 0, 100 if not formula else 94)
    opt.add_objective("Firmness", 50, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    if formula:
        opt.set_formula("Salt", "= 0.5 + 0.05 * Pea")
    else:
        opt.set_formula("Water", "= rest")
    for k in range(6):
        r = {"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4}
        if formula:
            r["Water"] = 100 - r["Pea"] - r["Oil"] - (0.5 + 0.05 * r["Pea"])
        elif fixed:
            r["Salt"] = 1.5
        r = opt.fill_formulas(r)
        opt.tell(r, _truth(r), formulation_no=k + 1, batch_no=k // 3 + 1)
    return opt


def _record(opt, size=None, made=None):
    """Generate Round 3 (three formulations), re-size it to `size` (None:
    as generated) and record every row at the amounts made — or at
    `made[j]` for its j-th row."""
    opt.ask(3)
    if size is not None:
        opt.scale_round(size)
    for j, row in enumerate(list(opt.pending_batch)):
        rec = row["recipe"] if made is None else made[j]
        # The same results whatever size it was made at: a Round's results
        # do not depend on its batch size.
        no = row["formulation"]
        opt.tell(rec, {"Firmness": 3 + 0.3 * (no % 5), "Juiciness": 4 + 0.7 * (no % 3)},
                 formulation_no=no, batch_no=opt.pending_batch_no)
    return opt


def _spy(opt):
    seen = []
    real = opt._build_gp

    def spy(X, Y, dim, input_tf):
        seen.append(X.detach().clone())
        return real(X, Y, dim, input_tf)
    opt._build_gp = spy
    return seen


@pytest.mark.parametrize("formula", [False, True])
def test_a_round_made_at_500_g_is_learned_where_it_was_made(formula):
    """Review B2 test (b). The Round learned from 500 g rows is the Round
    as generated, to 1e-12 (×5 then ×0.2 is not exact in floats); and the
    suggestions equal those of the same project with that Round recorded
    at the Formulation total, allclose 1e-9. The twin records the amounts
    at the total exactly as model_recipe gives them: from the rows as
    generated instead, the 1e-15 input differences move the optimiser's
    answer by 1e-4 g (Water fills) to 0.35 g (a formula with a constant) —
    the multi-start search's sensitivity on a flat acquisition, not the
    rescaling's."""
    big = _record(_project(f"big_{formula}", formula), 500)
    generated = _record(_project(f"gen_{formula}", formula))
    at_total = [big.model_recipe(i) for i in range(6, 9)]
    small = _record(_project(f"small_{formula}", formula), made=at_total)
    assert [round(big.ingredient_total(r), 6) for r in big.recipe_history[6:]] == [500.0] * 3
    assert big.recipe_history[6] != small.recipe_history[6]          # stored as made
    got = torch.tensor(big.model_X(range(9)), dtype=torch.double)
    assert torch.allclose(got, torch.tensor(generated.X_history, dtype=torch.double),
                          rtol=0, atol=1e-12)
    seen_big, seen_small = _spy(big), _spy(small)
    a, b = big.ask(3), small.ask(3)
    assert torch.allclose(seen_big[0], seen_small[0], rtol=0, atol=1e-12)
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for key in x:
            assert x[key] == pytest.approx(y[key], abs=1e-9), key


def test_before_the_fix_the_model_saw_rows_far_outside_its_frame():
    """The finding (02 C1): normalised inputs near 5 on a 0-1 frame. Now
    every row the models learn sits inside it."""
    opt = _record(_project("frame"), 500)
    raw = insights.normalize(torch.tensor(opt.X_history, dtype=torch.double), opt._get_bounds())
    assert float(raw.max()) > 3
    X, _ = insights.xy(opt, insights.SCORE, list(range(9)))
    assert float(X.max()) <= 1 + 1e-9 and float(X.min()) >= -1e-9


def test_insights_places_a_resized_row_where_the_suggestions_learn_it():
    """One rule on both sides (review B1): at_total, and so start_recipe,
    'tested near' and every distance, agree with model_X."""
    opt = _record(_project("one_rule"), 500)
    for i in range(9):
        assert opt._encode(insights.start_recipe(opt, i)) == pytest.approx(
            opt.model_X([i])[0], abs=1e-9)


def test_a_row_inside_one_percent_is_not_rescaled():
    """1 % (twice the band), so a snap the limits refused, at the band's
    edge, never counts."""
    opt = _project("margin", formula=True)
    r = dict(opt.recipe_history[0])
    r["Water"] -= 0.9                                       # 99.1 g: 0.9 % off
    opt.tell(r, _truth(r), formulation_no=7, batch_no=3)
    assert opt.model_X([6])[0] is opt.X_history[6]
    r = dict(opt.recipe_history[0])
    r["Water"] -= 1.2                                       # 98.8 g: 1.2 % off
    opt.tell(r, _truth(r), formulation_no=8, batch_no=3)
    assert opt.model_X([7])[0] != opt.X_history[7]
    assert opt.ingredient_total(opt.model_recipe(7)) == pytest.approx(100.0)


def test_no_total_and_process_only_projects_are_untouched():
    """Review B2 test (c): the rows handed to the model ARE X_history's."""
    opt = FoodOptimizer("no_total")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Water", 80, 94)
    opt.add_objective("Firmness", 1, goal="max", min_val=0, max_val=10)
    for k in range(6):
        r = {"Pea": 4 + k * 0.5, "Water": 85.0}
        opt.tell(r, {"Firmness": r["Pea"]}, formulation_no=k + 1, batch_no=1)
    opt.ask(2)
    opt.scale_round(500)
    for row in list(opt.pending_batch):
        opt.tell(row["recipe"], {"Firmness": 5.0}, formulation_no=row["formulation"],
                 batch_no=opt.pending_batch_no)
    assert all(a is b for a, b in zip(opt.model_X(range(8)), opt.X_history))
    ferment = FoodOptimizer("ferment")
    ferment.add_process_parameter("Mixing time", 1, 10)
    ferment.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10)
    for k in range(3):
        ferment.tell({"Mixing time": 2.0 + k}, {"Acidity": 5.0}, formulation_no=k + 1)
    assert all(a is b for a, b in zip(ferment.model_X(range(3)), ferment.X_history))


def test_the_record_and_the_downloads_keep_the_amounts_as_made():
    """Review B2 test (d)."""
    opt = _record(_project("as_made"), 500)
    made = opt.recipe_history[6]
    assert opt.ingredient_total(made) == pytest.approx(500.0)
    csv = opt.history_csv()
    csv = csv if isinstance(csv, str) else csv.decode()
    assert f"{made['Pea']:.2f}" in csv or f"{made['Pea']}" in csv
    frame = opt.history_export_frame()
    assert any(abs(float(v) - made["Pea"]) < 0.01 for v in frame.to_numpy().ravel()
               if isinstance(v, (int, float)))


def test_edit_past_formulations_round_trips_at_the_amounts_as_made():
    """Review M16: the edit is seeded with the as-made amounts and the
    corrected row is learned at the total again after the save."""
    opt = _record(_project("edit"), 500)
    before = opt.model_X([6])[0]
    opt.edit_amounts(6, dict(opt.recipe_history[6]))       # saved unchanged
    assert opt.model_X([6])[0] == pytest.approx(before, abs=1e-12)
    corrected = dict(opt.recipe_history[6])
    corrected["Pea"] += 5.0                                  # +1 g at the total
    corrected = opt.fill_formulas(corrected, batch=500)
    opt.edit_amounts(6, corrected)
    learned = opt.model_recipe(6)
    assert opt.ingredient_total(learned) == pytest.approx(100.0)
    assert learned["Pea"] == pytest.approx(corrected["Pea"] / 5, abs=1e-9)


def test_the_first_resize_keeps_what_each_formulation_added_up_to():
    """Review B1: metadata for a later wave (projects with no total)."""
    opt = _project("sums")
    opt.ask(2)
    no = opt.pending_batch_no
    opt.scale_round(250)
    opt.scale_round(500)                                   # the first sums stay
    sums = opt.batch_generated_totals[no]
    assert sorted(sums) == sorted(r["formulation"] for r in opt.pending_batch)
    assert all(v == pytest.approx(100.0) for v in sums.values())


def test_the_metadata_survives_a_save_and_a_damaged_entry_is_dropped():
    opt = _project("meta")
    opt.ask(2)
    opt.scale_round(500)
    no = opt.pending_batch_no
    record = {"seed": 7, "rows": [1, 2, 3, 4, 5, 6], "n": 2, "regime": "warm", "key": "abc"}
    opt.record_generation(no, record)
    again = FoodOptimizer("meta")
    assert again.round_generation == {no: record}
    assert all(v == pytest.approx(100.0) for v in again.batch_generated_totals[no].values())
    state = opt.export_json()
    state["round_generation"]["99"] = {"seed": "x"}
    state["batch_generated_totals"]["98"] = "nonsense"
    other = FoodOptimizer("meta_copy")
    other.import_json(state)
    assert set(other.round_generation) == {no}
    assert set(other.batch_generated_totals) == {no}


def test_the_metadata_goes_with_its_round():
    opt = _project("undo")
    opt.ask(2)
    no = opt.pending_batch_no
    opt.scale_round(500)
    opt.record_generation(no, {"seed": 7, "rows": [1], "n": 2, "regime": "warm", "key": ""})
    opt.set_pending_batch(None)
    opt._prune_round_records()
    assert no not in opt.round_generation and no not in opt.batch_generated_totals


def test_the_cache_keys_follow_the_models_inputs():
    """Review I11: a chart drawn on the old inputs is never reused: its key
    (data_key) changes when the Formulation total changes."""
    opt = _record(_project("keys"), 500)
    key = insights.data_key(opt)
    opt.formulation_total = 120.0
    assert insights.data_key(opt) != key


# ------------------------------------------------------------------ #
#  Review Focus (plan): inputs the spec implies but does not name
# ------------------------------------------------------------------ #

def test_a_new_formulation_total_moves_every_row_to_it():
    """The Formulation total changed after Rounds were recorded: every row
    is learned at the new total, in its own proportions; the record keeps
    what was made; Generate still gives rows at the new total."""
    opt = _record(_project("new_total"))
    made = [dict(r) for r in opt.recipe_history]
    opt.set_formulation_total(120)
    for i in range(9):
        learned = opt.model_recipe(i)
        assert opt.ingredient_total(learned) == pytest.approx(120.0)
        assert learned["Pea"] / learned["Oil"] == pytest.approx(made[i]["Pea"] / made[i]["Oil"])
    assert opt.recipe_history == made
    # What the models learn is these amounts (at 120 g a row can sit past a
    # Highest set for 100 g: the model is then told where it really was).
    X, _ = insights.xy(opt, insights.SCORE, list(range(9)))
    want = insights.normalize(torch.tensor([opt._encode(opt.model_recipe(i)) for i in range(9)],
                                           dtype=torch.double), opt._get_bounds())
    assert torch.allclose(X, want)
    for r in opt.ask(2):
        assert opt.ingredient_total(r) == pytest.approx(120.0, rel=0.006)


def test_an_own_formulation_typed_at_the_rounds_batch_size_is_learned_at_the_total():
    """A formulation of the user's own added to a Round re-sized to 500 g,
    typed at 500 g (Water varies here, so nothing refills it to the
    default size), is learned where it was made, like the suggestions."""
    opt = _project("own_500", formula=True)
    opt.ask(2)
    opt.scale_round(500)
    no = opt.add_to_pending_batch({"Pea": 30.0, "Oil": 20.0, "Water": 448.0},
                                  note="Own formulation")
    row = next(r for r in opt.pending_batch if r["formulation"] == no)
    assert opt.ingredient_total(row["recipe"]) == pytest.approx(500.0)
    opt.tell(row["recipe"], {"Firmness": 5.0, "Juiciness": 5.0}, formulation_no=no,
             batch_no=opt.pending_batch_no)
    learned = opt.model_recipe(len(opt.recipe_history) - 1)
    assert opt.ingredient_total(learned) == pytest.approx(100.0)
    assert learned["Pea"] / learned["Oil"] == pytest.approx(1.5)
    assert learned["Pea"] == pytest.approx(6.0, abs=0.05)      # Salt's 0.5 g does not scale


# ------------------------------------------------------------------ #
#  Fix round 1 (T1 review I1, I2, M1-M4): the other doors, and the
#  metadata's edges
# ------------------------------------------------------------------ #

def _twins(name, fixed=False, waiting=False):
    """(big, small): Round 3 made at 500 g, and the same Round recorded at
    the amounts at the total model_recipe gives. With `waiting` the last
    row of Round 3 waits for Juiciness in both."""
    big = _project(f"{name}_big", fixed=fixed)
    big.ask(3)
    big.scale_round(500)
    rows = list(big.pending_batch)
    for j, row in enumerate(rows):
        no = row["formulation"]
        results = {"Firmness": 3 + 0.3 * (no % 5), "Juiciness": 4 + 0.7 * (no % 3)}
        if waiting and j == len(rows) - 1:
            del results["Juiciness"]
        big.tell(row["recipe"], results, formulation_no=no, batch_no=big.pending_batch_no)
    small = _project(f"{name}_small", fixed=fixed)
    small.ask(3)
    for j, row in enumerate(list(small.pending_batch)):
        no = row["formulation"]
        results = {"Firmness": 3 + 0.3 * (no % 5), "Juiciness": 4 + 0.7 * (no % 3)}
        if waiting and j == 2:
            del results["Juiciness"]
        small.tell(big.model_recipe(6 + j), results, formulation_no=no,
                   batch_no=small.pending_batch_no)
    return big, small


def test_a_fixed_rows_frame_is_the_rows_as_learned():
    """T1 review I1 (1): _search_bounds' `seen` reads model_X. A fixed Salt
    1.5 g recorded at 500 g as 7.5 g must not widen its frame to 7.5."""
    big, small = _twins("fixed", fixed=True)
    salt = [r["Salt"] for r in big.recipe_history[6:]]
    assert salt == pytest.approx([7.5] * 3)
    assert big._search_bounds() == small._search_bounds()
    seen_big, seen_small = _spy(big), _spy(small)
    big.ask(2)
    small.ask(2)
    assert torch.allclose(seen_big[0], seen_small[0], rtol=0, atol=1e-12)


def test_a_waiting_row_leaves_the_rest_learned_at_the_total():
    """T1 review I1 (2): the waiting branch of _ask_optimize (a row still
    missing a result) reads model_X too."""
    big, small = _twins("waiting", waiting=True)
    assert big.waiting_rows() and big.waiting_rows() == small.waiting_rows()
    seen_big, seen_small = _spy(big), _spy(small)
    big.ask(2)
    small.ask(2)
    assert seen_big[0].shape[0] == 8
    assert torch.allclose(seen_big[0], seen_small[0], rtol=0, atol=1e-12)


def test_the_models_settings_carry_the_total():
    """Planner ruling 1: the one-time invalidation that stands in for a
    MODEL_SALT bump."""
    opt = _project("settings")
    assert insights._model_settings(opt)["total"] == 100


def test_the_in_memory_key_follows_a_formula_edit():
    """T1 review M3: data_key hashes the rows as learned, so a fit made
    before a formula's constant changed is not served afterwards."""
    opt = _record(_project("formula_key", formula=True), 500)
    key = insights.data_key(opt)
    before = opt.model_X([6])[0]
    opt.set_formula("Salt", "= 1.5 + 0.05 * Pea")
    assert opt.model_X([6])[0] != before
    assert insights.data_key(opt) != key


def test_an_off_total_categorical_row_is_brought_to_the_total():
    """T1 review M2: at_total keeps a categorical row's choice, as
    model_recipe does, instead of failing on it."""
    opt = FoodOptimizer("categorical")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 2, 8)
    opt.add_ingredient("Water", 0, 100)
    opt.variables.append({"name": "Binder", "type": "categorical", "options": ["A", "B"]})
    opt.add_objective("Firmness", 1, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    recipe = {"Pea": 20.0, "Water": 480.0, "Binder": "B"}
    got = insights.at_total(opt, recipe)
    assert got["Binder"] == "B"
    assert got["Pea"] == pytest.approx(4.0) and got["Water"] == pytest.approx(96.0)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_an_infinite_number_in_the_metadata_is_dropped_not_fatal(bad):
    """T1 review M1: json reads Infinity and NaN; the entry is dropped and
    the copy still opens."""
    import json
    opt = _project("inf")
    opt.ask(2)
    opt.scale_round(500)
    no = opt.pending_batch_no
    first = opt.pending_batch[0]["formulation"]
    state = opt.export_json()
    state["round_generation"] = {str(no): {"seed": bad, "rows": [1], "n": 2,
                                           "regime": "warm", "key": ""}}
    state["batch_generated_totals"][str(no)][str(first)] = bad
    state = json.loads(json.dumps(state))
    other = FoodOptimizer("inf_copy")
    other.import_json(state)
    assert other.round_generation == {}
    assert first not in other.batch_generated_totals.get(no, {})


def test_a_regenerated_round_keeps_what_its_new_formulations_added_up_to():
    """T1 review M4: "Generate a different batch" keeps the Round's number
    with new formulation numbers; each keeps its sum at its first re-size."""
    opt = _project("regen")
    opt.ask(2)
    no = opt.pending_batch_no
    first = [r["formulation"] for r in opt.pending_batch]
    opt.scale_round(500)
    opt.ask(2, batch_no=no, discarded=first)
    again = [r["formulation"] for r in opt.pending_batch]
    assert opt.pending_batch_no == no and not set(again) & set(first)
    opt.scale_round(250)
    sums = opt.batch_generated_totals[no]
    assert set(again) <= set(sums)
    assert all(sums[n] == pytest.approx(100.0) for n in again)



@pytest.mark.parametrize("bad", [{"rows": "123"}, {"picks": ["a"]}, {"picks": "12"},
                                 {"rows": [1, "x"]}])
def test_a_malformed_generation_record_is_dropped_whole(bad):
    """Final review M-4: rows and picks are lists of numbers, or the record
    goes (the Round then reads recomputed)."""
    import food_bo
    record = {"seed": 7, "rows": [1, 2], "n": 2, "regime": "warm", "key": "k", "picks": [8, 9]}
    assert food_bo._generation_records_from({"3": record}) == {3: record}
    assert food_bo._generation_records_from({"3": {**record, **bad}}) == {}
