"""Algorithm alignment with the web app: the scale a FIXED row is learned on.

The web app (origin/stable, e2752b7) deactivates a variable by pinning it
inside its own, unchanged bounds, so the GP normalizes the formulations made
while it varied on the same scale as before. The desktop fixes a row by
setting Lowest == Highest; it now remembers the range the row varied over
(`search_range`) and normalizes on that, widened to reach the fixed amount.
A row that never varied, and a row fixed by 0.9.0 or earlier (no memory),
keep the frame they had."""
import json

import pytest
import torch

import wording
from food_bo import FoodOptimizer


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _spy(opt):
    """Every normalized training input the suggestion GP is built on."""
    seen = []
    real = opt._build_gp

    def spy(X, Y, dim, input_tf):
        seen.append(X.detach().clone())
        return real(X, Y, dim, input_tf)
    opt._build_gp = spy
    return seen


SALT = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]


def _two_rounds(name):
    """Water 0-100 and Salt 0-10 g, both varied in Rounds 1 and 2."""
    opt = FoodOptimizer(name, robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Salt", 0, 10)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    for i, salt in enumerate(SALT):
        opt.tell({"Water": 20.0 + 10 * i, "Salt": salt},
                 {"Taste": 3.0 + 0.5 * i - 0.2 * salt})
    return opt


def _grid_fix(opt, name, value):
    """Fix a row the way the reader does: both boxes on the grid."""
    frame = opt.ingredient_grid_frame()
    row = next(i for i in frame.index if frame.loc[i, wording.NAME_LABEL] == name)
    frame.loc[row, wording.LOWEST_LABEL] = value
    frame.loc[row, wording.HIGHEST_LABEL] = value
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == [], errors


def test_a_row_fixed_at_round_3_is_learned_on_the_range_it_varied_over():
    opt = _two_rounds("fixed_r3")
    _grid_fix(opt, "Salt", 4.0)
    salt = opt._var_by_name("Salt")
    assert salt["bounds"] == (4.0, 4.0)
    assert salt["search_range"] == [0.0, 10.0]
    assert opt._search_bounds() == [(0.0, 100.0), (0.0, 10.0)]
    assert opt._get_fixed_features() == {1: pytest.approx(0.4)}
    seen = _spy(opt)
    batch = opt.ask(n_suggestions=2)
    assert all(r["Salt"] == pytest.approx(4.0) for r in batch)
    # The GP sees Salt on 0-10 g, as it did while it varied: not stretched
    # to the 1-6 g the history spans.
    assert torch.allclose(seen[0][:, 1],
                          torch.tensor(SALT, dtype=torch.double) / 10.0,
                          rtol=0, atol=1e-12)


def test_the_gp_inputs_do_not_jump_when_a_row_is_fixed():
    """The formulations already made are normalized exactly as in Round 2."""
    varying = _two_rounds("before_fix")
    seen_before = _spy(varying)
    varying.ask(n_suggestions=1)
    fixed = _two_rounds("after_fix")
    fixed.add_ingredient("Salt", 4.0, 4.0)
    seen_after = _spy(fixed)
    fixed.ask(n_suggestions=1)
    assert torch.allclose(seen_before[0], seen_after[0], rtol=0, atol=1e-12)


def test_a_fixed_amount_outside_the_remembered_range_widens_it():
    """The web app refuses that pin; the desktop lets the reader type it, so
    the frame reaches it rather than clamping the pin."""
    opt = _two_rounds("outside")
    opt.add_ingredient("Salt", 12.0, 12.0)
    assert opt._var_by_name("Salt")["search_range"] == [0.0, 10.0]
    assert opt._search_bounds()[1] == (0.0, 12.0)
    assert opt._get_fixed_features() == {1: pytest.approx(1.0)}
    assert all(r["Salt"] == pytest.approx(12.0) for r in opt.ask(n_suggestions=2))


def test_moving_a_fixed_row_keeps_the_memory_and_varying_again_drops_it():
    opt = _two_rounds("move_and_free")
    opt.add_ingredient("Salt", 4.0, 4.0)
    opt.add_ingredient("Salt", 5.0, 5.0)
    assert opt._var_by_name("Salt")["search_range"] == [0.0, 10.0]
    opt.add_ingredient("Salt", 2.0, 8.0)
    assert "search_range" not in opt._var_by_name("Salt")
    assert opt._search_bounds()[1] == (2.0, 8.0)
    # Fixed again: the range it varied over most recently is the one kept.
    opt.add_ingredient("Salt", 3.0, 3.0)
    assert opt._var_by_name("Salt")["search_range"] == [2.0, 8.0]


def test_a_process_setting_fixed_after_it_varied_keeps_its_range():
    opt = FoodOptimizer("setting", robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    for i, temp in enumerate([160.0, 170.0, 180.0]):
        opt.tell({"Water": 30.0 + i, "Cook temperature": temp}, {"Taste": 5.0 + i})
    opt.add_process_parameter("Cook temperature", 175, 175, unit="°C")
    assert opt._var_by_name("Cook temperature")["search_range"] == [150.0, 200.0]
    assert opt._search_bounds()[1] == (150.0, 200.0)


def test_a_never_varied_fixed_row_is_unchanged():
    """Added fixed and never given a range: no memory, and the frame is the
    one it always had — one unit wide when nothing it recorded spans more."""
    opt = _two_rounds("never_varied")
    opt.add_ingredient("Yeast", 2.0, 2.0, keep_lowest=True)
    yeast = opt._var_by_name("Yeast")
    assert "search_range" not in yeast
    lo, hi = opt._search_bounds()[2]
    # Encoded at 0 in the six formulations made before it existed.
    assert (lo, hi) == (0.0, 2.0)
    fresh = FoodOptimizer("never_varied_fresh", robust=False)
    fresh.set_amount_unit("g")
    fresh.add_ingredient("Water", 0, 100)
    fresh.add_ingredient("Salt", 3.0, 3.0)
    assert "search_range" not in fresh._var_by_name("Salt")
    assert fresh._search_bounds()[1] == (3.0, 4.0)


def test_a_row_fixed_by_090_keeps_the_stretched_frame():
    """A project saved by 0.9.0 has no memory of the range. Its frame stays
    the span of what was recorded, so its suggestions do not move."""
    opt = _two_rounds("legacy_fixed")
    opt.add_ingredient("Salt", 4.0, 4.0)
    state = opt.export_json()
    for var in state["variables"]:
        var.pop("search_range", None)
    old = FoodOptimizer("legacy_fixed_copy", robust=False)
    old.import_json(json.loads(json.dumps(state)))
    assert "search_range" not in old._var_by_name("Salt")
    assert old._search_bounds()[1] == (1.0, 6.0)


def _web_state(name):
    """A web-app project: Salt deactivated (active False) and pinned at 4 g
    inside its own 0-10 g, after it varied in two rounds."""
    opt = _two_rounds(name)
    state = json.loads(json.dumps(opt.export_json()))
    state["CLASS_VERSION"] = 9
    for var in state["variables"]:
        var["active"] = True
        if var["name"] == "Salt":
            var["active"] = False
            var["_frozen_at"] = 4.0
    return state


def test_a_migrated_web_project_is_learned_as_the_web_app_learns_it():
    opt = FoodOptimizer("web_copy", robust=False)
    opt.import_json(_web_state("web_src"))
    salt = opt._var_by_name("Salt")
    assert salt["bounds"] == (4.0, 4.0)
    assert salt["search_range"] == [0.0, 10.0]
    assert "active" not in salt and "_frozen_at" not in salt
    # The web app's _get_fixed_features: (4 - 0) / (10 - 0).
    assert opt._get_fixed_features() == {1: pytest.approx(0.4)}
    seen = _spy(opt)
    opt.ask(n_suggestions=1)
    # The web app's _fit_gp normalizes with the variable's own bounds.
    assert torch.allclose(seen[0][:, 1],
                          torch.tensor(SALT, dtype=torch.double) / 10.0,
                          rtol=0, atol=1e-12)


def test_a_migrated_web_row_held_without_a_number_still_remembers():
    state = _web_state("web_bare")
    for var in state["variables"]:
        var.pop("_frozen_at", None)
    opt = FoodOptimizer("web_bare_copy", robust=False)
    opt.import_json(state)
    salt = opt._var_by_name("Salt")
    assert salt["bounds"] == (0.0, 0.0)
    assert salt["search_range"] == [0.0, 10.0]
    assert opt._search_bounds()[1] == (0.0, 10.0)


def test_the_memory_survives_save_and_load():
    opt = _two_rounds("roundtrip")
    _grid_fix(opt, "Salt", 4.0)
    before = opt._search_bounds()
    reloaded = FoodOptimizer("roundtrip", robust=False)
    assert reloaded.load_error is None
    assert reloaded._var_by_name("Salt")["search_range"] == [0.0, 10.0]
    assert reloaded._search_bounds() == before
    # And through a copy written to disk as JSON text.
    clone = FoodOptimizer("roundtrip_clone", robust=False)
    clone.import_json(json.loads(json.dumps(opt.export_json())))
    assert clone._search_bounds() == before
    for got, want in zip(clone.ask(n_suggestions=2), opt.ask(n_suggestions=2)):
        assert got["Salt"] == want["Salt"] == pytest.approx(4.0)
        assert got["Water"] == pytest.approx(want["Water"], rel=1e-6)


def test_a_damaged_or_stale_memory_is_dropped_on_load():
    """A row given a range again by a version that kept no memory still
    carries the key; a hand-edited one may be anything. Neither is read."""
    opt = _two_rounds("stale")
    opt.add_ingredient("Salt", 4.0, 4.0)
    state = json.loads(json.dumps(opt.export_json()))
    for var in state["variables"]:
        if var["name"] == "Water":
            var["search_range"] = [10.0, 20.0]      # varying row: stale
        if var["name"] == "Salt":
            var["search_range"] = ["x", None]        # damaged
    clone = FoodOptimizer("stale_clone", robust=False)
    clone.import_json(state)
    assert "search_range" not in clone._var_by_name("Water")
    assert "search_range" not in clone._var_by_name("Salt")
    assert clone._search_bounds() == [(0.0, 100.0), (1.0, 6.0)]
