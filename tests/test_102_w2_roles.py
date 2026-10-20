"""W2 of the 0.8.0 e2e fixes (spec 2026-10-01 rev. 2; review I3, M3):
one role rule for the Compared with cell and the Why fold's Role column,
model-free, read at the Formulation total, and the Compared with changes
ranked over amounts and settings together. Pinned on the campaign's own
projects (tests/e2e_fixtures.py) and on small projects built here."""
from collections import Counter

import pytest

import wording
from food_bo import CLOSE_TO_THE_BEST, FAR_FROM_TESTED, FoodOptimizer
from tests import e2e_fixtures as E

REFINES, BALANCED, EXPLORES = (wording.SUGGESTION_CLOSE, wording.SUGGESTION_BALANCED,
                               wording.SUGGESTION_DIFFERENT)


def _roles(name):
    """{Formulation number: role} for every generated row of every warm
    Round, each read on the project as it stood before its Round (Best so
    far and the formulations recorded then), as the Round's own table read
    it while it was open."""
    full = E.load(name)
    out = {}
    for r in sorted({int(b) for b in full.batch_history if b is not None})[1:]:
        before = E.load(name, last_round=r - 1)
        for i, b in enumerate(full.batch_history):
            if b == r and not full.notes_history[i]:
                kind = before.suggestion_role(full.recipe_history[i])
                if kind != wording.SUGGESTION_SPREAD:
                    out[int(full.formulation_ids[i])] = kind
    return out


def test_the_far_cut_is_twice_the_near_one():
    """Review I3: one cut used twice left "balanced" empty."""
    assert FAR_FROM_TESTED == pytest.approx(2 * CLOSE_TO_THE_BEST) == pytest.approx(0.30)


def test_the_03_clean_run_refines_and_balances_and_never_explores():
    """Review I3's table: clean, 10 refines and 14 balanced (the Repeats
    are the user's own and say so); no explores, the search never jumped."""
    roles = _roles(E.BAR_CLEAN)
    assert Counter(roles.values()) == {REFINES: 6, BALANCED: 14}
    assert EXPLORES not in roles.values()


def test_the_03_drift_run_never_explores():
    assert EXPLORES not in _roles(E.BAR_DRIFT).values()


def test_the_03_noisy_run_explores_where_the_search_jumped():
    """Review I3: "explores" only where the search jumped to a new region,
    noisy Rounds 3-7, 8 rows."""
    roles = _roles(E.BAR_NOISY)
    assert sorted(n for n, k in roles.items() if k == EXPLORES) == [
        10, 11, 13, 17, 22, 25, 26, 27]


def test_the_02_big_moves_never_refine():
    """02 I1: protein −9.3 g (F12, Round 3 at 500 g) read "refines the best"
    in Why and "Trying something different" in the grid. Now one word, and
    never "refines"; F22, the 02 jump the review names, explores."""
    roles = _roles(E.PATTY)
    assert roles[12] == EXPLORES and roles[22] == EXPLORES
    assert _roles(E.PATTY_ROUND3)[12] == EXPLORES


def test_a_500g_round_is_read_at_the_total_on_both_sides():
    """The cell's changes, without a table total, are read at the
    Formulation total too: a row stored at 500 g against a best made at
    100 g listed five-fold changes."""
    before = E.load(E.PATTY_ROUND3, last_round=2)
    r3 = E.load(E.PATTY_ROUND3)
    cell = before.compared_with_text(r3.recipe_history[11], max_changes=2)
    assert cell == "Explores · Textured pea protein −9.30 g, Methylcellulose −0.50 g"
    # And beside the table printed at 500 g, the numbers it shows.
    assert before.compared_with_text(r3.recipe_history[11], scale_to=500.0,
                                     max_changes=2) == (
        "Explores · Textured pea protein −46.50 g, Methylcellulose −2.50 g")


# ------------------------------------------------------------------ #
#  The rule on a small project
# ------------------------------------------------------------------ #

def _opt(tmp_path, monkeypatch, name="roles"):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name)
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Wheat gluten", 0, 100)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 40.0, "Wheat gluten": 20.0}, {"Firmness": 9.0},
             formulation_no=1, batch_no=1)
    for i in range(2, 6):
        opt.tell({"Water": 40.0 + i, "Wheat gluten": 20.0}, {"Firmness": 3.0},
                 formulation_no=i, batch_no=1)
    return opt


def test_a_repeat_of_best_so_far_counts_as_it(tmp_path, monkeypatch):
    """Review I3: within 0.15 of Best so far OR of a Repeat of it. F6 is a
    Repeat of F1 (0.4 g off: _same's 0.5 % of the span), scored lower."""
    opt = _opt(tmp_path, monkeypatch)
    opt.tell({"Water": 40.4, "Wheat gluten": 20.0}, {"Firmness": 8.0},
             formulation_no=6, batch_no=2)
    assert opt.best_formulation_no() == 1
    # 15.3 g from F1, 14.9 g from its Repeat.
    assert opt.suggestion_role({"Water": 55.3, "Wheat gluten": 20.0}) == REFINES
    # A formulation near it that is not a Repeat of Best so far does not count.
    opt.tell({"Water": 42.0, "Wheat gluten": 22.0}, {"Firmness": 8.0},
             formulation_no=7, batch_no=2)
    assert opt.suggestion_role({"Water": 40.0, "Wheat gluten": 37.5}) == BALANCED


def test_explores_counts_every_formulation_made_before_the_round(tmp_path, monkeypatch):
    """Review I3: "farther than 0.30 from every formulation recorded before
    the Round, scored, waiting, Not scored or own" — and not the Round's own
    rows."""
    opt = _opt(tmp_path, monkeypatch)
    far = {"Water": 90.0, "Wheat gluten": 20.0}
    assert opt.suggestion_role(far) == EXPLORES
    # A Not scored formulation near it, from an earlier Round.
    opt.record_skipped(6, 1, {"Water": 85.0, "Wheat gluten": 20.0})
    assert opt.suggestion_role(far) == BALANCED


def test_a_row_of_the_open_round_is_not_earlier_than_its_round(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    far = {"Water": 90.0, "Wheat gluten": 20.0}
    opt.set_pending_batch([far, {"Water": 88.0, "Wheat gluten": 20.0}])
    no = opt.pending_batch_no
    # The other row of the Round is recorded first; it is not earlier.
    opt.tell({"Water": 88.0, "Wheat gluten": 20.0}, {"Firmness": 3.0},
             formulation_no=opt.pending_batch[1]["formulation"], batch_no=no)
    assert opt.suggestion_role(far) == EXPLORES
    # Read for the next Round, it is.
    assert opt.suggestion_role(far, before=no + 1) == BALANCED


def test_the_cold_start_and_own_rows_keep_their_kinds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("cold")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 40.0}, {"Firmness": 9.0}, formulation_no=1, batch_no=1)
    assert opt.suggestion_role({"Water": 90.0}) == wording.SUGGESTION_SPREAD
    opt = _opt(tmp_path, monkeypatch, "own")
    assert opt.compared_with_text({"Water": 90.0, "Wheat gluten": 20.0}, own=True) == (
        "Your own formulation · Water +50.00 g")


def test_the_role_does_not_ask_the_model(tmp_path, monkeypatch):
    """Model-free (review I3; reverts 11152ff's role part): nothing fitted,
    no pool, no checks."""
    import food_bo
    opt = _opt(tmp_path, monkeypatch)
    monkeypatch.setattr(food_bo, "fit_gpytorch_mll", lambda *a, **k: pytest.fail("fitted"))
    assert opt.suggestion_role({"Water": 90.0, "Wheat gluten": 20.0}) == EXPLORES


# ------------------------------------------------------------------ #
#  B2: Compared with ranks amounts and settings together (review M3)
# ------------------------------------------------------------------ #

def test_the_03_temperature_change_is_named_first():
    """03 I6: clean R4, F14 read "Rolled oats −0.67 g, Peanut butter
    +0.41 g"; the change that mattered was Baking temperature +10.7 °C (27 %
    of its span). Ranked by share of span, and a setting with a setting's
    decimals (03 M3: never thirteen)."""
    before = E.load(E.BAR_CLEAN, last_round=3)
    full = E.load(E.BAR_CLEAN)
    assert before.compared_with_text(full.recipe_history[13], max_changes=2) == (
        "Balanced · Baking temperature +10.66 °C, Rolled oats −0.67 g")


def test_fill_to_total_and_residue_are_left_out(tmp_path, monkeypatch):
    """Review M3: the Fill-to-total row is the other rows' difference (02 M7
    "Water +0.01 g"), and a change under one display step reads 0.00 (03 M6
    "Honey 0.00 g")."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("fill")
    opt.add_ingredient("Pea", 10, 30)
    opt.add_ingredient("Oil", 5, 20)
    opt.add_ingredient("Water", 0, 100)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell(opt.fill_formulas({"Pea": 20.0, "Oil": 10.0}), {"Firmness": 9.0},
             formulation_no=1, batch_no=1)
    moved = opt.fill_formulas({"Pea": 20.004, "Oil": 12.0})
    assert opt.vs_best_text(moved) == "Oil +2.00 g"
    assert [n for n, _ in opt._variable_deltas(moved, opt.recipe_history[0])] == ["Oil"]


def test_the_round_table_writes_settings_to_two_decimals(tmp_path, monkeypatch):
    """03 M3, 02 M2: 141.4638215303421 °C and 2.3705324605107307 min on the
    table the bench reads."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("dial")
    opt.add_ingredient("Pea", 0, 100)
    opt.add_process_parameter("Baking temperature", 140, 180, unit="°C")
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_pending_batch([{"Pea": 10.0, "Baking temperature": 141.4638215303421}])
    frame = opt.batch_frame(opt.pending_batch)
    assert frame["Baking temperature (°C)"].iloc[0] == 141.46
