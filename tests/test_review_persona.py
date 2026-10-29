"""Persona fixes (prototype, 2026-10-02; two simulated demo viewers, a
senior food scientist and an R&D director):

- a formulation of the user's own with a note says the note as its source
  ("Engine's pick 6.1"), never "Your own formulation" (generic: any
  note);
- Called it? leads with the Overall score whenever its order held; the
  measurement lines say "pairs in the right order", never "calls right",
  and never "highest" for a measurement with a target; the record is a
  second short line; the slopegraph heads say "Forecast order" and
  "Measured order";
- Results: the Round's forecast (Predicted, 95 % range) next to Measured;
- What next?: a line per suggestion that sets it apart, a learning-round
  line, the widen line, the stop headline;
- the strip, the Model "?" and the Next "?" agree with the verdicts.

Synthetic projects and hand-made checks only; no pilot data."""
import re

import pytest

import wording
from food_bo import FoodOptimizer
from tests.test_food_bo import _BANNED


# ------------------------------------------------------------------ #
#  1. The note shown as source
# ------------------------------------------------------------------ #

def _opt(tmp_path, monkeypatch, name="persona"):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Wheat gluten", 0, 50)
    opt.add_objective("Firmness", 1.0, goal="target", target=6, min_val=0, max_val=10)
    for k, (w, g, f) in enumerate([(40, 20, 5), (50, 10, 6), (30, 30, 4), (60, 5, 7),
                                   (45, 25, 5.5), (35, 15, 6.2)], 1):
        opt.tell({"Water": float(w), "Wheat gluten": float(g)}, {"Firmness": f},
                 formulation_no=k, batch_no=1 + (k > 3))
    return opt


def test_a_note_is_shown_as_the_source(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    recipe = {"Water": 90.0, "Wheat gluten": 20.0}
    cell = opt.compared_with_text(recipe, own=True, note="Engine's pick 6.1")
    assert cell.startswith("Engine's pick 6.1 · ")
    assert "Your own" not in cell
    # Generic: any note, not a pilot's.
    assert opt.compared_with_text(recipe, own=True, note="Bench control B").startswith(
        "Bench control B · ")
    # No note, or the note the app types itself: as before.
    for note in (None, "", wording.OWN_FORMULATION_NOTE):
        assert opt.compared_with_text(recipe, own=True, note=note).startswith(
            "Your own formulation · ")


def test_the_round_table_shows_the_note_as_source(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    opt.add_to_pending_batch({"Water": 55.0, "Wheat gluten": 12.0}, "Engine's pick 3.1")
    opt.add_to_pending_batch({"Water": 42.0, "Wheat gluten": 22.0}, wording.OWN_FORMULATION_NOTE)
    frame = opt.batch_frame(opt.pending_batch)
    cells = list(frame[opt.compared_with_column()])
    assert cells[0].startswith("Engine's pick 3.1")
    assert cells[1].startswith("Your own formulation")


# ------------------------------------------------------------------ #
#  2. Called it?, re-aimed
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  3-4. Progress, the score built
# ------------------------------------------------------------------ #


def test_the_score_sentence_matches_the_apps_formula(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("score")
    opt.add_ingredient("Water", 0, 100)
    for n in ("A", "B", "C", "D"):
        opt.add_objective(n, 25.0, goal="target", target=5, min_val=0, max_val=15)
    opt.set_shares({n: 25.0 for n in "ABCD"})
    # 25 × distance ÷ 15 lost per measurement: A off by 3, others on target.
    got = opt._compute_utility({"A": 8, "B": 5, "C": 5, "D": 5})
    assert got == pytest.approx(100 - 25 * 3 / 15)


# ------------------------------------------------------------------ #
#  5. Results: the Predicted column
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  6. What next?
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  7-8. Tooltips that agree, plain first
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Wording limits and _BANNED
# ------------------------------------------------------------------ #

def _persona_strings():
    src = open(wording.__file__, encoding="utf-8").read().split("# --- persona fixes ---", 1)[1]
    return re.findall(r'"((?:[^"\\\n]|\\.)*)"', src)


def test_persona_strings_pass_banned():
    texts = [t for t in _persona_strings() if t.strip()]
    hits = [(t, p.pattern) for t in texts for p in _BANNED if p.search(t)]
    assert hits == []


