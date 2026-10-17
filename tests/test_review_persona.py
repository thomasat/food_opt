"""Persona fixes (prototype, 2026-10-02; two simulated demo viewers, a
senior food scientist and an R&D director):

- a formulation of the user's own with a note says the note as its source
  ("Pilot batch 6.1 · EGBO"), never "Your own formulation" (generic: any
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
from types import SimpleNamespace as NS

import pandas as pd
import pytest

import bo_views
import insights
import learned
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from insights import Check
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


def _check(no, r, name, pred, meas, low=None, high=None, full=True):
    low = pred - 1 if low is None else low
    high = pred + 1 if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=pred, sd=0.5)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


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
    cell = opt.compared_with_text(recipe, own=True, note="Pilot batch 6.1 · EGBO")
    assert cell.startswith("Pilot batch 6.1 · EGBO · ")
    assert "Your own" not in cell
    # Generic: any note, not a pilot's.
    assert opt.compared_with_text(recipe, own=True, note="Bench control B").startswith(
        "Bench control B · ")
    # No note, or the note the app types itself: as before.
    for note in (None, "", wording.OWN_FORMULATION_NOTE):
        assert opt.compared_with_text(recipe, own=True, note=note).startswith(
            "Your own formulation · ")
    assert bo_views.role(opt, {"recipe": recipe, "note": "Random starting set 1.1"}) == \
        "Random starting set 1.1"
    assert bo_views.role(opt, {"recipe": recipe, "note": wording.OWN_FORMULATION_NOTE}) == \
        wording.OWN_FORMULATION_KIND


def test_the_round_table_shows_the_note_as_source(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    opt.add_to_pending_batch({"Water": 55.0, "Wheat gluten": 12.0}, "Pilot batch 3.1 · EGBO")
    opt.add_to_pending_batch({"Water": 42.0, "Wheat gluten": 22.0}, wording.OWN_FORMULATION_NOTE)
    frame = opt.batch_frame(opt.pending_batch)
    cells = list(frame[opt.compared_with_column()])
    assert cells[0].startswith("Pilot batch 3.1 · EGBO")
    assert cells[1].startswith("Your own formulation")


# ------------------------------------------------------------------ #
#  2. Called it?, re-aimed
# ------------------------------------------------------------------ #

class _Obj:
    """A project the Called it? helpers read: one target measurement."""

    def __init__(self, goal="target"):
        self.objectives = [{"name": "Consistency", "goal": goal, "target": 12}]

    def measurements_by_importance(self):
        return [{"name": "Consistency"}]


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0 if n == insights.SCORE
                                                                else 15.0))
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)


def _round6(score_meas=(98.33, 98.33, 91.67), cons_meas=(12, 12, 13)):
    """Rounds 3-5 earn Consistency Predictable (9 of 9; round 2: three
    Rounds, so it stays Predictable once Holm adjusts for the 2 results);
    Round 6 as the demo: Overall forecast F17 > F16 > F18, measured F16 =
    F17 > F18."""
    checks = []
    for r, base in ((3, 7), (4, 10), (5, 13)):
        checks += [_check(base + k, r, "Consistency", 13 - k, 13 - k) for k in range(3)]
        checks += [_check(base + k, r, insights.SCORE, 95 - k, 95 - k, low=92 - k, high=98 - k)
                   for k in range(3)]
    for no, sp, sm, cp, cm in zip((16, 17, 18), (94.3, 94.7, 93.7), score_meas,
                                  (13.5, 13.4, 13.6), cons_meas):
        checks.append(_check(no, 6, insights.SCORE, sp, sm, low=sp - 2.5, high=sp + 2.5))
        checks.append(_check(no, 6, "Consistency", cp, cm, low=cp - 2.5, high=cp + 2.5))
    return checks


def _lf(opt, checks):
    # Persona fixes 2: no noise here (every pair counts); the noise rule
    # has its own tests (test_review_persona2).
    orderings = learned.orderings(opt, checks, {})
    r, calls = learned.called_it(opt, checks, 6, {})
    score_call = learned.round_call(checks, insights.SCORE, 6)
    named = [(insights.SCORE, score_call)] + calls
    items = [(n, call, learned.round_checks(checks, n, 6)) for n, call in named]
    mine = learned.round_checks(checks, insights.SCORE, 6)
    landing = (sum(c.inside for c in mine), sum(c.measured > c.high for c in mine),
               sum(c.measured < c.low for c in mine))
    return U.LearnedFacts(orderings, {}, (r, calls), score_call, landing, "", items)


def test_called_it_landing_words():
    assert wording.called_order([[16, 17], [18]]) == "F16, F17 above F18"
    assert wording.called_order([[1], [2], [3], [4]]) == "F1 top, F4 bottom"


# ------------------------------------------------------------------ #
#  3-4. Progress, the score built
# ------------------------------------------------------------------ #

def test_progress_strip_and_answer_words():
    assert wording.strip_progress_tail("98.3", "3.3") == "· best 98.3 (+3.3)"
    # Review critique (owner's decision): the assumed noise said beside it.
    assert wording.progress_rise_answer("95.0", "98.3", "3.3", "3.0") == \
        "Best 95.0 → 98.3 (+3.3; needs 3.0 if tasting noise ≈ 1 point)"
    assert wording.answer_line("Rising", wording.progress_rise_answer("95.0", "98.3", "3.3",
                                                                       "3.0")) == \
        "**Rising.** Best 95.0 → 98.3 (+3.3; needs 3.0 if tasting noise ≈ 1 point)."
    # Round 2 (P1 #6): what 3.0 is, in one line.
    need = wording.progress_need_plain("3.0", q=3)
    assert need.startswith("3.0: the best of 3 new results must beat noise by this much.")
    assert "about 1 point" in need and "no Repeats" in need


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

def test_results_round_view_has_predicted_beside_measured(plain):
    opt = _Obj()
    frame = U.round_predicted_frame(opt, _round6(), 6)
    # Review critique (fix 11): where each landed, and every measurement.
    assert list(frame.columns) == ["Result", "Formulation", "Measured", "Predicted (95 % range)",
                                   "In / Above / Below range"]
    first = frame.iloc[0].to_dict()
    assert first == {"Result": "Overall score", "Formulation": "F16", "Measured": "98.3",  # round 2: F16
                     "Predicted (95 % range)": "94.3 (91.8–96.8)",
                     "In / Above / Below range": "Above"}
    assert set(frame["Result"]) == {"Overall score", "Consistency"}
    # Review critique (fix 12): a top below the point never prints.
    assert frame[frame.Result == "Consistency"].iloc[2]["Predicted (95 % range)"] == \
        "13.6 (11.1–15.0)"  # round 2: the top with the point's decimals
    assert U.round_predicted_frame(opt, _round6(), 9) is None


def test_results_round_view_on_a_real_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        memo = {}
        with insights.memo_in_use(memo):
            checks = insights.check_rounds(opt)
    last = insights._ordered_rounds(checks)[-1]
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    frame = U.round_predicted_frame(opt, checks, last)
    assert frame is not None and len(frame)
    assert (frame["Result"] == "Overall score").any()
    assert frame["Predicted (95 % range)"].str.contains(r"\(.+–.+\)").all()


# ------------------------------------------------------------------ #
#  6. What next?
# ------------------------------------------------------------------ #

def test_apart_lines():
    assert wording.apart_line("explores", [("less", "Tapioca starch"),
                                           ("less", "Nutritional yeast")], beyond=True) == \
        "Explores: less Tapioca starch and Nutritional yeast than anything tested"
    assert wording.apart_line("balanced", [("more", "Coconut cream"), ("less", "Salt")]) == \
        "Balanced: more Coconut cream, less Salt than Best so far"
    assert wording.apart_line("refines the best", []) == "Refines the best"


def _rows(*means, recipes=None):
    recipes = recipes or [None] * len(means)
    return [U.PreviewRow(k, "explores", "98", "", "", m, r)
            for k, (m, r) in enumerate(zip(means, recipes), 1)]


# ------------------------------------------------------------------ #
#  7-8. Tooltips that agree, plain first
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("vs_best,phrase", [(0.32, "more accurate than"),
                                            (-0.11, "about as accurate as"),
                                            (-0.5, "less accurate than")])
def test_skill_in_plain_words(vs_best, phrase):
    line = wording.skill_plain(10, 12, 0.69, vs_best)
    assert f"{phrase} assuming each new formulation scores like your best" in line


# ------------------------------------------------------------------ #
#  Wording limits and _BANNED
# ------------------------------------------------------------------ #

def _persona_strings():
    src = open(wording.__file__).read().split("# --- persona fixes ---", 1)[1]
    return re.findall(r'"((?:[^"\\\n]|\\.)*)"', src)


def test_persona_strings_pass_banned():
    texts = [t for t in _persona_strings() if t.strip()]
    texts += [wording.LEARNING_ROUND, wording.WIDEN_BEYOND, wording.STOP_HEAD,
              wording.STOP_WIDEN, wording.misses_scoped(["A"], 1, 2),
              wording.model_plain(["A"], ["B"], ["C"]), wording.roles_help("15", "30"),
              wording.results_predicted_help(6), wording.stop_help("1.0 points"),
              wording.called_score_head(6, "F1 above F2", "scored about 4 above forecast"),
              wording.called_so_far([("Overall", 1, 2)]), wording.EXPERT_TERMS,
              wording.progress_need_plain("3.0"), wording.single_record(1, 2)]
    hits = [(t, p.pattern) for t in texts for p in _BANNED if p.search(t)]
    assert hits == []


def test_persona_wording_limits():
    assert _words(wording.LEARNING_ROUND) <= 10
    assert _words(wording.RESULTS_PREDICTED_CAPTION) <= 8
    for label in (wording.CALLED_HEAD_FORECAST, wording.CALLED_HEAD_MEASURED,
                  wording.RESULTS_PREDICTED_COLUMN, wording.PREVIEW_APART_COLUMN):
        assert _words(label) <= 5
    assert _words("Rising " + wording.strip_progress_tail("98.3", "3.3")) <= 10


def test_forecast_panel_never_says_highest_for_a_target():
    line = wording.forecast_line("Consistency", ["F16 13.5 (11.1–≥ 15)", "F18 13.6 (10.0–≥ 15)"],
                                 None) + wording.forecast_order_tail([18, 16])
    assert line.endswith("Order, high to low: F18, F16.")
    assert "Highest" not in line
