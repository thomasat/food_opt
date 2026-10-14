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
    assert bo_views.role(opt, {"recipe": recipe, "note": "Pilot starting set 1.1"}) == \
        "Pilot starting set 1.1"
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


def test_called_it_leads_with_the_overall_score(plain):
    opt = _Obj()
    lf = _lf(opt, _round6())
    line = U.called_line(opt, lf)
    # Persona fixes 2: led by where the scores landed (a miss above its
    # range is a miss), then the order of the pairs that count.
    # Review critique: measured ties are counted aloud.
    # Round 2: no brackets inside the line.
    assert line == ("**Round 6: scored higher than forecast.** 2 of 3 above their range; "
                    "order right: 2 pairs, 1 tie.")
    # "about 4": the mean of measured − forecast over those above their range.
    above, below = U.score_landing(learned.round_checks(_round6(), insights.SCORE, 6))
    assert above == pytest.approx(((98.33 - 94.3) + (98.33 - 94.7)) / 2) and below is None
    assert "calls right" not in line and "highest" not in line
    # The cumulative record, a second short line.
    assert U.called_so_far(opt, lf) == "So far: Overall score 11 of 11, Consistency 11 of 11."
    assert _words(U.called_so_far(opt, lf)) <= 12   # the coordinator's line


def test_called_it_measurement_lines_never_say_highest_for_a_target(plain):
    opt = _Obj("target")
    detail = U.called_detail(opt, _lf(opt, _round6()))
    assert "Consistency: order right (2 of 2 pairs)." in detail
    assert "calls right" not in detail
    cons = detail.split("Consistency:", 1)[1].split("\n\n", 1)[0]
    assert "highest" not in cons.lower() and "lowest" not in cons.lower()
    # A higher-is-better measurement may still name its top pick.
    opt = _Obj("max")
    detail = U.called_detail(opt, _lf(opt, _round6()))
    # Review critique: the pairs and the top pick are the "?"'s expert part,
    # kept when it fits the 120 words; the words themselves:
    assert wording.called_top(18, [18], True) == "Forecast highest: F18 ✓"
    assert "Consistency: order right (2 of 2 pairs)." in detail


def test_called_it_off_orders_say_pairs_in_the_right_order(plain):
    opt = _Obj()
    lf = _lf(opt, _round6(cons_meas=(13, 12, 11)))      # Consistency reversed
    line = U.called_line(opt, lf)
    assert line.startswith("**Round 6: scored higher than forecast.**")
    # Review critique: "; 1 tie" said too: at most 14 words besides numbers.
    assert _words(re.sub(r"[\d–]+", "", line)) <= 16
    assert "Consistency: 1 of 3 pairs in the right order." in U.called_detail(opt, lf)
    # The Overall order off too, and misses both ways, said plainly.
    lf = _lf(opt, _round6(score_meas=(90, 91, 97), cons_meas=(13, 12, 11)))
    assert U.called_line(opt, lf) == ("**Round 6: scored outside forecast, both ways.** All 3 "
                                      "outside their range; order right in 1 of 3 pairs.")
    assert wording.called_record("Firm", 8, 8) == "**Firm: 8 of 8 pairs in the right order.**"


def test_called_it_landing_words():
    assert wording.called_landing(3.8) == "scored about 4 above forecast"
    assert wording.called_landing(None, -2.2) == "scored about 2 below forecast"
    # Sceptic review (2026-10-14): the count and the reference, not "as forecast".
    assert wording.called_landing() == "all inside their 95 % range"
    assert wording.called_landing(n=3) == "all 3 inside their 95 % range"
    assert wording.called_order([[16, 17], [18]]) == "F16, F17 above F18"
    assert wording.called_order([[1], [2], [3], [4]]) == "F1 top, F4 bottom"


def test_slopegraph_heads_say_order(plain, monkeypatch):
    opt = _Obj()
    lf = _lf(opt, _round6())
    chart = U.called_chart(opt, lf).to_dict()
    texts = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            texts.add(node)
    walk(chart)
    # Review critique (fix 1): no slopegraph; one shared axis, its legend
    # naming the range, the forecast and the measured value in or out.
    assert {wording.CHECK_KEY_RANGE, wording.CHECK_KEY_FORECAST, wording.CHECK_KEY_IN,
            wording.CHECK_KEY_OUT} <= texts
    assert "Forecast order" not in texts


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


def test_progress_answer_text_from_facts(monkeypatch):
    monkeypatch.setattr(U, "rise_and_need", lambda a: ("3.3", "3.0"))
    facts = NS(best=NS(score=98.33), improvement=NS(rise=3.33), progress="rising")
    assert U.progress_answer_text(None, facts) == (
        "Best 95.0 → 98.3 (+3.3; needs 3.0 if tasting noise ≈ 1 point)")
    facts.progress = "flat"
    assert U.progress_answer_text(None, facts) is None


def test_the_score_is_explained_from_the_projects_shares(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    opt = NS(objectives=[{"name": n, "weight": 25.0, "goal": "target", "min_val": 0,
                          "max_val": 15} for n in "ABCD"], utility_ceiling=lambda: 100.0)
    assert U.score_built_text(opt) == ("Each measurement is worth 25 of 100 points; it loses "
                                       "points the further it is from its target (25 × "
                                       "distance ÷ 15).")
    opt.objectives[0].update(weight=40.0, goal="max")
    mixed = U.score_built_text(opt)
    assert mixed.startswith("Points out of 100: A 40, B 25") and "target" in mixed


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


def test_apart_text_says_beyond_anything_tested(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    # Below every Water tested (30-60): beyond.
    line = U.apart_text(opt, {"Water": 10.0, "Wheat gluten": 15.0}, "explores")
    assert line.startswith("Explores: less Water") and line.endswith("than anything tested")
    line = U.apart_text(opt, {"Water": 50.0, "Wheat gluten": 20.0}, "balanced")
    assert line.endswith("than Best so far")


def _rows(*means, recipes=None):
    recipes = recipes or [None] * len(means)
    return [U.PreviewRow(k, "explores", "98", "", "", m, r)
            for k, (m, r) in enumerate(zip(means, recipes), 1)]


def test_learning_round_line(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    best = float(opt.Y_history[opt.best_index()])
    assert U.learning_round(opt, _rows(best - 0.3, best, best - 2))
    assert not U.learning_round(opt, _rows(best - 0.3, best + 1.0))
    assert not U.learning_round(opt, _rows(best, None))
    assert _words(wording.LEARNING_ROUND) <= 10


def test_widen_line_when_a_suggestion_moves_away_from_the_limit(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    there = insights.start_recipe(opt, opt._best_recipe_index())
    up = dict(there, Water=there["Water"] + 5)
    down = dict(there, Water=there["Water"] - 5)
    lowest = NS(kind="close", binding=NS(kind="lowest", name="Water"))
    assert U.widen_beyond(opt, lowest, _rows(90, 90, recipes=[down, up]))
    assert not U.widen_beyond(opt, lowest, _rows(90, recipes=[down]))
    highest = NS(kind="close", binding=NS(kind="highest", name="Water"))
    assert U.widen_beyond(opt, highest, _rows(90, recipes=[down]))
    assert not U.widen_beyond(opt, NS(kind="see"), _rows(90, recipes=[up]))
    assert wording.WIDEN_BEYOND == "Suggestions stay inside your limits; widening goes beyond them."


def test_stop_headline(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    facts = NS(next=NS(kind="close", binding=NS(kind="lowest", name="Coconut cream")))
    assert U.stop_headline(None, facts) == \
        "**Little left to gain inside your limits.** Widen a limit or stop."
    facts.next.binding = None
    assert U.stop_headline(None, facts).endswith("Add an ingredient or stop.")
    assert _words(U.stop_headline(None, facts)) <= 14


# ------------------------------------------------------------------ #
#  7-8. Tooltips that agree, plain first
# ------------------------------------------------------------------ #

def test_next_tooltip_scopes_the_misses(monkeypatch):
    """The Model item says Predicts; the Next "?" must not say "recent
    results fell outside": it names the measurements that missed and the
    Overall score's own record."""
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U, "status_full_texts", lambda o, f: {
        "next": "Next: " + wording.next_plain_misses()})
    monkeypatch.setattr(U, "tooltip_for", lambda o, w, f: "RULE")
    monkeypatch.setattr(U, "evidence_line", lambda o, f, w: "EVIDENCE")
    monkeypatch.setattr(U, "open_rows", lambda o: [])
    items = [NS(name="Thickness uniformity", misses=4), NS(name="Creaminess", misses=3),
             NS(name="Tangy", misses=1)]
    facts = NS(next=NS(kind="misses", diagnosis=NS(items=items)),
               headline=NS(score=NS(inside=10, total=12)), best=NS(formulation=16))
    opt = NS(pending_batch=None, pending_batch_no=None)
    tip = U.status_help(opt, facts, "next")
    assert "Recent results fell outside" not in tip
    assert tip.startswith("Single measurements (Thickness uniformity, Creaminess) landed "
                          "outside their predicted range more often than expected")
    assert "The Overall score landed in range 10 of 12 times." in tip
    assert "Repeat of Formulation 16" in tip


def test_model_tooltip_opens_plain_and_ends_for_experts(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U, "learned_label", lambda o, n: "Overall score"
                        if n == insights.SCORE else n)
    o = lambda v, r, p: NS(verdict=v, right=r, pairs=p)
    lf = U.LearnedFacts({insights.SCORE: o(learned.PREDICTABLE, 10, 11),
                         "Consistency": o(learned.PREDICTABLE, 8, 8),
                         "Creaminess": o(learned.NOT_YET, 3, 8)}, {}, (6, []), None,
                        (1, 2, 0), "Overall score: 10 of 12 inside. Skill over 12 checks: "
                        "+0.69 against the average of earlier scores, −0.11 against Best so far.")
    lf.plain_line = wording.skill_plain(10, 12, 0.69, -0.11)
    # The old card: no warning, its sentence said "still very uncertain".
    card = U.Card("Reliable", [("10 / 12 in range", False)], None,
                  "those for Consistency are still very uncertain")
    facts = NS(headline=NS(inside=27, total=36))
    got = U.learned_card(None, lf, card, facts)
    assert got.word == "Predicts:"     # review critique: the strip's colon
    tip = got.help
    # Persona fixes 2: the Overall score named when Predictable.
    # Rebuilt framing: the forecast had none of that Round's results, which
    # is true live and in a replay; "before tasting" read as a timing claim.
    # Specific wording (2026-10-12), re-pinned: the pairs behind Predicts.
    # Sceptic review (2026-10-14), re-pinned: called of all pairs (these
    # hand-made orderings have no ties or too-close pairs: all = called).
    assert tip.startswith("Forecasting each Round from earlier Rounds, the model put the "
                          "Overall score (11 of 11 pairs called, 10 right) and Consistency "
                          "(8/8 called) in the right order")
    assert "before tasting" not in tip
    assert "still very uncertain" not in tip
    assert "about as accurate as assuming each new formulation scores like your best" in tip
    # Round 2: the window is stated.
    assert re.search(r"Single measurements(, last \d+ Rounds?)?: 27 of 36 in their 95\u00a0% "
                     r"range\.", tip)
    last = tip.split("\n\n")[-1]
    assert last.startswith("For experts:") and "qLogNEI" in last
    assert "posterior predictive" in last
    # The plain part never says "Skill".
    assert "Skill" not in tip.split("For experts:")[0]


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
              wording.called_score_head(6, "F1 above F2", wording.called_landing(4.0)),
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
