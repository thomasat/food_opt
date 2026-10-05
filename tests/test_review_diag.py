"""Review diagnosis (owner, 2026-10-02): Off with no disagreeing Repeat
never says "Check panel".

On the yogurt pilot (Rounds 5 and 6, no Repeats) the Predictions card said
"Check panel before Round 6" for any Off. Only a disagreeing Repeat may
point at the panel, suppliers or equipment (spec 2026-10-01 first-time
design; e2e fixes): Round-ahead errors cannot show drift. Off with no such
Repeat now says where the misses fell (new ground / one way / not explained
yet, a description, never a cause), the Overall score when it is as
expected, and Next asks for a Repeat to check noise. The explanation under
More detail is one story: the count, then the misses by measurement, side,
mean error and how many fell on new ground (suggestion_role's
FAR_FROM_TESTED rule, judged against what was tested before each Round).

The yogurt files are confidential and gitignored: those tests skip without
them, and assert no pilot numbers or names. A synthetic project (Line A's
make-ups, checks written by hand) covers the rule in CI."""
import json
import os
import re
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_review_cards import _install, _leaves, _open, _text, _words, cards

BENCH = re.compile(r"panel|supplier|equipment", re.I)
CONTRADICTION = ("too narrow", "Wide 95 %", "fall inside easily")

def _pilot_dir():
    """The pilot's folder: FOODOPT_PILOT_DIR, this checkout's data/pilot/demo,
    or, from a .claude/worktrees checkout, the main repo's."""
    root = Path(__file__).resolve().parent.parent
    found = [Path(os.environ["FOODOPT_PILOT_DIR"])] if os.environ.get("FOODOPT_PILOT_DIR") else []
    found.append(root / "data" / "pilot" / "demo")
    if root.parent.name == "worktrees" and root.parent.parent.name == ".claude":
        found.append(root.parents[2] / "data" / "pilot" / "demo")
    return next((d for d in found if d.is_dir()), found[-1])


PILOT = _pilot_dir()
R5, R6, ALL10 = ("Yogurt pilot - Round 5", "Yogurt pilot - Round 6 to score",
                 "Yogurt pilot - all 10 Rounds")
needs_pilot = pytest.mark.skipif(
    not all((PILOT / f"{n}.pkl").exists() for n in (R5, R6, ALL10)),
    reason="confidential yogurt pilot not on this machine")


def _all_text(at):
    """Every rendered text on the Review tab, help texts included."""
    out = []
    for el, _, _ in _leaves(at.tabs[3]):
        out.append(_text(el))
        h = getattr(el.proto, "help", "")
        if h:
            out.append(h)
    return [t for t in out if t]


# ------------------------------------------------------------------ #
#  Synthetic: Line A's make-ups, checks written by hand
# ------------------------------------------------------------------ #

def _check(f, r, name, miss=False, wide=False):
    low, high = (-1e3, 1e3) if wide else (4.0, 6.0)
    measured = 9.0 if miss else 5.0
    return insights.Check(f, r, name, 5.0, low, high, measured, low <= measured <= high, True,
                          5.0, 0.5)


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-diag-syn"))
    try:
        state = json.loads((D.HERE / f"{D.LINE_A}.json").read_text())
        opt = FoodOptimizer(state["project_name"])
        opt.import_json(state)
        yield opt
    finally:
        os.chdir(here)


def _synthetic(opt, far_misses=True):
    """Rounds 3-5 of Line A checked: Firmness and Juiciness miss above on
    every formulation far from what was tested before its Round
    (`far_misses`), else on the near ones, one side each in turn; Cook
    loss's intervals wide (the old text added "values fall inside easily"
    beside "too narrow"); every Overall score inside."""
    ids = [int(x) for x in opt.formulation_ids]
    checks = []
    for i, r in enumerate(opt.batch_history):
        if r not in (3, 4, 5):
            continue
        f = ids[i]
        far = insights.far_from_tested(opt, f, r)
        hit = far if far_misses else not far
        for k, name in enumerate(("Firmness", "Juiciness")):
            c = _check(f, r, name, miss=hit)
            if hit and not far_misses and (f + k) % 2:
                c.measured, c.inside = 1.0, False     # below, alternating
            checks.append(c)
        checks.append(_check(f, r, "Cook loss", wide=True))
        checks.append(_check(f, r, insights.SCORE))
    return checks


def test_far_misses_without_a_repeat_read_as_new_ground(line_a):
    checks = _synthetic(line_a)
    h = insights.headline(line_a, checks)
    assert insights.verdict_state(line_a, h, checks) == "unreliable"
    assert "Cook loss" in (h.wide or [])
    d = insights.miss_diagnosis(line_a, checks, h)
    assert d.kind == "new_ground"
    assert d.far == d.misses == 10 and d.far_checked < d.checked
    assert d.score_ok


def test_scattered_near_misses_are_not_explained(line_a):
    checks = _synthetic(line_a, far_misses=False)
    d = insights.miss_diagnosis(line_a, checks)
    assert d.far == 0
    assert d.kind == "unexplained"


def _facts(opt, checks):
    facts = insights.status_facts(opt, checks)
    facts.checks = checks
    return facts


def test_synthetic_cards_never_say_panel(line_a):
    checks = _synthetic(line_a)
    facts = _facts(line_a, checks)
    # Review four, ruling 1: every Overall score held (12 of 12), so the
    # word is the score's; the measurements' misses are one plain line and
    # where they fell is in the "?" (was: Off, the diagnosis on the card).
    assert facts.prediction_row != "off_coverage" and facts.disagreement is None
    assert facts.measure_verdict == "unreliable" and facts.measures_miss
    p = ui_insights.predictions_card(line_a, facts)
    n = ui_insights.next_card(line_a, facts)
    texts = [p.word, n.word] + [t for t, _ in p.lines + n.lines]
    assert not any(BENCH.search(t) for t in texts), texts
    assert (wording.CARD_MEASURES_MISS, False) in p.lines
    assert not any(warn for _, warn in p.lines)
    assert (wording.card_noise_repeat(facts.best.formulation), False) in n.lines
    for t, _ in p.lines + n.lines:
        assert _words(t) <= 6, t


def test_synthetic_tooltip_holds_the_evidence(line_a):
    checks = _synthetic(line_a)
    facts = _facts(line_a, checks)
    d = facts.diagnosis
    tip = ui_insights.tooltip_for(line_a, "predictions", facts)
    assert f"({d.misses} of {d.checked} values;" in tip
    assert f"On new ground: {d.far} of {d.misses} misses, {d.far_checked} of {d.checked}" in tip
    assert "Firmness 5 of " in tip and "(above, mean error +4.0)" in tip
    assert "Diagnosis rule, first that holds" in tip and "Here: new ground." in tip
    assert "only a disagreeing Repeat can" in tip
    assert "Overall score, Rounds 3–5: 12 of 12 in range." in tip
    assert not BENCH.search(tip)


def test_synthetic_explanation_is_one_story(line_a):
    checks = _synthetic(line_a)
    h = insights.headline(line_a, checks)
    text = ui_insights.verdict_text(line_a, h, "unreliable", checks=checks)
    for bad in CONTRADICTION:
        assert bad not in text, text
    assert "Many more misses than expected." in text
    assert "Firmness 5 of 12 (above, mean error +4.0)" in text
    assert wording.DIAG_READING["new_ground"] in text
    step = insights.next_step(line_a, checks)
    said, _ = ui_insights.next_step_text(line_a, step, h.rounds)
    assert "too narrow" not in said and "mostly on new ground" in said
    assert not BENCH.search(said)


# ------------------------------------------------------------------ #
#  Demo projects: Line B keeps its check; the others unchanged
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-diag-demos"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (D.SNACK, D.LINE_A, D.LINE_B, D.LINE_C, D.PROCESS_ONLY):
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


def test_line_b_still_checks_the_panel(demos):
    p, _, n = cards(demos[D.LINE_B])
    # Review polish, one action, one card: the Next card carries the check.
    assert ":red[**Check panel before Round 6**]" not in p["lines"]
    assert n["word"] == ":red[Check panel first]"
    assert "Look at the panel, suppliers and equipment." in n["help"]
    assert not any(line in p["lines"] for line in wording.CARD_DIAG.values())


def test_one_way_says_which_way():
    """Ruling 1: one way names the side when every measurement agrees."""
    assert wording.card_diag("one_way", "above") == "Results mostly above predictions"
    assert wording.card_diag("one_way", "below") == "Results mostly below predictions"
    assert wording.card_diag("one_way", "mixed") == "Each measurement misses one way"
    for side in ("above", "below", "mixed"):
        assert _words(wording.card_diag("one_way", side)) <= 6


@pytest.mark.parametrize("name,line", [(D.LINE_C, "Repeat Formulation 18 to check noise"),
                                       (D.PROCESS_ONLY, "Repeat Trial 21 to check noise")])
def test_cv_widening_asks_for_a_repeat_not_the_panel(demos, name, line):
    """Ruling 2: the CV-widening rule names the panel only with Repeat
    evidence; these two have one Repeat pair (1 df), so they ask for one."""
    p, _, n = cards(demos[name])
    # One action, one card (demo build): when the Next card already asks
    # for this Repeat (process-only: "Round 7: Repeat Trial 21"), the
    # Predictions card does not say it again.
    # Review learned 2: the request is What next?'s alone (the Model
    # section repeats no old-card line).
    asked = [t for t in n["lines"] if line.split(" to check")[0] in t]
    assert len(asked) == 1 and line not in p["lines"], (p["lines"], n["lines"])
    texts = [p["word"], *p["lines"], p["help"]]
    assert not [t for t in texts if BENCH.search(t)]
    if name == D.PROCESS_ONLY:
        # Review four, ruling 1: Off on the Overall score (4 of 16 missed);
        # the Repeat request is the Next card's alone. Review learned: the
        # word is the Model item's, the miss line still red.
        assert p["word"] == "Predicts:" and p["lines"][0] == ":red[**4 / 16 missed**]"
        return
    assert "to check noise" in p["help"]     # round 2: the one form of the Repeat request
    assert ("the same values again" in p["help"]) == (name == D.PROCESS_ONLY)


def test_repeats_that_vary_a_lot_back_the_panel_advice(line_a):
    """Ruling 2: Repeats measuring the noise (3+ df) at half the spread of
    scores or more: "Repeats vary a lot", the advice kept in the tooltip."""
    import types
    big = types.SimpleNamespace(from_repeats=True, value=50.0, df=3, pairs=3)
    small = types.SimpleNamespace(from_repeats=True, value=0.1, df=3, pairs=3)
    assumed = types.SimpleNamespace(from_repeats=False, value=50.0, df=1, pairs=1)
    assert insights.repeats_vary(line_a, big).sd == 50.0
    assert insights.repeats_vary(line_a, small) is None
    assert insights.repeats_vary(line_a, assumed) is None
    assert _words(wording.CARD_REPEATS_VARY) <= 6
    assert "results vary a lot" in wording.repeats_vary_line("50.0", 3, "4.6")


@pytest.mark.parametrize("name", [D.SNACK, D.LINE_A, D.LINE_C, D.PROCESS_ONLY])
def test_no_diagnosis_where_nothing_is_off(demos, name):
    for c in cards(demos[name]):
        assert not any(line in c["lines"] for line in wording.CARD_DIAG.values())
        assert "Misses by measurement" not in (c["help"] or "")


# ------------------------------------------------------------------ #
#  The yogurt pilot (skipped without it)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-diag-pilot")
    for n in (R5, R6, ALL10):
        shutil.copy(PILOT / f"{n}.pkl", work)
    os.chdir(work)
    try:
        with bo_views.fresh():
            yield work
    finally:
        os.chdir(here)


def _open_review(name):
    return _open(name)


def _score_round_6():
    """Type Round 6's results (read from the all-10 file, never written in
    this test) on Results and save; then reopen as the owner would."""
    done = FoodOptimizer(ALL10)
    ids = [int(x) for x in done.formulation_ids]
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = R6
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    for row in FoodOptimizer(R6).pending_batch:
        no = int(row["formulation"])
        for m, v in done.results_history[ids.index(no)].items():
            at.number_input(key=f"f{no}_{m}").set_value(float(v))
    at.run()
    at.button(key="save_results").click()
    at.run()
    assert FoodOptimizer(R6).pending_batch in (None, [])
    return _open_review(R6)


def _check_without_repeat(at, name, word):
    """Review four, ruling 1: the word is the Overall score's (`word`); the
    measurements' misses one plain line; no panel advice; a neutral Repeat
    asked once; the diagnosis's evidence in the "?"."""
    texts = _all_text(at)
    assert not [t for t in texts if BENCH.search(t)], name
    p, _, n = cards(at)
    assert p["word"] == word
    # Review learned 2: the line contradicted the per-measurement verdicts;
    # gone from the section.
    assert wording.CARD_MEASURES_MISS not in p["lines"]
    assert not any(line.startswith(":red[") for line in p["lines"])
    assert all(line != wording.CARD_DIAG["unexplained"] for line in p["lines"])
    asks = [line for line in p["lines"] + n["lines"]
            if line.startswith("Repeat Formulation ") and line.endswith("to check noise")]
    assert len(asks) == 1, (p["lines"], n["lines"])
    stop = wording.answer_line(wording.CEILING_HEAD, wording.CEILING_TAIL)
    for line in p["lines"] + n["lines"]:
        # Review critique, follow-up: Round 6 scored sits within noise of the
        # top score; the stop headline (at most 14 words) leads What next?.
        # Round 2 (P1 #5): the stop's concrete reason line, at most 12.
        reason = line.startswith("On target:")
        assert _words(line) <= (14 if line == stop else 12 if reason else 6), line
    if name == R6:
        assert stop in n["lines"]
    opt = FoodOptimizer(name)
    checks = insights.check_rounds(opt)
    d = insights.miss_diagnosis(opt, checks)
    # Review critique (fix 13): the strip's "?" keeps to about 120 words,
    # plain first; the diagnosis's counts and rule are in Method and
    # definitions.
    assert d is not None and len(p["help"].split()) <= wording.TOOLTIP_WORDS
    for bad in CONTRADICTION:
        assert not any(bad in t for t in texts)


@needs_pilot
def test_pilot_round_5_never_says_panel(pilot):
    # 9 Overall-score checks: under SKILL_MIN, Not yet (ruling 1).
    # Review learned: the Model item's word (the ordering rule).
    # Persona fixes 2: pairs too close to call (forecasts under one noise SD
    # apart) no longer count, so fewer verdicts reach Predicts.
    # Review critique: the permutation test and the noise split: Partly.
    _check_without_repeat(_open_review(R5), R5, "Partly:")


@needs_pilot
def test_pilot_round_6_scored_never_says_panel(pilot):
    # The Overall score 10 of 12 in range, skill 0.69: Reliable (ruling 1).
    # Persona fixes 2: pairs too close to call (forecasts under one noise SD
    # apart) no longer count, so fewer verdicts reach Predicts.
    # Round 2 (P1 #2): Holm across the 5 results checked: the Overall
    # score's 5 of 5 is Partly, not Predicts.
    _check_without_repeat(_score_round_6(), R6, "Partly:")


@needs_pilot
def test_pilot_all_10_is_not_off(pilot):
    at = _open_review(ALL10)
    p = cards(at)[0]
    assert p["word"] == "Predicts:"     # review learned: the Model item (critique: colon)
    assert not [t for t in _all_text(at) if BENCH.search(t)]
