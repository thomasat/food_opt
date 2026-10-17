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

The yogurt pilot samples ship in data/samples/yogurt_pilot: those tests always
run, and assert no pilot numbers or names. A synthetic project (Line A's
make-ups, checks written by hand) covers the rule in CI."""
import importlib.util
import json
import os
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import sample_projects as sp
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.review_screens import _install, _leaves, _open, _text, _words

BENCH = re.compile(r"panel|supplier|equipment", re.I)
CONTRADICTION = ("too narrow", "Wide 95 %", "fall inside easily")

R5, R6, ALL10 = "Yogurt pilot - Round 5", sp.ROUND6, sp.ALL10


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


def test_one_way_says_which_way():
    """Ruling 1: one way names the side when every measurement agrees."""
    assert wording.card_diag("one_way", "above") == "Results mostly above predictions"
    assert wording.card_diag("one_way", "below") == "Results mostly below predictions"
    assert wording.card_diag("one_way", "mixed") == "Each measurement misses one way"
    for side in ("above", "below", "mixed"):
        assert _words(wording.card_diag("one_way", side)) <= 6


# ------------------------------------------------------------------ #
#  The yogurt pilot (shipped samples)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory, pilot_dir):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-diag-pilot")
    for n in (R5, R6, ALL10):
        shutil.copy(pilot_dir / f"{n}.pkl", work)
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


