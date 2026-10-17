"""W1 of the 0.8.0 e2e fixes (spec 2026-10-01-e2e-fixes-design.md rev. 2;
research/2026-10-01-e2e/spec-review.md, rulings B1, B2, I1, I2, I4, I5,
I6, I7, I13, D4): Insights' logic and text.

- B1: a steady bias, per output, over every Round checked, with the Round
  as the unit from 4 Rounds; said once as a fact; never "check" advice. The
  "check" sentence comes only from a disagreeing Repeat, conditionally.
- B2: the noise threshold is unchanged; below 3 pairs it is said to be
  assumed, and Next step asks for a Repeat of Best so far.
- I2: Next step's order is repeat, unreliable, early, no_skill, the rest.
- I4: ingredient-effect claims need 10 Overall-score checks and skill 0.20.
- I13: the Repeat SD says its degrees of freedom.

The pins on the campaign's own projects (tests/e2e_fixtures.py) and the
demo projects come last; they fit models, so they are slow."""
import math
from types import SimpleNamespace

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import insights
import ui_insights
import wording
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _check(name, rnd, no, measured, predicted=50.0, sd=1.0, full=True):
    low, high = predicted - insights.Z95 * sd, predicted + insights.Z95 * sd
    return insights.Check(no, rnd, name, predicted, low, high, measured,
                          low <= measured <= high, full, predicted, sd)


def _errors(name, by_round, predicted=50.0, sd=5.0, no=100):
    """Checks of `name` with the given errors (measured − predicted), one
    list per Round, Rounds numbered from 2."""
    out = []
    for r, errors in enumerate(by_round, start=2):
        for k, e in enumerate(errors):
            out.append(_check(name, r, no + 10 * r + k, predicted + e, predicted, sd))
    return out


def _repeat_of(opt, src, dst):
    opt.recipe_history[dst] = dict(opt.recipe_history[src])
    opt.X_history[dst] = list(opt.X_history[src])


def _clean(*texts):
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t


# ------------------------------------------------------------------ #
#  B1: the error record, with the Round as the unit from 4 Rounds
# ------------------------------------------------------------------ #

def test_the_score_record_spans_every_round_checked():
    """Overrides I1: the Overall score's line, mean error and steady bias use
    every Round checked, not the verdict's last 3, and name their Rounds."""
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.1] * 10] * 5, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0] * 3, [-4.0] * 3, [0.0] * 3, [0.0] * 3,
                                         [0.0] * 3]))
    h = insights.headline(opt, checks)
    assert h.rounds == [4, 5, 6]
    assert h.score.rounds == [2, 3, 4, 5, 6]
    assert (h.score.total, h.score.mean_miss) == (15, pytest.approx(-1.6))


def test_from_four_rounds_the_round_is_the_unit_of_the_standard_error():
    """Ruling B1: a Round's formulations share the model's error, so its mean
    counts once: the SE is the spread of the Round means ÷ √Rounds."""
    four = _errors(insights.SCORE, [[-3.0, -3.2, -2.8], [-1.0, -1.1, -0.9],
                                    [-5.0, -5.1, -4.9], [-3.0, -3.1, -2.9]])
    rec = insights.error_record(four, insights.SCORE)
    means = [-3.0, -1.0, -5.0, -3.0]
    assert rec.se == pytest.approx(__import__("statistics").stdev(means) / 2)
    three = four[:9]
    rec3 = insights.error_record(three, insights.SCORE)
    gaps = [c.measured - c.predicted for c in three]
    assert rec3.se == pytest.approx(__import__("statistics").stdev(gaps) / 3)


def test_a_steady_bias_needs_the_threshold_and_two_round_unit_standard_errors():
    """Within-Round values agree closely, so the value SE is tiny; but the
    Round means swing, so with the Round as the unit it is not a bias."""
    swinging = _errors(insights.SCORE, [[-6.0] * 3, [2.0] * 3, [-7.0] * 3, [1.0] * 3])
    assert insights.bias_gap(insights.error_record(swinging, insights.SCORE), 1.0) is None
    steady = _errors(insights.SCORE, [[-3.0] * 3, [-3.5] * 3, [-2.5] * 3, [-3.2] * 3])
    assert insights.bias_gap(insights.error_record(steady, insights.SCORE),
                             1.0) == pytest.approx(-3.05)


def test_the_bias_rule_still_reads_a_record_with_no_round_se():
    rec = insights.ScoreRecord(11, 11, -1.1, 1.36)
    assert insights.bias_gap(rec, 1.0) == pytest.approx(-1.1)
    assert insights.bias_gap(insights.ScoreRecord(11, 11, -1.1, 3.0), 1.0) is None


def test_steady_bias_is_per_output_and_a_measurement_has_no_floor_without_repeats():
    """Ruling I2: a measurement's floor is its Repeat SD from 3 pairs, else
    none (2 SE alone). The Overall score keeps the noise threshold."""
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.4, 0.42, 0.38]] * 2, predicted=5.0, sd=0.5)
              + _errors("Juiciness", [[0.0, 0.1, -0.1]] * 2, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-0.8, -0.9, -0.7]] * 2))
    found = insights.steady_bias(opt, checks)
    # Firmness +0.4 is a bias with no floor; the score's −0.8 is under 1 point.
    assert [(b.name, round(b.mean, 2), b.rounds) for b in found] == [("Firmness", 0.4, [2, 3])]


def test_steady_bias_lists_the_overall_score_first():
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.4, 0.42, 0.38]] * 2, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0, -4.2, -3.8]] * 2))
    assert [b.name for b in insights.steady_bias(opt, checks)] == [insights.SCORE, "Firmness"]


# ------------------------------------------------------------------ #
#  B1, B2: Method and details' Mean error and Fitted noise SD columns
# ------------------------------------------------------------------ #

class _Gp:
    """A display model whose fitted noise sd is `sd` in the output's units."""
    def __init__(self, sd):
        import torch
        self.likelihood = SimpleNamespace(noise=torch.tensor([sd ** 2]))
        self.outcome_transform = SimpleNamespace(stdvs=torch.tensor([1.0]))


def test_details_carry_each_outputs_mean_error_and_its_bias():
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.4, 0.42, 0.38]] * 2, predicted=5.0, sd=0.5)
              + _errors("Juiciness", [[0.3, -0.5, 0.1]] * 2, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0, -4.2, -3.8]] * 2))
    rows = {r.name: r for r in insights.details(opt, checks)}
    assert rows["Firmness"].mean_error == pytest.approx(0.4) and rows["Firmness"].bias
    assert rows["Juiciness"].mean_error == pytest.approx(-0.0333, abs=1e-3)
    assert not rows["Juiciness"].bias
    assert rows[insights.SCORE].mean_error == pytest.approx(-4.0) and rows[insights.SCORE].bias


def test_details_carry_the_fitted_noise_sd_or_say_it_is_fixed():
    opt = _patty(rows=6)
    checks = _errors(insights.SCORE, [[0.1, -0.1, 0.0]] * 2)
    fitted = {insights.SCORE: insights.Fitted(insights.SCORE, _Gp(1.3), None, 6, 0.0, 100.0)}
    [row] = insights.details(opt, checks, fitted)
    assert row.fitted_noise == pytest.approx(1.3) and not row.noise_fixed
    opt.bo_config = {"kernel": "rbf", "noise": "fixed_tiny"}
    [row] = insights.details(opt, checks, fitted)
    assert row.fitted_noise is None and row.noise_fixed


def _computed(opt, checks, fitted=None):
    return {"checks": checks, "fitted": fitted or {}}


# ------------------------------------------------------------------ #
#  I2: Next step's order — repeat, unreliable, early, no_skill, the rest;
#  B1: "check" only from a disagreeing Repeat, worded conditionally
# ------------------------------------------------------------------ #

@pytest.fixture
def order(monkeypatch):
    """next_step with the verdict, the Repeat, the bias and the skill set
    directly."""
    opt = _patty(rows=6)
    state = {"verdict": "within", "gap": None, "repeat": None, "skill": None}

    def headline(o, checks):
        score = (None if state["gap"] is None
                 else insights.ScoreRecord(10, 10, state["gap"], rounds=[3, 4, 5]))
        return insights.Headline("verdict", [4, 5], 10, 10, "holding", score, [])
    monkeypatch.setattr(insights, "headline", headline)
    monkeypatch.setattr(insights, "verdict_state", lambda o, h, c: state["verdict"])
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda o, c, m=None: state["repeat"])
    monkeypatch.setattr(insights, "skill", lambda o, c, name=insights.SCORE: state["skill"])
    monkeypatch.setattr(insights, "improvement",
                        lambda o, facts=None, spread=None, yardstick=None: None)

    def run(standing=None, **kw):
        state.update(kw)
        return insights.next_step(opt, [], standing)
    run.opt = opt
    return run


def _fact(value, n=16):
    return insights.SkillFact(value, n, [3, 4, 5], insights._skill_state(value, n))


_REPEAT = insights.Disagreement(21, 11, -10.5, 20, 7.4)


def test_a_disagreeing_repeat_comes_before_an_unreliable_verdict(order):
    step = order(verdict="unreliable", gap=-5.4, repeat=_REPEAT, skill=_fact(-1.2))
    assert (step.kind, step.disagreement) == ("repeat", _REPEAT)


def test_there_is_no_bias_row_any_more(order):
    for verdict in ("unreliable", "within", "caution"):
        assert order(verdict=verdict, gap=-6.0).kind != "bias"


def test_unreliable_comes_before_early_and_no_skill(order):
    assert order(verdict="unreliable", skill=_fact(-1.2)).kind == "misses"


def test_early_comes_before_no_skill(order):
    assert order(verdict="early", skill=_fact(-1.2)).kind == "early"
    assert order(verdict="within", skill=_fact(-1.2)).kind == "no_skill"


# ------------------------------------------------------------------ #
#  B2: the 1-point threshold is assumed below 3 pairs; Next step asks for
#  a Repeat of Best so far. I13: the Repeat SD says its df.
# ------------------------------------------------------------------ #

def test_the_yardstick_is_assumed_below_three_pairs():
    opt = _patty(rows=6)
    y = insights.noise_yardstick(opt)
    assert (y.value, y.pairs, y.assumed) == (1.0, 0, True)
    _repeat_of(opt, 0, 3)
    y = insights.noise_yardstick(opt)
    assert (y.value, y.pairs, y.assumed, y.df, y.groups) == (1.0, 1, True, 1, 1)


def test_next_step_carries_the_assumed_threshold(order):
    assert order(verdict="within").assumed


# ------------------------------------------------------------------ #
#  I4 (A3): effect claims need 10 Overall-score checks and skill 0.20
# ------------------------------------------------------------------ #

def _skill_checks(n, value_error=1.0):
    """`n` fully measured Overall-score checks of Round 3 on _patty(6),
    each `value_error` from its prediction (a skill set by the spread of
    the measured scores around the earlier rows' average)."""
    return [_check(insights.SCORE, 3, 100 + k, 60.0 + (k % 5) * 3 + value_error,
                   60.0 + (k % 5) * 3, 5.0) for k in range(n)]


def test_below_ten_checks_effects_wait_with_the_count():
    opt = _patty(rows=6)
    gate = insights.skill_gate(opt, _skill_checks(6))
    assert (gate.state, gate.n) == ("wait", 6)
    assert insights.skill_gate(opt, ()).n == 0


@pytest.mark.parametrize("value,state", [(-0.5, "no_skill"), (-0.1, "unclear"),
                                         (0.19, "unclear"), (0.2, None), (0.9, None)])
def test_effects_need_skill_point_two(value, state):
    opt = _patty(rows=6)
    gate = insights.skill_gate(opt, [], fact=_fact(value, 24))
    assert (None if gate is None else gate.state) == state
    if gate is not None:
        assert gate.n == 24 and gate.fact.value == value


def test_the_noise_guard_withholds_effects_by_the_skill_gate():
    opt = _patty(rows=6)
    for value, fires in ((0.09, True), (0.36, False)):
        g = insights.noise_guard(opt, None, [], fact=_fact(value, 24))
        assert g.fires == fires and (g.reasons == ["skill"]) == fires
        assert (g.gate is not None) == fires
    wait = insights.noise_guard(opt, None, _skill_checks(6))
    assert wait.skill_only and wait.gate.state == "wait"
    # A measurement's own guard has no skill trigger (review I3 scope).
    assert insights.noise_guard(opt, None, _skill_checks(6), name="Firmness").reasons == []


    # One wording with the effect line (polish round, tests/test_106_last_fixes.py).


@pytest.mark.parametrize("value", [None, 0.09])
def test_next_step_under_the_gate_falls_through_to_rising_or_see(order, monkeypatch, value):
    """Review I4: untested, close and explore cite the model's predicted
    best; while effects are withheld they fall through."""
    fact = None if value is None else _fact(value, 24)
    untested = SimpleNamespace(higher=True, tested_near=False, binding=[])
    assert order(untested, skill=fact).kind == "see"
    monkeypatch.setattr(insights, "improvement", lambda *a, **k: insights.Improvement(
        "barely", 0.5, None, 0))
    close = SimpleNamespace(higher=False, tested_near=True, binding=[])
    assert order(close, skill=fact).kind == "see"
    monkeypatch.setattr(insights, "improvement", lambda *a, **k: insights.Improvement(
        "yes", 3.0, None, 0))
    assert order(untested, skill=fact).kind == "rising"


def test_with_skill_the_model_rows_stay(order):
    untested = SimpleNamespace(higher=True, tested_near=False, binding=[])
    assert order(untested, skill=_fact(0.36, 24)).kind == "untested"


# ------------------------------------------------------------------ #
#  Pins on the campaign's own projects and the demo projects (slow: each
#  project's Round-ahead checks are fitted once per module)
# ------------------------------------------------------------------ #

from tests import demo_projects as D  # noqa: E402
from tests import e2e_fixtures as E  # noqa: E402

NOISY_RECOMMENDED = "03_bar_noisy (Recommended)"


def _load(name, last_round=None):
    if name == NOISY_RECOMMENDED:
        return E.load(E.BAR_NOISY, last_round=last_round, recommended=True)
    if name in E.NAMES:
        return E.load(name, last_round=last_round)
    return D.load(name)


@pytest.fixture(scope="module")
def project():
    """name -> (project, its Round-ahead checks), fitted once. A Round's
    checks depend only on the rows before it and its own rows, so a
    project stopped at Round r has the checks of Rounds up to r
    (`upto`)."""
    seen = {}

    def get(name):
        if name not in seen:
            opt = _load(name)
            seen[name] = (opt, insights.check_rounds(opt))
        return seen[name]
    return get


def upto(project, name, r):
    opt, checks = project(name)
    return _load(name, last_round=r), [c for c in checks if c.round <= r]


def test_02_falls_through_to_rising_or_see_under_the_gate(project):
    """02 at Round 5: skill −0.10 (not clearly better than the average), so
    a predicted best in untested amounts is not offered as the next step."""
    opt, checks = upto(project, E.PATTY, 5)
    untested = SimpleNamespace(higher=True, tested_near=False, binding=[])
    assert insights.skill_gate(opt, checks).state == "unclear"
    assert insights.next_step(opt, checks, untested).kind in ("rising", "see")


# ------------------------------------------------------------------ #
#  I6 (A5): the expected-improvement sentence names its reference
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  The tab itself, on the campaign's projects
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def tabs(tmp_path_factory):
    """02 after Round 3, 02 final (also with Method and details open) and
    03 drift, each opened on Insights as the app opens a saved project."""
    import os

    import bo_views
    from streamlit.testing.v1 import AppTest

    from food_bo import FoodOptimizer
    from tests.test_app_ui import APP_PATH

    def install(name):
        state = E.state(name)
        opt = FoodOptimizer(state["project_name"])
        opt.import_json(state)
        opt.save()
        return opt.project_name

    def open_tab(project, fold=None):
        at = AppTest.from_file(APP_PATH, default_timeout=300)
        at.session_state["_loaded_project"] = project
        at.run()
        at.session_state["main_tab"] = wording.TAB_INSIGHTS
        if fold is not None:
            at.session_state[fold] = True
        at.run()
        assert not at.exception
        return at

    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("w1tabs"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (E.PATTY_ROUND3, E.PATTY, E.BAR_DRIFT):
                out[name] = open_tab(install(name))
            out["02 how"] = open_tab(install(E.PATTY), ui_insights.HOW_KEY)
            yield out
    finally:
        os.chdir(here)


def _tab_texts(at):
    tab = at.tabs[3]
    return [m.value for m in tab.markdown] + [c.value for c in tab.caption]


def test_the_drift_tab_makes_no_causal_claim(tabs):
    said = " ".join(_tab_texts(tabs[E.BAR_DRIFT]))
    assert 'If something changed (panel, supplier, equipment), check it before relying on the next suggestions.' not in said and "panel" not in said


