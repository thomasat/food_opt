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


def test_the_steady_bias_line_names_every_biased_output_once():
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.4, 0.42, 0.38]] * 2, predicted=5.0, sd=0.5)
              + _errors("Juiciness", [[-0.6, -0.62, -0.58]] * 2, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0, -4.2, -3.8]] * 2))
    line = ui_insights.steady_bias_text(opt, checks)
    assert line == wording.steady_bias_line(
        high=[wording.bias_item(wording.OVERALL_SCORE_COLUMN, "4.0"),
              wording.bias_item("Juiciness", "0.6", "")],
        low=[wording.bias_item("Firmness", "0.4", "")], rounds=[2, 3])
    assert "check" not in line.lower()
    _clean(line)


def test_no_steady_bias_no_line():
    opt = _patty(rows=6)
    assert ui_insights.steady_bias_text(opt, _errors(insights.SCORE, [[0.1, -0.1, 0.0]] * 2)) == ""


def test_the_score_line_names_its_rounds_and_leaves_the_bias_to_its_own_line():
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.1] * 10] * 3, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0, -4.2, -3.8]] * 3))
    line = ui_insights.score_line(opt, checks)
    assert line == wording.score_gap_line(9, 9, -4.0, "", n=9, rounds=[2, 3, 4])
    assert line.startswith("Overall score, Rounds 2–4: ") and "read about" not in line


def test_make_a_rounds_track_line_names_every_round_checked():
    """Overrides I1 (speed review I16): the track line is the Overall score's
    record over every Round checked, so it names those Rounds."""
    opt = _patty(rows=6)
    checks = (_errors("Firmness", [[0.4, -0.4] * 5] * 5, predicted=5.0, sd=0.5)
              + _errors(insights.SCORE, [[-4.0, -4.2, -3.8]] * 5))
    t = insights.round_track(opt, checks)
    assert t.rounds == [2, 3, 4, 5, 6] and t.total == 15
    assert t.gap == pytest.approx(-4.0)
    assert ui_insights.track_text(t).startswith("Overall score, Rounds 2–6: 15 of 15")


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


def test_unreliable_is_the_misses_row_and_carries_a_steady_bias_as_a_fact(order):
    step = order(verdict="unreliable", gap=-4.1)
    assert (step.kind, step.gap, step.bias_rounds) == ("misses", -4.1, [3, 4, 5])
    text, change = ui_insights.next_step_text(order.opt, step, rounds=[4, 5])
    assert text == wording.next_misses([4, 5], bias_text=wording.next_bias(
        "4.1", below=True, rounds=[3, 4, 5]))
    assert change is None and "check" not in text.lower()
    assert order(verdict="unreliable", gap=None).gap is None


def test_there_is_no_bias_row_any_more(order):
    for verdict in ("unreliable", "within", "caution"):
        assert order(verdict=verdict, gap=-6.0).kind != "bias"


def test_unreliable_comes_before_early_and_no_skill(order):
    assert order(verdict="unreliable", skill=_fact(-1.2)).kind == "misses"


def test_early_comes_before_no_skill(order):
    assert order(verdict="early", skill=_fact(-1.2)).kind == "early"
    assert order(verdict="within", skill=_fact(-1.2)).kind == "no_skill"


def test_only_the_repeat_row_says_check_and_it_is_conditional(order):
    step = order(repeat=_REPEAT)
    text, _ = ui_insights.next_step_text(order.opt, step)
    assert text.endswith(wording.CHECK_IF_CHANGED)
    assert "Formulation 21, a Repeat of Formulation 11" in text
    for kind in ("misses", "early", "see", "rising", "explore"):
        other, _ = ui_insights.next_step_text(order.opt, insights.NextStep(kind, gap=-3.0))
        assert wording.CHECK_IF_CHANGED not in other and "Check" not in other, kind


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


def test_three_pairs_measure_the_threshold_and_say_their_df():
    opt = _patty(rows=6)
    for dst in (1, 2):
        _repeat_of(opt, 0, dst)
    for i, v in zip((0, 1, 2), (70.0, 72.0, 71.0)):
        opt.Y_history[i] = v
    y = insights.noise_yardstick(opt)
    # Ftue fix A (cold read I11): 3 pairs on 2 df do not yet measure it.
    assert (y.pairs, y.from_repeats, y.assumed, y.df, y.groups) == (3, False, True, 2, 1)
    assert ui_insights.repeats_text(y) == wording.stand_repeats(
        f"{y.sd:.1f}", 3, df=2, groups=1)
    # Fix round B (statistician I11): the df, not the pairs; fix round A
    # moves the gate itself to 3 df.
    assert ", 2 df (3 results of 1 formulation)" in ui_insights.repeats_text(y)


def test_the_threshold_text_says_assumed_below_three_pairs():
    assumed = insights.Yardstick(1.0, None, 0, False, assumed=True)
    assert ui_insights.threshold_text(assumed) == wording.beat_by(None, assumed=True, pairs=0)
    assert "(assumed: no Repeats yet;" in ui_insights.threshold_text(assumed)
    assert ui_insights.threshold_text(assumed, assumed=False) == wording.beat_by()
    two = insights.Yardstick(1.0, 0.8, 2, False, assumed=True)
    assert "(assumed: 2 pairs of Repeats, 3 needed;" in ui_insights.threshold_text(two)
    measured = insights.Yardstick(2.7, 2.7, 3, True)
    assert ui_insights.threshold_text(measured) == wording.beat_by("2.7")


def test_rising_and_within_noise_say_the_threshold_is_assumed():
    """The 02 Round 3 pin's words (review B2): '…above the 1-point noise
    threshold (assumed …)'."""
    yard = insights.Yardstick(1.0, None, 0, False, assumed=True)
    # Fix A follow-up (decision 4): the threshold (√2 × 1.645 × 1 = 2.3)
    # and its make-up, the assumed noise threshold its last factor.
    yes = ui_insights.stand_text(insights.Improvement("yes", 6.3, None, 0, 2.326, 1), yard)
    assert yes == wording.stand_yes("6.3", wording.rise_needs(
        "2.3", "√2 × 1.64", wording.beat_by(None, assumed=True)))
    barely = ui_insights.stand_text(insights.Improvement("barely", 0.4, None, 0, 2.326, 1), yard)
    assert barely == ("Within noise: up 0.4 points in the last Round, under the 2.3 points "
                      "needed: √2 × 1.64 × " + wording.beat_by(None, assumed=True) + ".")
    # Measured, or not said to be assumed: today's words.
    assert ui_insights.stand_text(insights.Improvement("yes", 6.3, None, 0, 2.326, 1)) == \
        wording.stand_yes("6.3", wording.rise_needs("2.3", "√2 × 1.64", wording.beat_by()))
    _clean(yes, barely)


def test_the_assumed_form_is_said_once_on_progress(monkeypatch):
    """Line 2 says "assumed" only when line 1 did not name the threshold:
    the reader is told once (the owner's sparse-screen rule)."""
    opt = _patty(rows=6)
    yard = insights.Yardstick(1.0, None, 0, False, assumed=True)
    p = insights.Prediction(80.0, 75.0, 85.0, 80.0, 2.0)
    best = insights.PredictedBest({}, p, None, 0, 1.0, p)
    standing = SimpleNamespace(best=best, higher=False, tested_near=True, binding=[],
                               close_no=None, luck=False)
    monkeypatch.setattr(insights, "improvement", lambda *a, **k: insights.Improvement(
        "yes", 6.3, None, 0))
    assert "(assumed" not in ui_insights.standing_text(opt, standing, yard)
    monkeypatch.setattr(insights, "improvement", lambda *a, **k: insights.Improvement(
        "not_lately", None, None, 3))
    assert "(assumed: no Repeats yet;" in ui_insights.standing_text(opt, standing, yard)


@pytest.mark.parametrize("kind", ["close", "explore", "see", "rising"])
def test_next_step_asks_for_a_repeat_while_the_threshold_is_assumed(kind):
    opt = _patty(rows=6)
    text, _ = ui_insights.next_step_text(opt, insights.NextStep(kind, assumed=True))
    plain, _ = ui_insights.next_step_text(opt, insights.NextStep(kind))
    assert text == f"{plain} {wording.NEXT_REPEAT_TO_MEASURE}"
    assert wording.NEXT_REPEAT_TO_MEASURE not in plain


@pytest.mark.parametrize("kind", ["misses", "early", "untested"])
def test_the_other_rows_do_not_ask_for_a_repeat(kind):
    opt = _patty(rows=6)
    text, _ = ui_insights.next_step_text(opt, insights.NextStep(kind, assumed=True))
    assert wording.NEXT_REPEAT_TO_MEASURE not in text


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


def test_the_guard_line_says_why_effects_wait():
    opt = _patty(rows=6)
    wait = insights.noise_guard(opt, None, _skill_checks(6))
    assert ui_insights.guard_text(wait) == wording.effects_wait_line(6)
    assert ui_insights.guard_text(wait, settings_only=True) == wording.effects_wait_line(6, True)
    unclear = insights.noise_guard(opt, None, [], fact=_fact(0.09, 24))
    assert ui_insights.guard_text(unclear) == wording.effects_unclear_line("0.09", 24)
    none = insights.noise_guard(opt, None, [], fact=_fact(-0.47, 16))
    assert ui_insights.guard_text(none) == wording.guard_skill_line()
    both = insights.Guard(["floor", "skill"], insights.SkillGate("unclear", 24, _fact(0.09, 24)))
    assert ui_insights.guard_text(both) == wording.guard_line([wording.GUARD_FLOOR])


    # One wording with the effect line (polish round, tests/test_106_last_fixes.py).


def test_the_expected_improvement_says_why_it_is_withheld():
    gate = insights.SkillGate("wait", 6)
    assert ui_insights.gain_reason("skill", gate) == wording.gain_reason_too_few(6)
    gate = insights.SkillGate("unclear", 24, _fact(0.09, 24))
    assert ui_insights.gain_reason("skill", gate) == wording.gain_reason_unclear("0.09", 24)
    assert ui_insights.gain_reason("skill", insights.SkillGate("no_skill", 16, _fact(-0.5))) \
        == wording.GAIN_REASON_SKILL
    assert ui_insights.gain_reason("verdict", None) == wording.GAIN_REASON_VERDICT
    assert ui_insights.gain_reason("guard", None) == wording.GAIN_REASON_GUARD


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


def _said(opt, checks):
    """What Prediction check and Next step say from the checks alone."""
    h = insights.headline(opt, checks)
    state = insights.verdict_state(opt, h, checks)
    step = insights.next_step(opt, checks, None)
    texts = [ui_insights.verdict_text(opt, h, state, checks=checks),
             ui_insights.score_line(opt, checks), ui_insights.steady_bias_text(opt, checks),
             ui_insights.skill_text(opt, checks),
             ui_insights.next_step_text(opt, step, h.rounds)[0]]
    flagged = insights.disagreeing_repeat(opt, checks)
    if flagged is not None:
        texts.append(ui_insights.disagreement_text(flagged))
    return step, [t for t in texts if t]


@pytest.mark.parametrize("name", [E.PATTY, E.BAR_CLEAN, E.BAR_DRIFT])
def test_no_check_advice_at_any_round_of_02_clean_or_drift(project, name):
    """Ruling B1: 02 and 03 clean show no "check" at any Round (their
    biases are the model's), and the drift run makes no causal claim: the
    Round-ahead check cannot see a change it learned from its first Round.
    Fix A's follow-up (owner, decision 2): 03 clean's F16, a Repeat of F12
    2.84 points apart, beat √2 × 1.96 × the assumed 1 point (2.77) at
    Rounds 4-6; until the Repeat SD has 3 df a pair must also beat its own
    predictive limit, which it does not, so fix A's exception is gone.
    03 drift at Round 7: F20, a Repeat of F14, is 1.62 points off against
    √2 × t_{0.975, 3} × 0.49 = 2.20 (the other Repeats' SD on 3 df; t,
    not 1.96, whenever the SD comes from Repeats), so no flag."""
    for r in range(3, 9):
        opt, checks = upto(project, name, r)
        step, texts = _said(opt, checks)
        assert step.kind != "repeat", (name, r)
        for t in texts:
            assert wording.CHECK_IF_CHANGED not in t and "Check" not in t, (name, r, t)
            assert "changed" not in t and "panel" not in t, (name, r, t)


def test_the_drift_runs_repeat_is_reported_with_its_numbers(project):
    """03 drift: F20, a Repeat of F14 across the drift, agrees, so it is
    part of the Repeat SD the screen states with its basis, not a check."""
    opt, checks = project(E.BAR_DRIFT)
    assert insights.disagreeing_repeat(opt, checks) is None
    yard = insights.noise_yardstick(opt, checks)
    assert (yard.pairs, yard.df, yard.groups, yard.from_repeats) == (4, 4, 4, True)
    assert ui_insights.repeats_text(yard) == (
        "Repeat SD of the Overall score: 0.7 points, 4 df (8 results of 4 "
        "formulations); this is the noise threshold.")


def test_line_b_keeps_check_through_its_disagreeing_repeat(project):
    opt, checks = project(D.LINE_B)
    step, texts = _said(opt, checks)
    assert (step.kind, step.disagreement.later, step.disagreement.earlier) == ("repeat", 21, 11)
    # Fix A's follow-up (decision 2): under 3 df the pair beats both limits,
    # √2 × 1.96 × the assumed 1 point (2.77) and its own predictive limit.
    d = step.disagreement
    assert (d.basis, d.df) == ("assumed", 0)
    assert abs(d.diff) > d.limit >= d.predictive > 2.77
    text = ui_insights.next_step_text(opt, step)[0]
    assert text.startswith("Formulation 21, a Repeat of Formulation 11, scored 10.5 points lower.")
    assert text.endswith(wording.CHECK_IF_CHANGED)


def test_02_round_3_reads_assumed_and_asks_for_a_repeat(project):
    """Ruling B2's pin: 02 R3 (the app's own copy after Round 3) reads
    '…above the 1-point noise threshold (assumed …)' and Next step asks
    for a Repeat. F11's +6.3 was a +7 noise draw; only a Repeat tells."""
    opt, checks = project(E.PATTY_ROUND3)
    yard = insights.noise_yardstick(opt, checks)
    answer = insights.improvement(opt, yardstick=yard)
    # Fix A follow-up (decision 4): the threshold of a Round of 4, 3.2.
    assert ui_insights.stand_text(answer, yard) == (
        "Rising: up 6.3 points in the last Round, above the 3.2 points needed: √2 × 2.23 "
        "(best of 4) × the 1-point noise threshold (assumed: no Repeats yet; a Repeat of Best "
        "so far would measure it).")
    step = insights.next_step(opt, checks, None)
    assert step.kind == "rising"
    assert ui_insights.next_step_text(opt, step)[0].endswith(wording.NEXT_REPEAT_TO_MEASURE)


@pytest.mark.parametrize("name,state", [
    (NOISY_RECOMMENDED, "unclear"), (E.PATTY, "unclear"), (E.PATTY_ROUND3, "wait"),
    (D.LINE_A, None), (D.TIMING, None), (E.BAR_CLEAN, None), (E.BAR_DRIFT, None)])
def test_the_a3_gate(project, name, state):
    """Ruling I4's pins: the 03 noisy run (Recommended settings) and 02 lose
    their effect claims, and say why; Line A, the Timing check, 03 clean and
    03 drift keep them."""
    opt, checks = project(name)
    gate = insights.skill_gate(opt, checks)
    assert (None if gate is None else gate.state) == state
    guard = insights.noise_guard(opt, None, checks)
    assert ("skill" in guard.reasons) == (state is not None)
    if name == E.PATTY:
        assert ui_insights.guard_text(guard) == wording.effects_unclear_line("0.09", 24)
    if name == NOISY_RECOMMENDED:
        assert ui_insights.guard_text(guard) == wording.effects_unclear_line("0.08", 20)


def test_02_falls_through_to_rising_or_see_under_the_gate(project):
    """02 at Round 5: skill −0.10 (not clearly better than the average), so
    a predicted best in untested amounts is not offered as the next step."""
    opt, checks = upto(project, E.PATTY, 5)
    untested = SimpleNamespace(higher=True, tested_near=False, binding=[])
    assert insights.skill_gate(opt, checks).state == "unclear"
    assert insights.next_step(opt, checks, untested).kind in ("rising", "see")


@pytest.mark.parametrize("name,line", [
    (E.PATTY, "Results have come in below the predictions: Overall score by about 3.7 points "
              "(Rounds 3–8)."),
    (E.BAR_DRIFT, ""),
    (D.LINE_A, ""),
])
def test_the_steady_bias_on_the_campaign_projects(project, name, line):
    """02's predictions ran about 4 points high from its first checked Round
    (the model's optimism at its maximum): a fact, no advice."""
    opt, checks = project(name)
    assert ui_insights.steady_bias_text(opt, checks) == line


def test_03_clean_says_its_bias_both_ways_without_advice(project, monkeypatch):
    """Under the t rule (ftue fix A, cold read I5) 03 clean's Crumbliness
    bias no longer shows; its Overall score's still does. The both-ways
    wording is checked under the earlier 2-standard-error rule."""
    opt, checks = project(E.BAR_CLEAN)
    # Re-pinned: fix round B says a bias as results coming in above or below
    # the predictions (statistician's cold read), not "Predictions have run".
    assert ui_insights.steady_bias_text(opt, checks) == (
        "Results have come in above the predictions: Overall score by about 2.1 points "
        "(Rounds 3–8).")
    monkeypatch.setattr(insights, "t_critical", lambda df, p=0.975: 2.0)
    line = ui_insights.steady_bias_text(opt, checks)
    assert line.startswith("Results have come in below the predictions: Crumbliness")
    assert "; above them: Overall score by about 2.1 points" in line and line.endswith(
        "(Rounds 3–8).")


# ------------------------------------------------------------------ #
#  I6 (A5): the expected-improvement sentence names its reference
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("acquisition,ref", [
    (None, "the highest the model draws for the tested formulations (approximate)"),
    ("qlogei", "Best so far's measured {best}")])
def test_the_expected_improvement_is_an_average_gain_over_its_reference(
        monkeypatch, acquisition, ref):
    """Review I6: under qLogNEI (the default) the reference is the highest
    the model draws for the tested formulations (fix round B, statistician
    I1: no single incumbent); under qLogEI, Best so far's measured score.
    Never "the best single new formulation beats … by"."""
    import bo_views
    opt = _patty(rows=6)
    if acquisition:
        opt.bo_config = {"acquisition": acquisition}
    monkeypatch.setattr(insights, "skill_gate", lambda o, c, fact=None: None)
    monkeypatch.setattr(ui_insights, "latest_gain", lambda *a, **k: bo_views.RoundGain(
        None, 80.0, None, 2.66, True, None))
    yard = insights.noise_yardstick(opt)
    got = ui_insights.gain_text(opt, {"checks": [], "fitted": {}}, yard)
    best = f"{opt.Y_history[opt.best_index()]:.2f}"
    assert got == wording.gain_line("2.66", None, wording.beat_by(), above=True,
                                    measured_text=best if acquisition else None)
    assert ref.format(best=best) in got and "counting no gain as zero" in got
    assert "beats" not in got


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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_02_round_3_tab_says_assumed_once_and_asks_for_a_repeat(tabs):
    texts = _tab_texts(tabs[E.PATTY_ROUND3])
    said = " ".join(texts)
    assert said.count("(assumed:") == 1, said
    # Fix A follow-up (decision 4): the threshold, then its make-up.
    assert any(t.startswith("Rising: up 6.3 points in the last Round, above the 3.2 points "
                            "needed: √2 × 2.23 (best of 4) × the 1-point noise threshold "
                            "(assumed: no Repeats yet;") for t in texts)
    nxt = texts[texts.index(f"##### {wording.Q_NEXT}") + 1]
    assert nxt.endswith(wording.NEXT_REPEAT_TO_MEASURE)
    # Fewer than 10 checks: effects and the expected gain wait.
    assert wording.gain_unreliable(wording.gain_reason_too_few(4)) in texts


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_02_final_tab_states_its_bias_and_never_says_check(tabs):
    texts = _tab_texts(tabs[E.PATTY])
    assert ("Results have come in below the predictions: Overall score by about 3.7 points "
            "(Rounds 3–8).") in texts
    said = " ".join(texts)
    assert wording.CHECK_IF_CHANGED not in said and "Check the panel" not in said
    nxt = texts[texts.index(f"##### {wording.Q_NEXT}") + 1]
    assert nxt.startswith("Measured values in Rounds 6–8 fell outside")
    assert nxt.endswith("Overall scores have come in about 3.7 points below the predictions "
                        "(Rounds 3–8).")
    assert wording.gain_unreliable(wording.GAIN_REASON_VERDICT) in texts


def test_the_drift_tab_makes_no_causal_claim(tabs):
    said = " ".join(_tab_texts(tabs[E.BAR_DRIFT]))
    assert wording.CHECK_IF_CHANGED not in said and "panel" not in said


