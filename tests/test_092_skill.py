"""0.8.0 BO views, Task 2 (spec 2026-09-30-bo-views §2, §8; review B3,
I1-I4): Round-ahead skill against the average of earlier results, over
every Round checked, pinned on the demo projects; its place in the next
step and in Make a round's track line; and the display-side noise guard.
The suggestion model is not touched."""
import dataclasses
import random
import statistics
from types import SimpleNamespace

import pytest
import torch

import insights
from tests import coverage_sim as sim
from tests import demo_projects as demo


@pytest.fixture(scope="module")
def lines():
    """{name: (opt, checks, fitted Overall-score model)} for Lines A-C."""
    out = {}
    for name in (demo.LINE_A, demo.LINE_B, demo.LINE_C):
        opt = demo.load(name)
        checks = insights.check_rounds(opt)
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
        out[name] = (opt, checks, score)
    return out


# ------------------------------------------------------------------ #
#  Skill (review B3, I2)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name,value,n,state", [
    (demo.LINE_A, 0.36, 15, "skill"),
    (demo.LINE_B, -1.23, 16, "no_skill"),
    (demo.LINE_C, -0.47, 16, "no_skill"),
])
def test_the_demo_lines_skill_is_pinned(lines, name, value, n, state):
    """Over every Round checked, as 02 measured it (review B3)."""
    opt, checks, _ = lines[name]
    fact = insights.skill(opt, checks)
    assert fact.value == pytest.approx(value, abs=0.02)
    assert (fact.n, fact.state, fact.rounds) == (n, state, [2, 3, 4, 5])


def test_each_measurement_has_its_own_skill(lines):
    """Spec §2.1 'per output' (02 I1's table, Line A)."""
    opt, checks, _ = lines[demo.LINE_A]
    got = {o['name']: insights.skill(opt, checks, o['name']).value for o in opt.objectives}
    assert got == pytest.approx({"Firmness": 0.43, "Juiciness": -0.04, "Cook loss": 0.45},
                                abs=0.02)


def _check(no, rnd, measured, predicted, name=insights.SCORE):
    return insights.Check(no, rnd, name, predicted, predicted - 5, predicted + 5, measured,
                          True, True, predicted, 2.5)


def _opt(values):
    """A stand-in project: row i measured `values[i]`, Round 1 = rows 0-4,
    Round 2 = the rest."""
    return SimpleNamespace(
        batch_history=[1] * 5 + [2] * (len(values) - 5),
        Y_history=list(values), is_set_aside=lambda i: False,
        fully_measured=lambda i: True)


def test_nothing_is_said_below_ten_checks_or_with_nothing_to_divide_by(monkeypatch):
    monkeypatch.setattr(insights, "measured", lambda opt, name, i: opt.Y_history[i])
    opt = _opt([50.0] * 5 + [60.0] * 9)
    nine = [_check(6 + k, 2, 60.0, 58.0) for k in range(9)]
    assert insights.skill(opt, nine) is None
    flat = _opt([50.0] * 15)
    assert insights.skill(flat, [_check(6 + k, 2, 50.0, 49.0) for k in range(10)]) is None


@pytest.mark.parametrize("miss,value,state", [(12.0, -0.18, "no_skill"),
                                               (10.5, 0.10, "small"), (1.0, 0.99, "skill")])
def test_the_three_states(monkeypatch, miss, value, state):
    """Review I2: ≤ 0, 0-0.2, ≥ 0.2; errors pooled over the checks. The
    earlier rows average 50; results 60 and 62 (squared errors of the
    average: 122 on average); every prediction `miss` below its result."""
    monkeypatch.setattr(insights, "measured", lambda opt, name, i: opt.Y_history[i])
    opt = _opt([50.0] * 5 + [60.0, 62.0] * 5)
    checks = [_check(6 + k, 2, 60.0 + 2 * (k % 2), 60.0 + 2 * (k % 2) - miss)
              for k in range(10)]
    fact = insights.skill(opt, checks)
    assert fact.value == pytest.approx(value, abs=0.01)
    assert fact.state == state and fact.n == 10


def test_only_fully_measured_overall_scores_count(monkeypatch):
    monkeypatch.setattr(insights, "measured", lambda opt, name, i: opt.Y_history[i])
    opt = _opt([50.0] * 5 + [60.0] * 10)
    checks = [_check(6 + k, 2, 60.0, 59.0) for k in range(10)]
    checks[0] = dataclasses.replace(checks[0], fully_measured=False)
    assert insights.skill(opt, checks) is None


@pytest.mark.parametrize("n,allowed", [(10, 1), (12, 2), (35, 3), (36, 4)])
def test_allowed_misses(n, allowed):
    """03 §8: the largest miss count the verdict still calls As expected."""
    assert insights.allowed_misses(n) == allowed
    assert insights.binomial_tail(allowed, n) >= 0.10
    assert insights.binomial_tail(allowed + 1, n) < 0.10


# The trust audit's truth for the "good" simulated projects (review-trust
# sim.py, copied): smooth, no drift, noise known — a correctly specified
# problem, so a model that predicts the expected score is a correct model.
def _truth(r):
    p = (r["Pea protein"] - 15) / 15
    o = (r["Coconut oil"] - 5) / 10
    b = (r["Beet juice"] - 1) / 4
    return {"Firmness": 2.0 + 5.5 * p - 1.2 * o + 1.5 * p * o - 1.0 * p * p,
            "Juiciness": 3.0 + 4.0 * o - 1.5 * p + 1.2 * b - 1.5 * o * o,
            "Cook loss": 30 - 10 * p - 9 * o + 6 * p * o,
            "Colour": 2 + 6 * b - 1.5 * b * b + 1.0 * p}


def _expected_score(opt, recipe, noise, draws=100):
    t, rng, scores = _truth(recipe), random.Random(0), []
    for _ in range(draws):
        scores.append(opt._compute_utility({
            o['name']: min(o['max_val'], max(o['min_val'], t[o['name']] + rng.gauss(
                0, noise * (o['max_val'] - o['min_val'])))) for o in opt.objectives}))
    return statistics.mean(scores)


def test_the_plan_gate_keeps_the_no_skill_cut_at_zero():
    """Review I2's gate: a correct model (the expected score) on the 24
    "good" simulated projects' own measured scores, every window of 10, 12
    and 20 consecutive checks. At or below 0 in fewer than 1 window in 10
    at every n, so NO_SKILL stays 0 (measured: 3.7 %, 4.0 %, 2.3 %)."""
    assert insights.NO_SKILL == 0.0
    paths = {p.stem: p for p in sim.projects()}
    windows = {10: [], 12: [], 20: []}
    for entry in sim.fixture()["projects"]:
        if entry["meta"]["scen"] != "good":
            continue
        opt, meta = sim.load(paths[entry["project"]])
        ids = [int(x) for x in opt.formulation_ids]
        checks = [dataclasses.replace(c, predicted=_expected_score(
                      opt, opt.recipe_history[ids.index(c.formulation)], meta["noise"]))
                  for c in sim.checks_of(entry)
                  if c.name == insights.SCORE and c.fully_measured]
        for n, got in windows.items():
            for start in range(len(checks) - n + 1):
                got.append(insights.skill(opt, checks[start:start + n]).value)
    rates = {n: sum(v <= insights.NO_SKILL for v in got) / len(got)
             for n, got in windows.items()}
    assert rates == pytest.approx({10: 0.037, 12: 0.040, 20: 0.023}, abs=0.005)
    assert all(rate < 0.10 for rate in rates.values())


# ------------------------------------------------------------------ #
#  Next step and Make a round's track line (review I1)
# ------------------------------------------------------------------ #

def test_no_skill_is_the_next_step_before_untested_and_close(lines):
    opt, checks, _ = lines[demo.LINE_C]
    standing = SimpleNamespace(higher=True, tested_near=False, binding=[])
    step = insights.next_step(opt, checks, standing)
    assert step.kind == "no_skill" and step.skill.value == pytest.approx(-0.47, abs=0.02)


def test_an_unreliable_verdict_still_comes_first(lines):
    """Line B: the panel shift is the verdict's (bias), not "no skill"."""
    opt, checks, _ = lines[demo.LINE_B]
    assert insights.next_step(opt, checks).kind == "bias"


def test_a_model_with_skill_keeps_its_next_step(lines):
    opt, checks, _ = lines[demo.LINE_A]
    assert insights.next_step(opt, checks).kind != "no_skill"


def test_the_track_line_carries_no_skill_even_when_coverage_holds(lines):
    opt, checks, _ = lines[demo.LINE_C]
    t = insights.round_track(opt, checks)
    assert t.no_skill is not None and t.no_skill.n == 16
    assert t.record and not t.rough and t.rounds
    opt_a, checks_a, _ = lines[demo.LINE_A]
    assert insights.round_track(opt_a, checks_a).no_skill is None


@pytest.mark.parametrize("state", ["over_wide"])
def test_no_skill_is_said_alone_where_coverage_is_not(lines, monkeypatch, state):
    opt, checks, _ = lines[demo.LINE_C]
    monkeypatch.setattr(insights, "verdict_state", lambda *a: state)
    t = insights.round_track(opt, checks)
    assert t is not None and not t.record and t.no_skill is not None


@pytest.mark.parametrize("state", ["early", "wide"])
def test_no_line_while_early_or_wide(lines, monkeypatch, state):
    opt, checks, _ = lines[demo.LINE_C]
    monkeypatch.setattr(insights, "verdict_state", lambda *a: state)
    assert insights.round_track(opt, checks) is None


# ------------------------------------------------------------------ #
#  The noise guard (review I3)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name,reasons", [
    (demo.LINE_A, []),
    (demo.LINE_B, ["skill"]),
    (demo.LINE_C, ["floor", "skill"]),
])
def test_the_guard_on_the_demo_lines(lines, name, reasons):
    """Line C's Overall-score fit sits on BoTorch's correlation-length floor
    (02 C2); Line B has no skill (a panel shift, said as "not reported
    until…", never "fitting noise"); Line A is clean."""
    opt, checks, score = lines[name]
    guard = insights.noise_guard(opt, score, checks)
    assert guard.reasons == reasons
    assert guard.skill_only == (reasons == ["skill"])


def test_the_floor_is_the_kernels_own(lines):
    """Review I3 (a): read from the module, so an expert kernel with
    another bound is judged by its own."""
    opt, checks, score = lines[demo.LINE_C]
    kernel = next(m for m in score.model.covar_module.modules()
                  if hasattr(m, "raw_lengthscale_constraint"))
    assert float(kernel.raw_lengthscale_constraint.lower_bound) == pytest.approx(0.025)
    assert insights._lengthscale_at_floor(score.model)
    opt_a, _, score_a = lines[demo.LINE_A]
    assert not insights._lengthscale_at_floor(score_a.model)


class _Noise:
    """A fitted model stand-in: its noise sd in points, and no kernel."""

    def __init__(self, sd_points, cls="GaussianLikelihood"):
        self.covar_module = torch.nn.Module()
        self.likelihood = type(cls, (), {"noise": torch.tensor([(sd_points / 4.0) ** 2])})()
        self.outcome_transform = SimpleNamespace(stdvs=torch.tensor([[4.0]]))


def test_the_noise_trigger_needs_three_pairs_and_a_fitted_noise(monkeypatch, lines):
    """Review I3 (b): √noise × SD(train Y) below half the Repeat SD, from
    3 pairs or more; never under fixed_tiny, whose noise is fixed."""
    opt, checks, _ = lines[demo.LINE_A]
    fitted = SimpleNamespace(model=_Noise(0.4))
    yard = lambda pairs: (lambda o, c=(), name=insights.SCORE:
                          insights.Yardstick(1.0, 1.0, pairs, pairs >= 3))
    monkeypatch.setattr(insights, "noise_yardstick", yard(3))
    assert insights.noise_guard(opt, fitted, checks).reasons == ["noise"]
    monkeypatch.setattr(insights, "noise_yardstick", yard(2))
    assert insights.noise_guard(opt, fitted, checks).reasons == []
    monkeypatch.setattr(insights, "noise_yardstick", yard(3))
    assert insights.noise_guard(opt, SimpleNamespace(model=_Noise(0.6)), checks).reasons == []
    assert insights.noise_guard(
        opt, SimpleNamespace(model=_Noise(0.4, "FixedNoiseGaussianLikelihood")), checks
    ).reasons == []
    monkeypatch.setattr(opt, "bo_config", {"noise": "fixed_tiny"}, raising=False)
    assert insights.noise_guard(opt, fitted, checks).reasons == []


def test_a_measurements_guard_has_no_skill_trigger(lines):
    """Review I3 scope: (c) is for the Overall score only."""
    opt, checks, _ = lines[demo.LINE_C]
    firm = insights.fit(opt, "Juiciness", list(range(len(opt.Y_history))))
    assert "skill" not in insights.noise_guard(opt, firm, checks, name="Juiciness").reasons


def test_nothing_here_changes_a_suggestion(lines):
    """The guard and the skill read the display models only: ask() is not
    reached, and the global RNG is where it was."""
    opt, checks, score = lines[demo.LINE_C]
    state = torch.get_rng_state()
    insights.skill(opt, checks)
    insights.noise_guard(opt, score, checks)
    assert torch.equal(state, torch.get_rng_state())
