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


@pytest.mark.parametrize("miss,value,state", [(13.0, -0.385, "no_skill"),
                                               (12.0, -0.18, "unclear"),
                                               (10.5, 0.10, "unclear"), (1.0, 0.99, "skill")])
def test_the_three_states(monkeypatch, miss, value, state):
    """Fix round 1 (ledger ruling on T2 I-1): at or below the cut c_n (−0.2
    at every n the gate measured), no better; below 0.2, not clearly
    better; 0.2 or more, better; errors pooled over the checks. The
    earlier rows average 50; results 60 and 62 (squared errors of the
    average: 122 on average); every prediction `miss` below its result."""
    monkeypatch.setattr(insights, "measured", lambda opt, name, i: opt.Y_history[i])
    opt = _opt([50.0] * 5 + [60.0, 62.0] * 5)
    checks = [_check(6 + k, 2, 60.0 + 2 * (k % 2), 60.0 + 2 * (k % 2) - miss)
              for k in range(10)]
    fact = insights.skill(opt, checks)
    assert fact.value == pytest.approx(value, abs=0.01)
    assert fact.state == state and fact.n == 10 and fact.cut == pytest.approx(-0.2)


@pytest.mark.parametrize("n", [10, 11, 12, 16, 20, 30, 55])
def test_the_state_boundaries(n):
    """Review T2 m-2: the cut itself reads no better, just above it not
    clearly better, 0.19 not clearly better, 0.2 and 0.21 better."""
    cut = insights.skill_cut(n)
    assert cut == pytest.approx(-0.2)
    assert insights._skill_state(cut, n) == "no_skill"
    assert insights._skill_state(cut + 1e-9, n) == "unclear"
    assert insights._skill_state(0.0, n) == "unclear"
    assert insights._skill_state(0.19, n) == "unclear"
    assert insights._skill_state(0.2, n) == "skill"
    assert insights._skill_state(0.21, n) == "skill"


def test_the_cut_follows_the_gate_table(monkeypatch):
    """c_n is the gate's 10th percentile at the largest tabulated n at or
    below n, capped at SKILL_CUT_MAX: a gate that measured lower moves it."""
    monkeypatch.setattr(insights, "SKILL_GATE_Q10", {10: -0.35, 12: -0.25, 20: 0.1})
    assert insights.skill_cut(10) == pytest.approx(-0.35)
    assert insights.skill_cut(11) == pytest.approx(-0.35)
    assert insights.skill_cut(12) == pytest.approx(-0.25)
    assert insights.skill_cut(19) == pytest.approx(-0.25)
    assert insights.skill_cut(40) == pytest.approx(-0.2)


def test_n_counts_the_checks_pooled(monkeypatch):
    """Review T2 m-3: a check whose Round has no baseline is not pooled,
    and not counted."""
    monkeypatch.setattr(insights, "measured", lambda opt, name, i: opt.Y_history[i])
    opt = _opt([50.0] * 5 + [60.0, 62.0] * 5)
    checks = [_check(6 + k, 2, 60.0 + 2 * (k % 2), 59.0 + 2 * (k % 2)) for k in range(10)]
    lost = [_check(1, 1, 50.0, 50.0)]      # Round 1 has no rows before it
    fact = insights.skill(opt, checks + lost)
    assert fact.n == 10 and fact.rounds == [2]
    assert insights.skill(opt, checks[:9] + lost) is None


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


def _gate_windows(reps=20, seed=2, ns=tuple(insights.SKILL_GATE_Q10)):
    """Fix round 1 (ledger ruling on T2 I-1; the T2 review's p3): a correct
    model with finite data — its Round-ahead prediction is the true expected
    score plus an error whose sd is its own claimed uncertainty (the check's
    predictive sd² less the score's noise variance) — on the 24 "good"
    simulated projects, every window of n consecutive checks, `reps` draws.
    {n: [skill of each window whose true (expected) skill is 0.2 or more]}."""
    def draws(opt, recipe, noise, k=200):
        t, rng, out = _truth(recipe), random.Random(0), []
        for _ in range(k):
            out.append(opt._compute_utility({
                o['name']: min(o['max_val'], max(o['min_val'], t[o['name']] + rng.gauss(
                    0, noise * (o['max_val'] - o['min_val'])))) for o in opt.objectives}))
        return statistics.mean(out), statistics.stdev(out)

    paths = {p.stem: p for p in sim.projects()}
    projects = []
    for entry in sim.fixture()["projects"]:
        if entry["meta"]["scen"] != "good":
            continue
        opt, meta = sim.load(paths[entry["project"]])
        ids = [int(x) for x in opt.formulation_ids]
        first = dict(insights.round_order(opt))
        rows = []
        for c in sim.checks_of(entry):
            if c.name != insights.SCORE or not c.fully_measured:
                continue
            mu, noise = draws(opt, opt.recipe_history[ids.index(c.formulation)], meta["noise"])
            seen = [insights.measured(opt, insights.SCORE, i) for i in range(first[c.round])]
            seen = [v for v in seen if v is not None]
            epi = max(0.0, (c.sd or 0) ** 2 - noise ** 2) ** 0.5
            rows.append((c, mu, noise, epi, sum(seen) / len(seen)))
        projects.append((opt, rows))
    rng, out = random.Random(seed), {n: [] for n in ns}
    for _ in range(reps):
        for opt, rows in projects:
            checks = [dataclasses.replace(c, predicted=mu + rng.gauss(0, epi))
                      for c, mu, _, epi, _ in rows]
            for n in ns:
                for s in range(len(rows) - n + 1):
                    w = rows[s:s + n]
                    true = 1 - (sum(e ** 2 + z ** 2 for _, _, z, e, _ in w)
                                / sum((mu - b) ** 2 + z ** 2 for _, mu, z, _, b in w))
                    if true >= 0.2:
                        out[n].append(insights.skill(opt, checks[s:s + n]).value)
    return out


def test_the_gate_sets_the_no_skill_cut():
    """The ruling's gate: c_n = the 10th percentile of that correct model's
    skill over windows with true skill ≥ 0.2, never above −0.2. Measured
    (ftue fix A's Round-grouped folds; before: 0.00, 0.03, 0.11, 0.25,
    0.32, 0.45): 0.11, 0.11, 0.14, 0.22, 0.31, 0.34 at n = 10, 12, 15, 20,
    25, 30, all above −0.2, so the cut is −0.2 at every n; there, a correct
    model is called "no better" in fewer than 1 window in 10 (about 4 % at
    10; 7 % before)."""
    windows = _gate_windows()
    q10 = {n: statistics.quantiles(v, n=10)[0] for n, v in windows.items()}
    assert q10 == pytest.approx(insights.SKILL_GATE_Q10, abs=0.01)
    for n, got in windows.items():
        cut = insights.skill_cut(n)
        assert cut == min(insights.SKILL_GATE_Q10[n], -0.2)
        assert sum(v <= cut for v in got) / len(got) < 0.10
    # The demo lines keep their reading under the cut (ledger ruling).
    assert -1.23 <= insights.skill_cut(16) and -0.47 <= insights.skill_cut(16)


# ------------------------------------------------------------------ #
#  Next step and Make a round's track line (review I1)
# ------------------------------------------------------------------ #

def test_no_skill_is_the_next_step_before_untested_and_close(lines):
    opt, checks, _ = lines[demo.LINE_C]
    standing = SimpleNamespace(higher=True, tested_near=False, binding=[])
    step = insights.next_step(opt, checks, standing)
    assert step.kind == "no_skill" and step.skill.value == pytest.approx(-0.47, abs=0.02)


def test_the_disagreeing_repeat_then_the_verdict_come_before_no_skill(lines):
    """Line B: the panel shift is said through its disagreeing Repeat (F21),
    the one row that says to check (0.8.0 e2e fixes, review B1, I2), not
    "no skill"."""
    opt, checks, _ = lines[demo.LINE_B]
    step = insights.next_step(opt, checks)
    assert step.kind == "repeat" and step.disagreement.later == 21


def test_a_model_with_skill_keeps_its_next_step(lines):
    opt, checks, _ = lines[demo.LINE_A]
    assert insights.next_step(opt, checks).kind != "no_skill"


@pytest.mark.parametrize("state", ["early", "wide", "none"])
def test_early_and_wide_come_before_no_skill(lines, monkeypatch, state):
    """Review T2 I-3: no_skill sits after early/wide — an "early" project is
    never told its predictions are no better than the average."""
    opt, checks, _ = lines[demo.LINE_C]
    monkeypatch.setattr(insights, "verdict_state", lambda *a: state)
    standing = SimpleNamespace(higher=True, tested_near=False, binding=[])
    assert insights.next_step(opt, checks, standing).kind == "early"


def test_a_disagreeing_repeat_comes_before_no_skill(lines, monkeypatch):
    """Review T2 I-3: and after the repeat row."""
    opt, checks, _ = lines[demo.LINE_C]
    flagged = insights.Disagreement(1, 2, 3.0, 1, 0.5)
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda *a, **k: flagged)
    step = insights.next_step(opt, checks)
    assert step.kind == "repeat" and step.disagreement is flagged


@pytest.mark.parametrize("gain,kind", [(None, "close"), (0.0, "close"), (1.0, "close"),
                                       (1.08, "explore")])
def test_close_to_the_top_needs_the_expected_improvement_under_the_threshold_too(
        lines, monkeypatch, gain, kind):
    """Fix round 1 (ledger ruling C1): nothing predicted clearly higher and
    Best so far not rising is "close to the top" only when the expected
    improvement Progress shows is at or under the noise threshold (1
    point here) too; above it, "explore". None: none shown."""
    opt, checks, _ = lines[demo.LINE_A]
    monkeypatch.setattr(insights, "improvement", lambda *a, **k: SimpleNamespace(
        state="not_lately"))
    monkeypatch.setattr(insights, "noise_yardstick", lambda *a, **k: insights.Yardstick(
        1.0, None, 0, False))
    standing = SimpleNamespace(higher=False, tested_near=True, binding=[])
    step = insights.next_step(opt, checks, standing, gain=gain)
    assert step.kind == kind
    # "close" carries the gain and the threshold it fell under (specific
    # wording, 6de8364), so the line can name both numbers.
    assert step.gain == gain
    assert step.noise == (1.0 if kind == "close" else None)


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


@pytest.mark.parametrize("how", ["no_record", "score_wide"])
def test_no_skill_is_said_alone_without_a_record_or_with_the_score_wide(lines, monkeypatch, how):
    """Review T2 m-4: the other two skill-alone paths."""
    opt, checks, _ = lines[demo.LINE_C]
    real = insights.headline

    def headline(o, c):
        h = real(o, c)
        if how == "no_record":
            return dataclasses.replace(h, score=None)
        return dataclasses.replace(h, wide=[insights.SCORE])
    monkeypatch.setattr(insights, "headline", headline)
    t = insights.round_track(opt, checks)
    assert t is not None and not t.record and t.no_skill is not None
    opt_a, checks_a, _ = lines[demo.LINE_A]
    assert insights.round_track(opt_a, checks_a) is None


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


def _kernel_gp(kernel):
    """A fitted-model stand-in with one kernel and no fitted noise."""
    return SimpleNamespace(covar_module=kernel, likelihood=None)


@pytest.mark.parametrize("factor,fires", [(1.04, True), (1.06, False)])
def test_the_floor_margin_is_five_per_cent_of_the_actual_length(factor, fires):
    """Review T2 I-3: GUARD_FLOOR's 1.05, on the ACTUAL correlation length
    of a softplus-constrained kernel (raw and actual differ there)."""
    from gpytorch.constraints import GreaterThan
    from gpytorch.kernels import RBFKernel
    kernel = RBFKernel(ard_num_dims=2, lengthscale_constraint=GreaterThan(0.025))
    kernel.lengthscale = torch.tensor([[0.025 * factor, 0.5]])
    assert float(kernel.raw_lengthscale.min()) < 0      # softplus: raw is not the length
    assert insights._lengthscale_at_floor(_kernel_gp(kernel)) is fires


@pytest.mark.parametrize("cfg", [{"kernel": "matern52"}, {"kernel": "matern32"},
                                 {"kernel": "rbf"}])
@pytest.mark.parametrize("length,fires", [(0.052, True), (0.053, False), (0.3, False)])
def test_an_expert_kernel_is_judged_against_an_absolute_floor(cfg, length, fires):
    """Fix round 1 (ledger ruling on T2 I-2): an expert's Matern or RBF has
    gpytorch's Positive() bound, 0; trigger (a) then uses 0.05 on the
    normalised inputs (within GUARD_FLOOR's 5 %)."""
    from food_bo import _build_covar
    kernel = _build_covar(cfg, 3)
    base = kernel.base_kernel
    assert float(base.raw_lengthscale_constraint.lower_bound) == 0.0
    base.lengthscale = torch.tensor([[length, 0.4, 0.9]])
    assert insights._lengthscale_at_floor(_kernel_gp(kernel)) is fires


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
    # Independent pairs: as many degrees of freedom as pairs (ftue fix A,
    # cold read I11: the gate counts df).
    yard = lambda pairs: (lambda o, c=(), name=insights.SCORE:
                          insights.Yardstick(1.0, 1.0, pairs, pairs >= 3, df=pairs))
    monkeypatch.setattr(insights, "noise_yardstick", yard(3))
    assert insights.noise_guard(opt, fitted, checks).reasons == ["noise"]
    monkeypatch.setattr(insights, "noise_yardstick", yard(2))
    assert insights.noise_guard(opt, fitted, checks).reasons == []
    monkeypatch.setattr(insights, "noise_yardstick", yard(3))
    assert insights.noise_guard(opt, SimpleNamespace(model=_Noise(0.6)), checks).reasons == []
    # Review T2 m-2: either side of GUARD_NOISE × a yardstick of 1.0.
    assert insights.noise_guard(opt, SimpleNamespace(model=_Noise(0.49)), checks).reasons == [
        "noise"]
    assert insights.noise_guard(opt, SimpleNamespace(model=_Noise(0.51)), checks).reasons == []
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
