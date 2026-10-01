"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2; review
R-B1, R-I9, R-I11): the three status lines' facts.

insights.status_facts reads headline, verdict_state, skill,
disagreeing_repeat, score_steady_bias, noise_guard, improvement,
best_facts and tie_partner, and sets one Predictions row (worst first, the
first that holds wins), its clauses, one Progress word and the Next row.
Every state of the review's rule table is pinned here with the inputs
stood in for, so each test reads as one row of that table; the demo
projects' real states are pinned on the rendered screen
(tests/test_107_w1_first_screen.py)."""
import types

import pytest

import insights
import ui_results
import wording
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _check(name, rnd, no, full=True):
    return insights.Check(no, rnd, name, 50.0, 48.0, 52.0, 50.0, True, full, 50.0, 1.0)


def _score_checks(n, rnd=3):
    return [_check(insights.SCORE, rnd, 100 + k) for k in range(n)]


def _skill(state, value=None, n=15):
    value = {"skill": 0.36, "unclear": 0.1, "no_skill": -0.47}[state] if value is None else value
    return insights.SkillFact(value, n, [2, 3, 4, 5], state)


def _headline(state="verdict", wide=()):
    return insights.Headline(state, [3, 4, 5], 32, 35, "holding",
                             insights.ScoreRecord(13, 15, -0.2, 1.0, [2, 3, 4, 5]), list(wide))


def _stand_in(monkeypatch, verdict="within", flagged=None, fact=None, bias=None, wide=(),
              guard=(), h_state="verdict"):
    """The inputs status_facts reads, stood in for: one row of the table."""
    monkeypatch.setattr(insights, "headline", lambda opt, checks: _headline(h_state, wide))
    monkeypatch.setattr(insights, "verdict_state", lambda opt, h, checks: verdict)
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda opt, checks, matches=None: flagged)
    monkeypatch.setattr(insights, "skill", lambda opt, checks, name=insights.SCORE: fact)
    monkeypatch.setattr(insights, "score_steady_bias", lambda opt, checks, flagged=None: bias)
    monkeypatch.setattr(insights, "fit", lambda opt, name, rows: None)
    monkeypatch.setattr(insights, "noise_guard",
                        lambda opt, fitted, checks=(), name=insights.SCORE, fact=None:
                        insights.Guard(list(guard)))


_REPEAT = insights.Disagreement(21, 11, -10.54, 20, 7.4)


@pytest.fixture
def opt():
    return _patty(rows=12)


# ------------------------------------------------------------------ #
#  Predictions: R-B1 rows 1-9, worst first
# ------------------------------------------------------------------ #

# (verdict, disagreeing Repeat, skill, Overall-score checks) -> row.
_ROWS = [
    # Row 1: a disagreeing Repeat, coverage not unreliable.
    ("within", _REPEAT, _skill("skill"), 15, "off_repeat"),
    ("caution", _REPEAT, _skill("no_skill"), 15, "off_repeat"),
    ("early", _REPEAT, None, 2, "off_repeat"),
    # Row 2: unreliable coverage; with a disagreeing Repeat too, the coverage
    # sentence is used and Next carries the Repeat (Line B).
    ("unreliable", None, _skill("skill"), 15, "off_coverage"),
    ("unreliable", _REPEAT, _skill("no_skill"), 16, "off_coverage"),
    # Row 4: no Round checked, too few values, or skill not computed yet.
    ("none", None, None, 0, "too_few"),
    ("early", None, None, 2, "too_few"),
    ("early", None, _skill("no_skill"), 12, "too_few"),
    ("within", None, None, 8, "too_few"),
    ("over_wide", None, None, 9, "too_few"),
    # Row 5: more misses than expected, said before too few checks
    # (integration ruling: a warning is never folded).
    ("caution", None, None, 9, "caution"),
    ("caution", None, _skill("no_skill"), 15, "caution"),
    ("caution", None, _skill("skill"), 15, "caution"),
    # Row 6: every output very uncertain.
    ("wide", None, _skill("no_skill"), 15, "wide"),
    ("wide", None, _skill("skill"), 15, "wide"),
    # Rows 7, 8: no skill, not clearly better.
    ("within", None, _skill("no_skill"), 16, "no_skill"),
    ("over_wide", None, _skill("no_skill"), 16, "no_skill"),
    ("within", None, _skill("unclear"), 16, "unclear"),
    # Ten checks or more, every one on its baseline: no skill to say.
    ("within", None, None, 12, "unclear"),
    # Row 9.
    ("within", None, _skill("skill"), 15, "reliable"),
    ("over_wide", None, _skill("skill"), 15, "reliable"),
]


@pytest.mark.parametrize("verdict,flagged,fact,n,row", _ROWS)
def test_each_state_sets_its_predictions_row(monkeypatch, opt, verdict, flagged, fact, n, row):
    _stand_in(monkeypatch, verdict=verdict, flagged=flagged, fact=fact)
    facts = insights.status_facts(opt, _score_checks(n))
    assert facts.prediction_row == row
    assert facts.predictions == insights.PREDICTION_STATUS[row]
    assert facts.checked == n


def test_the_gate_comes_before_any_check(monkeypatch):
    opt = _patty(rows=3)
    _stand_in(monkeypatch, verdict="none")
    facts = insights.status_facts(opt, ())
    assert (facts.prediction_row, facts.predictions, facts.scored) == ("gate", "not_yet", 3)
    # Next has no row at the gate: the screen says its tail alone.
    assert facts.next is None and facts.clauses == []


def test_nothing_scored_and_nothing_waiting_has_no_status_lines(monkeypatch):
    opt = _patty(rows=0)
    assert insights.status_facts(opt, ()) is None


def test_every_row_is_reached_by_some_state():
    """The table above covers all nine rows (the gate by its own test)."""
    assert {r for *_, r in _ROWS} | {"gate"} == set(insights.PREDICTION_ROWS)


# ------------------------------------------------------------------ #
#  The clauses (R-B1, R-I11), in their order
# ------------------------------------------------------------------ #

def test_some_outputs_wide_is_a_clause_naming_them(monkeypatch, opt):
    _stand_in(monkeypatch, verdict="within", fact=_skill("no_skill"),
              wide=["Firmness", "Juiciness", "Cook loss"])
    facts = insights.status_facts(opt, _score_checks(16))
    assert facts.clauses == [("wide", ["Firmness", "Juiciness", "Cook loss"])]


def test_every_output_wide_is_the_row_not_a_clause(monkeypatch, opt):
    _stand_in(monkeypatch, verdict="wide", fact=_skill("skill"),
              wide=["Firmness", "Juiciness", insights.SCORE])
    facts = insights.status_facts(opt, _score_checks(15))
    assert facts.prediction_row == "wide" and facts.clauses == []


def test_the_clauses_come_in_the_rulings_order(monkeypatch, opt):
    _stand_in(monkeypatch, verdict="over_wide", fact=_skill("skill"), bias=1.2,
              wide=["Cook loss"], guard=["floor"])
    facts = insights.status_facts(opt, _score_checks(15))
    assert facts.prediction_row == "reliable"
    assert facts.clauses == [("wide", ["Cook loss"]), ("over_wide",), ("bias", 1.2), ("guard",)]
    assert facts.bias == 1.2


@pytest.mark.parametrize("verdict,flagged,fact,n", [
    ("within", _REPEAT, _skill("skill"), 15),     # row 1 says the Repeat alone
])
def test_off_repeat_takes_no_clause(monkeypatch, opt, verdict, flagged, fact, n):
    _stand_in(monkeypatch, verdict=verdict, flagged=flagged, fact=fact, bias=-5.5,
              wide=["Firmness"])
    facts = insights.status_facts(opt, _score_checks(n))
    assert facts.clauses == [] and facts.disagreement is _REPEAT


def test_off_coverage_carries_the_bias_alone(monkeypatch, opt):
    _stand_in(monkeypatch, verdict="unreliable", flagged=_REPEAT, fact=_skill("no_skill"),
              bias=-5.5, wide=["Cook loss"])
    facts = insights.status_facts(opt, _score_checks(16))
    assert facts.clauses == [("bias", -5.5)]


@pytest.mark.parametrize("reasons,said", [(["floor"], True), (["noise"], True),
                                          (["floor", "noise"], True), (["skill"], False),
                                          ([], False)])
def test_the_guard_clause_is_floor_or_noise_under_reliable(monkeypatch, opt, reasons, said):
    _stand_in(monkeypatch, verdict="within", fact=_skill("skill"), guard=reasons)
    facts = insights.status_facts(opt, _score_checks(15))
    assert (("guard",) in facts.clauses) is said


def test_the_guard_clause_is_never_on_a_not_yet_row(monkeypatch, opt):
    _stand_in(monkeypatch, verdict="within", fact=_skill("unclear"), guard=["floor"])
    facts = insights.status_facts(opt, _score_checks(15))
    assert ("guard",) not in facts.clauses


# ------------------------------------------------------------------ #
#  Progress (R-B1, R-I10)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("state,word", [("yes", "rising"), ("barely", "within_noise"),
                                        ("not_lately", "flat")])
def test_improvements_answer_is_the_progress_word(monkeypatch, opt, state, word):
    _stand_in(monkeypatch, fact=_skill("skill"))
    monkeypatch.setattr(insights, "improvement",
                        lambda opt, facts=None, spread=None, yardstick=None:
                        insights.Improvement(state, 1.0, None, 2))
    facts = insights.status_facts(opt, _score_checks(15))
    assert facts.progress == word and facts.improvement.state == state


@pytest.mark.parametrize("since,word", [(1, "no_new_best"), (0, "too_early")])
def test_no_answer_is_no_new_best_or_too_early(monkeypatch, opt, since, word):
    _stand_in(monkeypatch, fact=_skill("skill"))
    monkeypatch.setattr(insights, "improvement",
                        lambda opt, facts=None, spread=None, yardstick=None: None)
    monkeypatch.setattr(insights, "best_facts",
                        lambda opt: insights.BestFacts(90.0, 5, 2, since, since == 0, []))
    facts = insights.status_facts(opt, _score_checks(15))
    assert facts.progress == word


def test_progress_on_a_real_project_follows_improvement():
    """Unpatched: 12 formulations, three to a Round; Best so far's Round
    decides the word."""
    opt = _patty(rows=12)
    f = insights.best_facts(opt)
    facts = insights.status_facts(opt, ())
    answer = insights.improvement(opt, f, yardstick=insights.noise_yardstick(opt))
    expected = ({"yes": "rising", "barely": "within_noise", "not_lately": "flat"}[answer.state]
                if answer else "no_new_best" if f.rounds_since == 1 else "too_early")
    assert facts.progress == expected and facts.best == f


# ------------------------------------------------------------------ #
#  Ties (R-I9): equal scores only, at two decimals; a Repeat is never one
# ------------------------------------------------------------------ #

def test_tie_partner_is_the_next_highest_different_formulation():
    opt = _patty(rows=6)
    order = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    best = order[0]
    assert insights.tie_partner(opt, best) == order[1]


def test_a_repeat_is_never_the_tie_partner():
    opt = _patty(rows=6)
    order = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    best, second = order[0], order[1]
    opt.recipe_history[second] = dict(opt.recipe_history[best])
    opt.X_history[second] = list(opt.X_history[best])
    assert insights.tie_partner(opt, best) == order[2]


def test_results_asks_insights_for_the_tie_partner(monkeypatch):
    opt = _patty(rows=6)
    monkeypatch.setattr(insights, "tie_partner", lambda opt, best: "asked")
    assert ui_results._tie_partner(opt, 0) == "asked"


def test_a_tie_at_level_0_needs_equal_scores(monkeypatch):
    opt = _patty(rows=6)
    order = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    best, second = order[0], order[1]
    _stand_in(monkeypatch)
    assert insights.status_facts(opt, ()).tie is None
    opt.Y_history[second] = float(opt.Y_history[best]) + 0.001     # 2 decimals: equal
    assert insights.status_facts(opt, ()).tie in (best, second)
    opt.Y_history[second] = float(opt.Y_history[best]) - 0.01
    assert insights.status_facts(opt, ()).tie is None


# ------------------------------------------------------------------ #
#  Luck (R-I9): at level 0 only under Reliable, with no guard clause
# ------------------------------------------------------------------ #

def _standing(luck=True):
    best = types.SimpleNamespace(repeat=insights.Prediction(90.0, 88.0, 92.0))
    return types.SimpleNamespace(luck=luck, best=best, higher=False, tested_near=True,
                                 binding=[])


@pytest.mark.parametrize("fact,guard,luck,said", [
    (_skill("skill"), [], True, True),
    (_skill("skill"), [], False, False),
    (_skill("skill"), ["floor"], True, False),
    (_skill("unclear"), [], True, False),
    (_skill("no_skill"), [], True, False),
])
def test_luck_is_said_only_under_reliable(monkeypatch, opt, fact, guard, luck, said):
    _stand_in(monkeypatch, fact=fact, guard=guard)
    facts = insights.status_facts(opt, _score_checks(15), standing=_standing(luck))
    assert facts.luck is said
    # The predicted Repeat the clause names rides with it, and only then.
    assert facts.luck_repeat == (90.0 if said else None)


def test_no_standing_no_luck(monkeypatch, opt):
    _stand_in(monkeypatch, fact=_skill("skill"))
    assert insights.status_facts(opt, _score_checks(15)).luck is False


# ------------------------------------------------------------------ #
#  Next: next_step's row, with R-I11's caution fall-through
# ------------------------------------------------------------------ #

def test_next_is_next_steps_row_past_the_gate(monkeypatch, opt):
    _stand_in(monkeypatch, fact=_skill("skill"))
    seen = {}

    def step(opt, checks, standing=None, gain=None):
        seen.update(standing=standing, gain=gain)
        return insights.NextStep("see")
    monkeypatch.setattr(insights, "next_step", step)
    standing = _standing()
    facts = insights.status_facts(opt, _score_checks(15), standing=standing, gain=1.1)
    assert facts.next.kind == "see" and seen == {"standing": standing, "gain": 1.1}


def _next_under(monkeypatch, verdict, higher=True, tested_near=False, since=0):
    """next_step with the verdict and the predicted best stood in for."""
    opt = _patty(rows=15)
    monkeypatch.setattr(insights, "headline", lambda opt, checks: _headline())
    monkeypatch.setattr(insights, "verdict_state", lambda opt, h, checks: verdict)
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda opt, checks, matches=None: None)
    monkeypatch.setattr(insights, "skill", lambda opt, checks, name=insights.SCORE: _skill("skill"))
    monkeypatch.setattr(insights, "improvement",
                        lambda opt, facts=None, spread=None, yardstick=None:
                        insights.Improvement("not_lately", None, None, 2))
    standing = types.SimpleNamespace(higher=higher, tested_near=tested_near, binding=[])
    return insights.next_step(opt, _score_checks(15), standing, gain=5.0)


@pytest.mark.parametrize("higher,tested_near,kind", [(True, False, "untested"),
                                                      (False, True, "explore")])
def test_next_step_unchanged_as_expected(monkeypatch, higher, tested_near, kind):
    assert _next_under(monkeypatch, "within", higher, tested_near).kind == kind


@pytest.mark.parametrize("higher,tested_near", [(True, False), (False, True)])
def test_under_caution_the_predicted_best_rows_fall_through(monkeypatch, higher, tested_near):
    """R-I11: untested, close and explore rest on the predicted best, which a
    More misses than expected record does not back: they fall through to
    rising or see, as under the I4 gate (display only)."""
    assert _next_under(monkeypatch, "caution", higher, tested_near).kind == "see"


# ------------------------------------------------------------------ #
#  The words exist for every key status_facts sets
# ------------------------------------------------------------------ #

def test_every_key_has_its_word():
    for row in insights.PREDICTION_ROWS:
        assert insights.PREDICTION_STATUS[row] in wording.PREDICTIONS_WORDS
    for state in insights.PROGRESS_STATES:
        assert state in wording.PROGRESS_WORDS
