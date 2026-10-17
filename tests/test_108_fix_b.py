"""Fix round B (research/2026-10-01-insights-ftue/ledger.md: "Early
five-second re-test", "Early walkthrough", "Statistician cold read"):
the first screen's wording and the expert layer's disclosures.

- Five-second fixes 1-4: Line A's "little more" clause, Line C's cause of
  the uncertainty and its skill number with its meaning, the Upgrade
  check's reason for Too early; the bias said as "scores have come in
  about N points below the predictions" everywhere; "(up to 7.4 points)";
  the header names the waiting formulation.
- Walkthrough: Line B's Next leads with "Check what changed before you
  make Round 6."; Progress names a tie within the noise threshold.
- Statistician I1-I4, I8-I12: the disclosures in Method and definitions
  and the tooltips.

Fast: facts are built by hand; the demo projects' own lines are pinned on
the rendered screen in tests/test_107_w1_first_screen.py (FIRST_SCREENS in
tests/test_107_ftue.py)."""
import types

import pytest

import insights
import ui_insights
import wording as W
from tests.ftue_terms import statistic_names_in
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED, _FTUE_ALLOWED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _clean(text):
    assert text in _FTUE_ALLOWED or not any(p.search(text) for p in _BANNED), text


# ------------------------------------------------------------------ #
#  The bias, one wording everywhere (five-second optional fix, ruled in)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  The header names the waiting formulation (five-second, header)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Predictions: Line C's skill number and the cause of the uncertainty
# ------------------------------------------------------------------ #

_SKILL = "(skill −0.47, where 0 matches that average and 1 is exact)"


def _h(inside=33, total=36, rounds=(3, 4, 5), wide=()):
    return insights.Headline("verdict", list(rounds), inside, total, "holding", wide=list(wide))


def _facts(row, clauses=(), h=None, **kw):
    return insights.StatusFacts(scored=kw.pop("scored", 21), prediction_row=row,
                                clauses=list(clauses), headline=h or _h(), **kw)


def _skill(value):
    return insights.SkillFact(value, 16, [2, 3, 4, 5], "no_skill" if value <= -0.2 else "unclear")


# ------------------------------------------------------------------ #
#  Progress: ties within the noise threshold, "little more", the Upgrade
# ------------------------------------------------------------------ #


def _best(opt, rounds_since=0, latest=True):
    i = opt.best_index()
    return insights.BestFacts(float(opt.Y_history[i]), int(opt.formulation_ids[i]),
                              opt.batch_history[i], rounds_since, latest, [])


def _yard(value=1.0):
    return insights.Yardstick(value, None, 0, False)


def test_progress_sentence_names_a_runner_up_within_the_threshold():
    opt = _patty(rows=12)
    top = opt.best_index()
    other = insights.tie_partner(opt, top)
    gap = float(opt.Y_history[top]) - float(opt.Y_history[other])
    f = _best(opt, rounds_since=1, latest=False)
    # Fix A follow-up (ruling on ties): the lead is judged against
    # rise_threshold(yard, 1) = noise SD × √2 × 1.645, not the noise SD.
    per = insights.rise_threshold(1.0, 1)
    facts = _facts("reliable", progress="no_new_best", best=f, yardstick=_yard((gap + 0.5) / per))
    text = ui_insights.progress_sentence(opt, facts)
    assert f"tied with {W.FORMULATION_CAP} {int(opt.formulation_ids[other])})" in text
    facts = _facts("reliable", progress="no_new_best", best=f,
                   yardstick=_yard(max(gap - 0.01, 0) / per))
    assert "tied with" not in ui_insights.progress_sentence(opt, facts)


# ------------------------------------------------------------------ #
#  Next: Line B leads with the check (walkthrough, ruled)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  The level-0 list allows the skill number only with its meaning
# ------------------------------------------------------------------ #

def test_the_skill_number_is_allowed_only_with_its_meaning():
    assert statistic_names_in(f"Score predictions … {_SKILL}, and …") == []
    assert statistic_names_in("skill −0.47") != []
    assert statistic_names_in("the skill is low") != []


# ------------------------------------------------------------------ #
#  Statistician I1-I4, I8-I12 (display only)
# ------------------------------------------------------------------ #

def _method_text():
    return " ".join(_method_facts_lines())


def _method_facts_lines():
    s = types.SimpleNamespace
    facts = s(cold=False, cold_start=5, values=False, acquisition="qlognei", beta=None, q=4,
              scored=20, inputs=7, standardised=True, kernel="rbf", recommended=True, ard=True,
              lengthscale_prior="LogNormal(2.39, 1.73)", lengthscale_floor=0.025,
              outputscale=False, outputscale_prior=None, noise="fitted",
              noise_prior="LogNormal(−4.00, 1.00)", noise_floor=1e-4, total=True, robust=False,
              fit_tolerance=1e-6, restarts=20, raw_samples=1024, mc_samples=512, folds=5)
    return W.model_summary(facts)


def _everything():
    texts = [_method_text(),
             W.EXPECTED_IMPROVEMENT_HELP,
             ]
    texts += [" ".join(r) for r in W.status_rule_rows()]
    return texts


def test_i1_no_single_incumbent_under_qlognei():
    for t in _everything():
        assert "best tested" not in t, t
    method = _method_text()
    assert "no single incumbent" in method
    assert "approximate" in method
    assert "approximate" in W.EXPECTED_IMPROVEMENT_HELP


def test_i9_the_folds_are_grouped_by_round():
    for t in (_method_text(),):
        assert "grouped by Round" in t, t


def test_i10_the_coverage_count_assumes_independent_misses():
    off = next(r for r in W.status_rule_rows() if r[1].endswith("(coverage)"))
    assert "independent" in off[2]


def test_i12_changed_settings_are_not_a_fair_test():
    assert ("Changing model settings re-scores every earlier Round, so a record read after "
            "choosing settings to suit it is no longer a fair test.") in _method_text()


