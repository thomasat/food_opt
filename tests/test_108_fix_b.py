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
from tests.test_107_ftue import statistic_names_in
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

def test_the_bias_is_said_as_scores_coming_in_below_the_predictions():
    assert W.bias_fact("4.3") == "scores have come in about 4.3 points below the predictions"
    assert W.bias_fact("1.2", False) == "scores have come in about 1.2 points above the predictions"
    assert W.bias_fact("1.0") == "scores have come in about 1 point below the predictions"


def test_the_bias_wording_on_insights_and_make_a_round():
    # Predictions, rows 2 and 7 (Lines B and C).
    assert W.pred_off_coverage([3, 4, 5], 10, 36, 4, W.bias_fact("5.5")) == (
        "In Rounds 3–5, 10 of 36 results fell outside the range the app predicted, where up to "
        "4 are normal, and scores have come in about 5.5 points below the predictions.")
    assert "scores have come in about 4.3 points below the predictions" in W.pred_no_skill(
        ("4.3", True))
    # Next step's misses row and Prediction figures' Overall-score line.
    assert W.next_bias("4.1", rounds=[3, 4, 5, 6, 7]) == (
        "Overall scores have come in about 4.1 points below the predictions (Rounds 3–7).")
    # Make a round's track line.
    assert W.round_track_bias(9, 10, "5.0").endswith(
        "Overall scores have come in about 5.0 points below the predictions.")
    assert "Overall scores have come in about 1 point above the predictions" in (
        W.round_track_rough(7, 10, "1", below=False))
    # Prediction figures' every-output line.
    assert W.steady_bias_line(["Overall score by about 4.3 points"], ["Firmness by about 3.0 N"],
                              [3, 4, 5]) == (
        "Results have come in below the predictions: Overall score by about 4.3 points; above "
        "them: Firmness by about 3.0 N (Rounds 3–5).")
    assert W.steady_bias_line([], ["Firmness by about 3.0 N"]) == (
        "Results have come in above the predictions: Firmness by about 3.0 N.")
    # The status table and glossary name the clause by its new words.
    rows = W.status_rule_rows()
    assert any(r[1] == "Clause: scores have come in below or above the predictions" for r in rows)
    assert not any("have run" in " ".join(r) for r in rows)
    assert not any("have run" in " ".join(r) for r in W.GLOSSARY_ROWS)


# ------------------------------------------------------------------ #
#  The header names the waiting formulation (five-second, header)
# ------------------------------------------------------------------ #

def test_the_header_names_the_waiting_formulations():
    assert W.insights_header_line(20, 5, 1, "Cook loss", numbers=[19]) == (
        "Based on 20 scored formulations (all results in), up to Round 5. Formulation 19 is "
        "waiting for Cook loss and is not used yet.")
    assert W.insights_header_line(20, 5, 2, "Cook loss", numbers=[19, 22]) == (
        "Based on 20 scored formulations (all results in), up to Round 5. Formulations 19 and "
        "22 are waiting for Cook loss and are not used yet.")
    # Many: the count, as before.
    assert W.insights_header_line(20, 5, 4, "Cook loss", numbers=[19, 20, 21, 22]).endswith(
        " 4 more are waiting for Cook loss and are not used yet.")


def test_header_text_reads_the_waiting_formulations_numbers():
    from tests.test_complete_results import _tell
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    assert opt.awaited_rows() == [6]
    assert ui_insights.header_text(opt).endswith(
        f" {W.FORMULATION_CAP} 7 is waiting for Cook loss.")
    assert ui_insights.header_detail(opt).endswith(
        f" {W.FORMULATION_CAP} 7 is waiting for Cook loss and is not used yet.")


# ------------------------------------------------------------------ #
#  Predictions: Line C's skill number and the cause of the uncertainty
# ------------------------------------------------------------------ #

_SKILL = "(skill −0.47, where 0 matches that average and 1 is exact)"


def test_no_skill_gives_the_skill_number_with_its_meaning():
    text = W.pred_no_skill(("4.3", True), skill_text="−0.47")
    assert text.startswith("Score predictions do no better than simply using the average of "
                           f"earlier scores {_SKILL}, and scores have come in about 4.3 points "
                           "below the predictions")
    assert W.pred_unclear(skill_text="−0.03").startswith(
        "Score predictions are not yet clearly better than simply using the average of earlier "
        "scores (skill −0.03, where 0 matches that average and 1 is exact)")


def test_the_cause_of_the_uncertainty_and_what_to_check():
    clause = W.clause_wide(["Firmness", "Juiciness", "Cook loss"], cause=True, most="Cook loss")
    assert clause == ("; those for Firmness, Juiciness and Cook loss are still very uncertain, "
                      "because results vary more from one formulation to the next than the app "
                      "can explain, most for Cook loss")
    assert W.clause_wide(["Cook loss"], cause=True) == (
        "; those for Cook loss are still very uncertain, because results vary more from one "
        "formulation to the next than the app can explain")
    assert W.clause_wide(["Cook loss"]) == "; those for Cook loss are still very uncertain"
    assert W.CHECK_SCATTER == ("If the panel or method is new, check how consistently it "
                               "measures.")


def _h(inside=33, total=36, rounds=(3, 4, 5), wide=()):
    return insights.Headline("verdict", list(rounds), inside, total, "holding", wide=list(wide))


def _facts(row, clauses=(), h=None, **kw):
    return insights.StatusFacts(scored=kw.pop("scored", 21), prediction_row=row,
                                clauses=list(clauses), headline=h or _h(), **kw)


def _skill(value):
    return insights.SkillFact(value, 16, [2, 3, 4, 5], "no_skill" if value <= -0.2 else "unclear")


def test_line_c_predictions_sentence(monkeypatch):
    """Line C: the skill number, the bias, the wide outputs with their
    cause (every one widened by half or more) and the one it is largest
    for, then what to check."""
    opt = _patty(rows=12)
    wide = ["Firmness", "Juiciness", "Cook loss"]
    facts = _facts("no_skill", [("wide", wide), ("bias", -4.28)], skill=_skill(-0.469))
    facts.wide_factors = {"Firmness": 2.02, "Juiciness": 2.24, "Cook loss": 7.70}
    facts.best = types.SimpleNamespace(formulation=18)
    # Review diagnosis ruling 2 (2026-10-02): no Repeats measuring large
    # noise, so the sentence asks for a Repeat instead of naming the panel.
    assert ui_insights.prediction_sentence(opt, facts) == (
        f"Score predictions do no better than simply using the average of earlier scores {_SKILL}, "
        "and scores have come in about 4.3 points below the predictions; those for Firmness, "
        "Juiciness and Cook loss are still very uncertain, because results vary more from one "
        "formulation to the next than the app can explain, most for Cook loss. "
        + W.next_plain_repeat_ask(18, open_round=bool(opt.pending_batch)))


@pytest.mark.parametrize("factors,expect", [
    # One output under ×1.5: its width is not shown to come from scatter.
    ({"Firmness": 2.0, "Cook loss": 1.2}, None),
    # Two close: no "most".
    ({"Firmness": 2.0, "Cook loss": 2.4}, ""),
    ({"Firmness": 2.0, "Cook loss": 3.0}, "Cook loss"),
    # Not known (the models are still being fitted): no cause.
    ({}, None),
])
def test_the_cause_rule(factors, expect):
    assert ui_insights.wide_cause(["Firmness", "Cook loss"], factors) == expect


def test_reliable_with_a_wide_clause_ends_with_the_check():
    opt = _patty(rows=12)
    facts = _facts("reliable", [("wide", ["Cook loss"]), ("bias", 1.2)])
    facts.wide_factors = {"Cook loss": 3.0}
    facts.best = types.SimpleNamespace(formulation=15)
    text = ui_insights.prediction_sentence(opt, facts)
    # Review diagnosis ruling 2: the Repeat request, not the panel check,
    # without Repeats that vary a lot.
    assert text.endswith("; scores have come in about 1.2 points above the predictions. "
                         + W.next_plain_repeat_ask(15, open_round=bool(opt.pending_batch)))
    assert W.CHECK_SCATTER not in text
    assert "because results vary more" in text


# ------------------------------------------------------------------ #
#  Progress: ties within the noise threshold, "little more", the Upgrade
# ------------------------------------------------------------------ #

def test_progress_builders_name_a_tie_and_the_little_more_clause():
    assert W.progress_no_new_best(5, "90.45", 100, 15, 4, near_no=18, little=(
        W.binding_plain("highest", "Textured pea protein"),
        W.binding_plain("lowest", "Beet juice powder"))) == (
        "Round 5 did not beat Best so far, 90.45 of 100 (Formulation 15, Round 4; tied with "
        "Formulation 18). The app expects little more inside your current limits; the best it "
        "predicts sits at the most Textured pea protein and the least Beet juice powder allowed.")
    assert W.progress_no_new_best(5, "90.45", 100, 15, 4, little=()) == (
        "Round 5 did not beat Best so far, 90.45 of 100 (Formulation 15, Round 4). The app "
        "expects little more inside your current limits.")
    # Fix A follow-up (decision 4): the threshold the rise beat, not the SD.
    assert W.progress_rising(5, "3.5", "89.84", 100, 18, "3.2", near_no=20) == (
        "Round 5 raised Best so far by 3.5 points, to 89.84 of 100 (Formulation 18, tied with "
        "Formulation 20), more than the 3.2 points needed to count as a rise.")
    assert W.progress_flat(7, "92.50", 100, 24, 5, luck_text="90", near_no=3) == (
        "No higher score in 7 Rounds. Best so far is 92.50 of 100 (Formulation 24, Round 5; tied "
        "with Formulation 3), probably helped by luck: a Repeat of it is predicted near 90.")
    assert W.progress_flat(2, "90.09", 100, 11, 3, near_no=10, little=(
        W.binding_plain("lowest", "Mixing time after fat", values=True),)) == (
        "No higher score in 2 Rounds. Best so far is 90.09 of 100 (Formulation 11, Round 3; tied "
        "with Formulation 10). The app expects little more inside your current limits; the best "
        "it predicts sits at the lowest Mixing time after fat allowed.")
    assert W.binding_plain("limit", "Fat", "Fat: at most 20 g") == "your limit Fat: at most 20 g"


def test_the_upgrade_checks_too_early_gives_its_reason():
    assert W.progress_too_early("90.50", 100, 6, 3, without=([1, 2], "Cook loss")) == (
        "Best so far is 90.50 of 100 (Formulation 6, Round 3). Rounds 1–2 were scored without "
        "Cook loss, so they are not compared with it.")
    assert W.progress_too_early("90.50", 100, 6, 3, without=([2], "Cook loss")) == (
        "Best so far is 90.50 of 100 (Formulation 6, Round 3). Round 2 was scored without "
        "Cook loss, so it is not compared with it.")
    # With truly no earlier scored formulation, today's wording.
    assert W.progress_too_early("90.50", 100, 6, 3) == (
        "Best so far is 90.50 of 100 (Formulation 6, Round 3); no earlier scored formulation to "
        "compare it with.")


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


def test_little_more_needs_both_numbers_at_or_under_the_threshold():
    s = types.SimpleNamespace
    standing = s(higher=False, binding=[insights.Binding("highest", "Pea", "Pea")])
    assert ui_insights.little_more(standing, 0.12, _yard(), withheld=None) == (
        W.binding_plain("highest", "Pea"),)
    assert ui_insights.little_more(standing, 1.08, _yard(), withheld=None) is None
    assert ui_insights.little_more(s(higher=True, binding=[]), 0.1, _yard(), None) is None
    assert ui_insights.little_more(standing, 0.12, _yard(), withheld="skill") is None
    assert ui_insights.little_more(standing, None, _yard(), withheld=None) is None
    assert ui_insights.little_more(None, 0.12, _yard(), withheld=None) is None
    assert ui_insights.little_more(s(higher=False, binding=[]), 0.1, _yard(), None) == ()


def test_little_more_is_said_only_for_no_new_best_and_flat():
    opt = _patty(rows=12)
    f = _best(opt)
    facts = _facts("reliable", progress="rising", best=f,
                   improvement=insights.Improvement("yes", 2.5, None, 0))
    facts.little = ()
    assert "little more" not in ui_insights.progress_sentence(opt, facts)
    f = _best(opt, rounds_since=1, latest=False)
    facts = _facts("reliable", progress="no_new_best", best=f)
    facts.little = ()
    assert ui_insights.progress_sentence(opt, facts).endswith(
        "The app expects little more inside your current limits.")


# ------------------------------------------------------------------ #
#  Next: Line B leads with the check (walkthrough, ruled)
# ------------------------------------------------------------------ #

def test_line_b_next_leads_with_the_check():
    text = W.next_plain_repeat(21, 11, "10.5", half_text="7.4", round_no=6)
    assert text == (
        "Check what changed before you make Round 6. Formulation 21, a Repeat of Formulation 11, "
        "scored 10.5 points lower, more than two makes of one formulation should differ (up to "
        "7.4 points). Look at the panel, suppliers and equipment.")
    assert "since Round" not in text
    assert W.next_plain_repeat(21, 11, "4.0", lower=False, values=True, round_no=7).startswith(
        "Check what changed before you run Round 7. ")


def test_next_sentence_names_the_open_round_or_the_next_one():
    opt = _patty(rows=12)
    d = types.SimpleNamespace(later=21, earlier=11, diff=-10.5, limit=7.4)
    facts = _facts("off_repeat", next=insights.NextStep("repeat", disagreement=d))
    text = ui_insights.next_sentence(opt, facts)
    assert text.startswith(f"Check what changed before you make Round {opt.next_batch_no()}. ")
    assert "(up to 7.4 points)" in text


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
    texts = [_method_text(), *W.MODEL_LIMITS, W.HOW_RANGE, W.predicted_score_help(20, "1.6"),
             W.HOW_WIDENED, W.HOW_WIDENED_SCORE, W.how_skill(), W.TIP_RELIABLE_LIMITS,
             W.EXPECTED_IMPROVEMENT_HELP, W.HOW_EXPECTED_IMPROVEMENT, W.BASIS_PREDICTIONS,
             W.BASIS_PROGRESS, W.HOW_NOISE_THRESHOLD, W.TIP_NOISE_THRESHOLD,
             W.TIP_THRESHOLD_ASSUMED, W.gain_line("0.12", 6, W.beat_by())]
    for rows in (W.GLOSSARY_ROWS, W.glossary_rows(True), W.status_rule_rows()):
        texts += [" ".join(r) for r in rows]
    return texts


def test_i1_no_single_incumbent_under_qlognei():
    for t in _everything():
        assert "best tested" not in t, t
    method = _method_text()
    assert "no single incumbent" in method
    assert "approximate" in method
    best = next(r for r in W.GLOSSARY_ROWS if r[0] == "Best so far")
    assert "no single incumbent" in best[2]
    ei = next(r for r in W.GLOSSARY_ROWS if r[0] == "expected gain")   # charts proto: one name
    assert "no single incumbent" in ei[2] and "approximate" in ei[2]
    assert "approximate" in W.EXPECTED_IMPROVEMENT_HELP


def test_i2_the_coverage_aim_and_its_measured_rate_from_one_constant(monkeypatch):
    for t in _everything():
        assert "19 in 20" not in t and "19 times in 20" not in t, t
    reading = W.coverage_reading()
    # Ftue fix A's re-run (I9 folds): 90 % of Overall scores, was 89 %.
    assert reading == ("Aims for 95 %; measured about 91 % (measurements) / 90 % (Overall "
                       "score) on simulated projects.")
    assert reading in W.HOW_RANGE
    assert W.predicted_score_help(20).endswith(reading)
    monkeypatch.setattr(W, "SIMULATED_COVERAGE", {"measurements": 92, "score": 88})
    assert "about 92 % (measurements) / 88 % (Overall score)" in W.coverage_reading()


def test_i3_the_second_widening_widens_and_does_not_debias():
    assert "corrects for" not in W.HOW_WIDENED_SCORE
    assert "does not correct the predicted value" in W.HOW_WIDENED_SCORE


def test_i4_the_skill_error_rates_are_exact_and_show_n():
    for t in (W.MODEL_LIMITS[1], W.TIP_RELIABLE_LIMITS):
        assert "24 simulated projects" in t, t
        assert "n = 10" in t and "n = 15" in t, t
        assert "about half the time" in t, t
    assert "24 simulated projects" in W.how_skill()
    assert "a correct model reads" not in W.how_skill()


def test_i8_the_first_matching_row_wins():
    assert W.STATUS_TABLE_ORDER == ("Within each line the rows are tried from top to bottom; "
                                    "the first that holds is the status shown.")


def test_i9_the_folds_are_grouped_by_round():
    for t in (_method_text(), W.BASIS_PREDICTIONS, W.HOW_WIDENED,
              next(r for r in W.GLOSSARY_ROWS if r[0] == "calibration factor")[2]):
        assert "grouped by Round" in t, t


def test_i10_the_coverage_count_assumes_independent_misses():
    assert any("not independent" in t for t in W.MODEL_LIMITS)
    off = next(r for r in W.status_rule_rows() if r[1].endswith("(coverage)"))
    assert "independent" in off[2]


def test_i11_the_threshold_is_gated_on_degrees_of_freedom():
    for t in (W.HOW_NOISE_THRESHOLD, W.BASIS_PROGRESS, W.TIP_NOISE_THRESHOLD,
              W.TIP_THRESHOLD_ASSUMED,
              next(r for r in W.GLOSSARY_ROWS if r[0] == "noise threshold")[2]):
        assert "3 pairs" not in t and "3 Repeat pairs" not in t, t
        assert "3 degrees of freedom" in t or "3 df" in t, t
    assert W.stand_repeats("0.8", 1, df=1, groups=1) == (
        "Repeat SD of the Overall score: 0.8 points, 1 df (2 results of 1 formulation). The "
        "noise threshold stays at 1 point until 3 df.")
    assert W.stand_repeats("0.7", 4, df=4, groups=4) == (
        "Repeat SD of the Overall score: 0.7 points, 4 df (8 results of 4 formulations); this "
        "is the noise threshold.")
    assert W.tip_threshold_measured("1.2", 4, 3) == "pooled Repeat SD 1.2 points, 3 df"
    assert W.beat_by(None, assumed=True, pairs=1, df=1) == (
        "the 1-point noise threshold (assumed: 1 df of Repeats, 3 needed; a Repeat of Best so "
        "far would measure it)")


def test_i12_changed_settings_are_not_a_fair_test():
    assert ("Changing model settings re-scores every earlier Round, so a record read after "
            "choosing settings to suit it is no longer a fair test.") in _method_text()


def test_every_new_text_is_clean():
    cells = [_method_facts_lines(), W.MODEL_LIMITS, W.status_rule_rows(), W.GLOSSARY_ROWS,
             W.glossary_rows(True)]
    for group in cells:
        for item in group:
            for t in (item if isinstance(item, tuple) else (item,)):
                _clean(t)
    for t in (W.HOW_RANGE, W.predicted_score_help(20, "1.6"), W.HOW_WIDENED,
              W.HOW_WIDENED_SCORE, W.how_skill(), W.TIP_RELIABLE_LIMITS,
              W.EXPECTED_IMPROVEMENT_HELP, W.HOW_EXPECTED_IMPROVEMENT, W.BASIS_PREDICTIONS,
              W.BASIS_PROGRESS, W.HOW_NOISE_THRESHOLD, W.TIP_NOISE_THRESHOLD,
              W.TIP_THRESHOLD_ASSUMED, W.gain_line("0.12", 6, W.beat_by()), W.STATUS_TABLE_ORDER,
              W.coverage_reading()):
        _clean(t)
    _clean(W.pred_no_skill(("4.3", True), [W.clause_wide(["Cook loss"], cause=True)],
                           skill_text="−0.47"))
    _clean(W.CHECK_SCATTER)
    _clean(W.next_plain_repeat(21, 11, "10.5", half_text="7.4", round_no=6))
