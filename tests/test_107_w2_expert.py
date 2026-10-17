"""Insights for a first-time user, W2: the expert layer (spec 2026-10-01
rev. 2 §3, §7; review R-B2, R-I6, R-I12, R-I13, M4, M5, M8, M9, M12).

- "Model and method" is read off a model actually built for the project
  (insights.model_summary), never from settings alone: changing the
  kernel, the noise setting, the acquisition or q changes the text, and a
  library default that moves is caught because the text follows the object.
- "How each status is decided" has one row per state and per Next row.
- Every status line's tooltip states its exact rule with the project's own
  numbers; every term tooltip reads plain first, then the BO term.
- Method and definitions draws the summary, the known limits, the rule
  table and the glossary; the glossary is never said through for_project.
- Make a round's Predicted Overall score column, track line and Why fold
  name their statistics."""
import math

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import torch
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_insights
import wording as W
from food_bo import COLD_START_RUNS, validate_bo_config
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _text(opt):
    """The summary as Method and definitions says it."""
    return " ".join(W.model_summary(insights.model_summary(opt)))


@pytest.fixture(scope="module")
def project():
    """name -> (project, its Round-ahead checks), fitted once."""
    seen = {}

    def get(name):
        if name not in seen:
            opt = D.load(name)
            checks = insights.check_rounds(opt) if opt.complete_count() >= insights.MIN_ROWS else []
            seen[name] = (opt, checks)
        return seen[name]
    return get


def _status(opt, checks):
    """insights.StatusFacts as W1 hands it to tooltip_for: W1's own
    status_facts, the one status logic (integration: no copy here)."""
    return insights.status_facts(opt, checks)


def _rule(opt, key):
    """The exact rule of a RULE_KEYS row, as the table says it."""
    return insights.status_rule(opt, key)


# ------------------------------------------------------------------ #
#  Model and method: read from the built model (§7, R-I12, R-I13)
# ------------------------------------------------------------------ #

def test_recommended_settings_are_read_off_the_built_model(project):
    """Line A (bo_config None): BoTorch's own default, read from the object
    _build_gp makes: RBF with ARD, LogNormal(√2 + ½ ln d, √3) lengthscale
    prior and its 0.025 floor, no outputscale, a Gaussian likelihood with
    a LogNormal(−4, 1) prior and its 1e-4 floor, Standardize(m=1)."""
    opt, _ = project(D.LINE_A)
    f = insights.model_summary(opt)
    d = len(insights.xy(opt, insights.SCORE, [0])[0][0])
    assert (f.cold, f.scored, f.recommended, f.inputs) == (False, 20, True, d)
    assert (f.kernel, f.ard, f.outputscale) == ("rbf", True, False)
    loc = math.sqrt(2) + 0.5 * math.log(d)
    assert f.lengthscale_prior == f"LogNormal({loc:.2f}, {math.sqrt(3):.2f})"
    assert f.lengthscale_floor == pytest.approx(0.025)
    assert (f.noise, f.noise_prior) == ("fitted", "LogNormal(−4.00, 1.00)")
    assert f.noise_floor == pytest.approx(1e-4)
    assert f.standardised
    assert (f.acquisition, f.beta, f.mc_samples) == ("qlognei", None, 512)
    assert (f.restarts, f.raw_samples) == (bo_views.ASK_RESTARTS, bo_views.ASK_RAW_SAMPLES)
    assert (f.folds, f.fit_tolerance, f.cold_start) == (insights.FOLDS, 1e-6, COLD_START_RUNS)
    assert f.q is None and not f.robust and not f.values and f.total
    text = _text(opt)
    assert text.startswith(W.MODEL_SUMMARY_LEAD)
    assert (f"Kernel (Recommended settings): squared exponential (RBF) with one lengthscale per "
            f"input (ARD), lengthscale prior LogNormal({loc:.2f}, 1.73), at least 0.025; no "
            "outputscale.") in text
    assert "Likelihood: Gaussian, noise fitted, prior LogNormal(−4.00, 1.00), at least 0.0001." in text
    assert "q = the number of formulations generated" in text
    assert "sequential greedy" in text and "maximum a posteriori" in text


def test_the_text_follows_the_object_not_a_hard_coded_default(project, monkeypatch):
    """R-I13: were BoTorch's default to change (it did in 0.12), the summary
    would follow: here the builder hands back a Matérn 5/2 for Recommended
    settings, and the text says so."""
    opt, _ = project(D.LINE_A)
    real = opt._build_gp

    def matern(X, Y, dim, input_tf):
        from botorch.models import SingleTaskGP
        from gpytorch.kernels import MaternKernel
        from botorch.models.transforms import Standardize
        return SingleTaskGP(X, Y, covar_module=MaternKernel(nu=2.5, ard_num_dims=dim),
                            outcome_transform=Standardize(m=1))

    monkeypatch.setattr(opt, "_build_gp", matern)
    text = _text(opt)
    assert "Kernel (Recommended settings): Matérn 5/2 with one lengthscale per input (ARD)" in text
    assert "squared exponential" not in text and "lengthscale prior" not in text
    monkeypatch.setattr(opt, "_build_gp", real)
    assert "squared exponential (RBF)" in _text(opt)


@pytest.mark.parametrize("cfg,needles", [
    ({"kernel": "matern52"},
     ["Kernel (your model settings): Matérn 5/2 with one lengthscale per input (ARD), "
      "lengthscale prior Gamma(3.00, 6.00); outputscale fitted.", "qLogNEI"]),
    ({"kernel": "matern32", "lengthscale_prior": "long"},
     ["Matérn 3/2 with one lengthscale per input (ARD), lengthscale prior Gamma(3.00, 1.00)"]),
    ({"kernel": "rbf", "noise": "low", "acquisition": "qucb"},
     ["Kernel (your model settings): squared exponential (RBF) with one lengthscale per input",
      "noise fitted, prior Gamma(1.10, 50.00), at least 0.0001",
      "qUCB (upper confidence bound, β = 0.2)"]),
    ({"kernel": "linear", "noise": "fixed_tiny", "acquisition": "qlogei"},
     ["Kernel (your model settings): linear; outputscale fitted.",
      "Likelihood: Gaussian, noise fixed near zero by your model settings.",
      "qLogEI (expected improvement over the best measured Overall score"]),
    ({"kernel": "poly2"}, ["Kernel (your model settings): polynomial of degree 2; outputscale"]),
])
def test_each_expert_setting_changes_the_summary(project, cfg, needles):
    opt, _ = project(D.LINE_A)
    before = _text(opt)
    saved = opt.bo_config
    opt.bo_config = validate_bo_config(cfg)
    try:
        text = _text(opt)
    finally:
        opt.bo_config = saved
    assert text != before
    for needle in needles:
        assert needle in text, (needle, text)
    if cfg["kernel"] in ("linear", "poly2"):
        assert "ARD" not in text and "lengthscale" not in text


def test_q_comes_from_the_generation_record(project):
    """The open Round's record (Oven trial: Round 6, 4 picks), else the last
    Round's; a different record changes the text."""
    opt, _ = project(D.PROCESS_ONLY)
    records = opt.generation_records()
    assert insights.model_summary(opt).q == records[opt.pending_batch_no]["n"] == 4
    assert "q = 4 per Round" in _text(opt)
    saved = dict(records[opt.pending_batch_no])
    records[opt.pending_batch_no]["n"] = 6
    try:
        assert "q = 6 per Round" in _text(opt)
    finally:
        records[opt.pending_batch_no] = saved
    # Without an open Round's record, the last Round's.
    pending, opt.pending_batch_no = opt.pending_batch_no, None
    try:
        assert insights.model_summary(opt).q == records[max(records)]["n"]
    finally:
        opt.pending_batch_no = pending


def test_robust_and_the_process_settings_form(project):
    opt, _ = project(D.PROCESS_ONLY)
    f = insights.model_summary(opt)
    assert f.values and f.inputs == 3
    text = _text(opt)
    assert "Inputs: each process setting normalised to [0, 1]" in text
    assert "Input warping" not in text
    opt.robust = True
    try:
        assert insights.model_summary(opt).robust
        assert "Input warping: on for the suggestion model (robust)" in _text(opt)
    finally:
        opt.robust = False


def test_the_gate_shows_only_the_cold_start(project):
    """M9: the Snack bar (3 scored) builds no model and says two lines."""
    opt, _ = project(D.SNACK)
    f = insights.model_summary(opt)
    assert f.cold and f.scored == 3
    lines = W.model_summary(f)
    assert len(lines) == 2 and lines[0].startswith("Cold start: the first 5 formulations")


def test_reading_the_model_leaves_the_global_rng_alone(project):
    """The summary builds a GP and its acquisition (qLogNEI prunes its
    baseline by sampling) inside insights._isolated(): nothing about how
    the next suggestion is made can move."""
    opt, _ = project(D.LINE_A)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    insights.model_summary(opt)
    assert torch.equal(before, torch.get_rng_state())


# ------------------------------------------------------------------ #
#  How each status is decided (I13, R-I13, M9)
# ------------------------------------------------------------------ #

def test_the_rule_keys_line_up_with_the_table():
    rows = W.status_rule_rows()
    assert len(insights.RULE_KEYS) == len(rows)
    for (line, key), row in zip(insights.RULE_KEYS, rows):
        word = {"predictions": W.STATUS_PREDICTIONS, "clause": W.STATUS_PREDICTIONS,
                "progress": W.STATUS_PROGRESS, "next": W.STATUS_NEXT}[line]
        assert row[0] == word, (key, row)
        if line == "predictions":
            assert row[1].startswith(W.PREDICTIONS_WORDS[insights.PREDICTION_STATUS[key]])
        if line == "progress":
            assert row[1] == W.PROGRESS_WORDS[key]
        if line == "clause":
            assert row[1].startswith("Clause")
    assert [k for line, k in insights.RULE_KEYS if line == "next"] == list(insights.NEXT_ROWS)


def test_status_rules_in_the_projects_own_form(project):
    opt, checks = project(D.LINE_A)
    assert insights.status_rules(opt, checks) == W.status_rule_rows("−0.20")
    oven, checks = project(D.PROCESS_ONLY)
    rows = insights.status_rules(oven, checks)
    assert rows == W.status_rule_rows("−0.20", settings_only=True)
    assert any("process setting" in r[1] for r in rows)


def test_at_the_gate_only_what_applies(project):
    """M9: the gate row and the Progress rows; no Next row is tried yet."""
    opt, _ = project(D.SNACK)
    rows = insights.status_rules(opt)
    assert [r[0] for r in rows] == [W.STATUS_PREDICTIONS] + [W.STATUS_PROGRESS] * 5
    assert rows[0][1] == "Not yet (gate)"


# ------------------------------------------------------------------ #
#  The status tooltips: exact rule and numbers (§7, R-B1, I13)
# ------------------------------------------------------------------ #

LINE_A_TIP = (
    # 33 of 35 since ftue fix A's Round-grouped folds (cold read I9); 32 before.
    "Measurements (Firmness, Juiciness, Cook loss), Rounds 3–5: 33 of 35 Round-ahead values "
    "inside their 95 % posterior predictive intervals (5-fold cross-validation widening); a "
    "binomial count at 5 % per value allows up to 3 misses in 35 at the 1-in-10 level. "
    "Overall score, Rounds 2–5: out-of-sample skill 1 − SSE/SSE(mean of the rows before "
    "each Round) = 0.36 over 15 formulations, at least 0.20 over at least 10 needed. Each "
    "Round is predicted by models refitted today on earlier Rounds only.")


def test_line_a_reads_the_reviews_exact_rule(project):
    opt, checks = project(D.LINE_A)
    facts = _status(opt, checks)
    assert facts.prediction_row == "reliable"
    tip = ui_insights.tooltip_for(opt, "predictions", facts)
    assert tip.startswith(LINE_A_TIP)
    assert W.TIP_RELIABLE_LIMITS in tip
    assert _rule(opt, ("predictions", "reliable")) in tip


@pytest.mark.parametrize("name", D.ALL)
def test_every_status_tooltip_names_its_rule_and_its_numbers(project, name):
    opt, checks = project(name)
    facts = _status(opt, checks)
    say = lambda text: W.for_project(opt, text)
    pred = ui_insights.tooltip_for(opt, "predictions", facts)
    assert say(_rule(opt, ("predictions", facts.prediction_row))) in pred
    assert any(ch.isdigit() for ch in pred)
    if facts.headline is not None and facts.headline.state == "verdict":
        h = facts.headline
        assert f"{h.inside} of {h.total} Round-ahead values" in pred
    if facts.skill is not None:
        assert ui_insights.skill_value(facts.skill.value) in pred
    if facts.disagreement is not None and facts.prediction_row == "off_repeat":
        assert f"{abs(facts.disagreement.diff):.1f}" in pred
    prog = ui_insights.tooltip_for(opt, "progress", facts)
    assert say(_rule(opt, ("progress", facts.progress))) in prog
    yard = facts.yardstick
    assert (W.TIP_THRESHOLD_ASSUMED in prog if not yard.from_repeats
            else f"pooled Repeat SD {yard.value:.1f} points" in prog)
    nxt = ui_insights.tooltip_for(opt, "next", facts)
    assert nxt.startswith("Next-step rule that holds: ")
    if facts.next is not None:
        assert say(_rule(opt, ("next", facts.next.kind))) in nxt
    else:
        assert say(_rule(opt, ("predictions", "gate"))) in nxt
    # The guard reads the source; on screen a process-settings study says
    # "trial" through for_project, its one sanctioned door.
    for tip in (pred, prog, nxt):
        if opt.has_ingredients():
            assert not any(p.search(tip) for p in _BANNED), tip
            assert "trial" not in tip.lower()


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_line_b_names_the_repeat_and_the_bias(project):
    """Line B: Off (coverage) with the Repeat in Next; its bias clause."""
    opt, checks = project(D.LINE_B)
    facts = _status(opt, checks)
    assert facts.prediction_row == "off_coverage"
    pred = ui_insights.tooltip_for(opt, "predictions", facts)
    # 28 of 36 since ftue fix A's Round-grouped folds (cold read I9); 26 before.
    assert "28 of 36 Round-ahead values" in pred and "up to 4 misses in 36" in pred
    # Ftue fix A (cold read I5): −5.5 is 2.45 standard errors, under
    # t_{0.975, 3} = 3.18, so no bias clause.
    assert facts.bias is None and "mean error −5.5 points" not in pred
    nxt = ui_insights.tooltip_for(opt, "next", facts)
    assert facts.next.kind == "repeat"
    # ±2.8 = √2 × 1.96 × the assumed 1 point (ftue fix A, cold read I7); ±7.4 before.
    # Fix A follow-up (decision 2): under 3 df the pair must beat both that
    # and √2 × its own Round-ahead half-width, so the limit is the larger, ±8.2.
    assert ("Formulation 21, a Repeat of Formulation 11, differs by 10.5 points: more than "
            "±8.2") in nxt


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_process_only_tooltips_say_trial(project):
    opt, checks = project(D.PROCESS_ONLY)
    facts = _status(opt, checks)
    assert facts.prediction_row == "unclear"
    pred = ui_insights.tooltip_for(opt, "predictions", facts)
    assert "over 16 trials" in pred and "formulation" not in pred.lower()


def test_status_tooltips_need_the_facts():
    """W1 always hands them in; without them there is nothing exact to say."""
    opt = D.load(D.LINE_A)
    for line in ("predictions", "progress", "next"):
        assert ui_insights.tooltip_for(opt, line) is None
    assert ui_insights.tooltip_for(opt, "no such line") is None


# ------------------------------------------------------------------ #
#  The term tooltips (§3, R-I6)
# ------------------------------------------------------------------ #

_TERM_LINES = [line for line in ui_insights.TIP_LINES
               if line not in ("predictions", "progress", "next", "basis_next",
                               "next_reasoning", "luck")]


@pytest.mark.parametrize("line", _TERM_LINES)
def test_each_term_line_reads_plain_first_then_the_bo_term(line):
    """One "?" per line (R-I6): the plain reading first, then the standard
    term (the strips' tip too, since T0′ at integration)."""
    opt = D.load(D.LINE_A)
    tip = ui_insights.tooltip_for(opt, line)
    assert tip, line
    head, sep, _ = tip.partition(" BO term: ")
    assert sep and head, (line, tip)
    assert not any(p.search(tip) for p in _BANNED), tip


def test_the_luck_tip_points_to_method(project):
    opt = D.load(D.LINE_A)
    assert ui_insights.tooltip_for(opt, "luck") == W.TIP_LUCK


@pytest.mark.parametrize("name", [D.LINE_A, D.LINE_B, D.LINE_C, D.TIMING])
def test_the_next_reasoning_and_basis_lines_name_the_row(project, name):
    opt, checks = project(name)
    facts = _status(opt, checks)
    reasoning = ui_insights.tooltip_for(opt, "next_reasoning", facts)
    assert " BO term: " in reasoning
    assert ui_insights.tooltip_for(opt, "basis_next", facts) == ui_insights.tooltip_for(
        opt, "next", facts)


def test_the_figures_lines_carry_their_numbers(project):
    opt, checks = project(D.LINE_A)
    facts = _status(opt, checks)
    verdict = ui_insights.tooltip_for(opt, "verdict", facts)
    assert verdict.startswith(W.TIP_INTERVAL) and W.TIP_VERDICT in verdict
    assert W.TIP_NOT_JUDGED not in verdict and "33 of 35 Round-ahead values" in verdict
    assert "= 0.36 over 15" in ui_insights.tooltip_for(opt, "skill", facts)
    early = ui_insights.tooltip_for(opt, "verdict")
    assert W.TIP_NOT_JUDGED in early


def test_the_process_only_term_tips_say_values():
    opt = D.load(D.PROCESS_ONLY)
    assert "these values" in ui_insights.tooltip_for(opt, "map")
    assert "Untested values" in ui_insights.tooltip_for(opt, "standing")
    assert "trial" in ui_insights.tooltip_for(opt, "standing").lower()


# ------------------------------------------------------------------ #
#  The basis lines (§7): drawn with their help at the top of each fold
# ------------------------------------------------------------------ #

def test_each_basis_line_is_its_text_and_its_help():
    opt = D.load(D.LINE_A)
    # Concision round: the short basis on screen, the full one first in its help.
    for which, text in (("predictions", W.BASIS_PREDICTIONS), ("progress", W.BASIS_PROGRESS)):
        line, help_text = ui_insights.basis_line(opt, which)
        assert line == W.BASIS_SHORT[which]
        assert help_text == f"{text}\n\n{ui_insights.tooltip_for(opt, f'basis_{which}')}"
    line, help_text = ui_insights.basis_line(opt, "next")
    assert line == W.BASIS_SHORT["next"] and help_text == W.BASIS_NEXT
    oven = D.load(D.PROCESS_ONLY)
    _, help_text = ui_insights.basis_line(oven, "predictions")
    assert "trials recorded before each Round" in help_text


# ------------------------------------------------------------------ #
#  Method and definitions on screen (§3, §7, R-B2, M4, M8, M12)
# ------------------------------------------------------------------ #

@pytest.fixture
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    with bo_views.fresh():
        yield


def _install(name):
    import json
    from food_bo import FoodOptimizer
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    return opt


def _method(name):
    opt = _install(name)
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = W.TAB_INSIGHTS
    at.session_state[ui_insights.HOW_KEY] = True
    at.run()
    assert not at.exception
    fold = next(e for e in at.tabs[3].expander if e.label == W.for_project(opt, W.FOLD_METHOD))
    return opt, fold


def _cells(table):
    frame = table.value
    return [str(v) for v in frame.index] + [str(v) for col in frame.columns
                                            for v in frame[col]]


@pytest.mark.usefixtures("_isolate")
def test_method_and_definitions_at_the_gate():
    opt, fold = _method(D.SNACK)
    text = "\n".join(m.value for m in fold.markdown)
    assert W.MODEL_SUMMARY_LEAD not in text
    for line in W.model_summary(insights.model_summary(opt)):
        assert line in text
    rules, _ = list(fold.table)
    assert len(rules.value) == 6
