"""0.8.0 BO views, Task 6 (spec 2026-09-30-bo-views §2, §3.2, §4, §8;
review I1-I7, M6, M8, M11): Insights says its statistics by name with
their basis — the verdict's miss count, the Overall score's skill, the
Repeat-based noise threshold — under noun headings; says expected
improvement by Round in Progress; and stops stating noise as ingredient
effects when its model is fitting noise."""
import re
import json
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as demo
from tests.test_app_ui import APP_PATH


def _install(name):
    """A demo project saved in the working folder, as the app opens it."""
    state = json.loads((demo.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    return opt


def _open(name, fold=None):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    if fold is not None:
        at.session_state[fold] = True
    at.run()
    assert not at.exception
    return at


def _texts(at):
    tab = at.tabs[3]
    return [m.value for m in tab.markdown], [c.value for c in tab.caption]


def _all(node):
    """Every markdown and caption under `node`, folds included."""
    return [m.value for m in node.markdown] + [c.value for c in node.caption]


@pytest.fixture(scope="module")
def tabs(tmp_path_factory):
    """Lines A, B and C, each opened once on Insights (the checks take a
    few seconds a project); Line C also with the first fold open."""
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("demo"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (demo.LINE_A, demo.LINE_B, demo.LINE_C):
                _install(name)
                out[name] = _open(name)
            out["C try one"] = _open(demo.LINE_C, "insights_try_one")
            out["A try one"] = _open(demo.LINE_A, "insights_try_one")
            out["A how"] = _open(demo.LINE_A, ui_insights.HOW_KEY)
            yield out
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  Headings and the verdict (03 §4, §5.2; review I7)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_three_sections_are_real_noun_headings(tabs):
    md, _ = _texts(tabs[demo.LINE_A])
    heads = [m for m in md if m.startswith("##### ")]
    assert heads[:3] == [f"##### {q}" for q in (wording.Q_TRUST, wording.Q_STAND,
                                                wording.Q_NEXT)]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_verdict_says_its_miss_count(tabs):
    opt = demo.load(demo.LINE_A)
    checks = insights.check_rounds(opt)
    h = insights.headline(opt, checks)
    md, _ = _texts(tabs[demo.LINE_A])
    verdict = wording.trust_within(h.inside, h.total, h.rounds,
                                   allowed=insights.allowed_misses(h.total))
    assert any(m.startswith(verdict) for m in md)
    assert not any("though" in m or "likely range" in m for m in md)


# ------------------------------------------------------------------ #
#  Skill (review I1, I2)
# ------------------------------------------------------------------ #

# The scale's two ends ride with the number (ftue spec §4, review R-I7,
# M11; T0).
@pytest.mark.parametrize("name,text", [
    (demo.LINE_A, "Overall score, Rounds 2–5: skill 0.36 (n = 15; 0 = as good as the average, "
                  "1 = perfect), better than the average of earlier results."),
    (demo.LINE_C, "Overall score, Rounds 2–5: skill −0.47 (n = 16; 0 = as good as the "
                  "average, 1 = perfect), no better than the average of earlier results."),
])
@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_skill_line_sits_under_the_verdict(tabs, name, text):
    md, _ = _texts(tabs[name])
    assert text in md
    assert md.index(text) < md.index(f"##### {wording.Q_STAND}")


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_skill_is_the_next_step_on_line_c(tabs):
    md, _ = _texts(tabs[demo.LINE_C])
    opt = demo.load(demo.LINE_C)
    assert wording.next_no_skill("−0.47", 16, opt.pending_batch_no, 4,
                                 rounds=[2, 3, 4, 5]) in md


# ------------------------------------------------------------------ #
#  §3.2 Expected gain by Round (review I4, I5, M8)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_line_a_says_the_expected_improvement_of_its_open_round(tabs):
    md, _ = _texts(tabs[demo.LINE_A])
    [line] = [m for m in md if m.startswith("Expected gain")]
    assert line.startswith("Expected gain, Round 6: 0.12 points (model, one "
                           "formulation): ")
    assert "; under the 1-point noise threshold." in line
    assert line.endswith("Recomputed from today's records.")
    assert md.index(line) < md.index(f"##### {wording.Q_NEXT}")


@pytest.mark.parametrize("name,reason", [(demo.LINE_B, wording.GAIN_REASON_VERDICT),
                                         (demo.LINE_C, wording.GAIN_REASON_SKILL)])
@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_number_is_withheld_when_the_model_is_not_trusted(tabs, name, reason):
    md, _ = _texts(tabs[name])
    assert wording.gain_unreliable(reason) in md


@pytest.fixture(scope="module")
def timing_generated(tmp_path_factory):
    """The Timing check straight after Generate (ui_batch.ask_alone, which
    records Round 13's generation), opened on Insights."""
    import ui_batch
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("timing"))
    try:
        with bo_views.fresh():
            opt = _install(demo.TIMING)
            ui_batch.ask_alone(opt, n_suggestions=3)
            got = _open(demo.TIMING), opt
    finally:
        os.chdir(here)       # the other tests open their projects where they were
    return got


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_after_generate_progress_and_next_step_say_one_thing(timing_generated):
    """Fix round 1 (T56 C1, ledger ruling): nothing predicted beyond the
    noise threshold, an expected gain above it. Progress shows both,
    with one sentence on where the second comes from; Next step is not
    "close to the top" but the explore row."""
    at, opt = timing_generated
    md, _ = _texts(at)
    # 0.8.0 e2e fixes (review B2): no Repeats yet, so the 1 point is said to
    # be assumed, once: here, since line 1 does not name the threshold.
    assert any("Predicted gain over Best so far: none beyond "
               + wording.beat_by(None, assumed=True) + "." in m for m in md)
    [line] = [m for m in md if m.startswith("Expected gain")]
    assert line.startswith(f"Expected gain, Round {opt.pending_batch_no}: ")
    assert line.endswith("above the 1-point noise threshold. It comes from the model's "
                         "uncertainty, not from a higher predicted score.")
    after = md[md.index(f"##### {wording.Q_NEXT}") + 1]
    assert after == (wording.next_explore(opt.pending_batch_no, 3) + " "
                     + wording.NEXT_REPEAT_TO_MEASURE)
    assert not any(m.startswith("Within these") for m in md)


@pytest.fixture(scope="module")
def timing_recorded_flat(tmp_path_factory):
    """The final review's p_stale: the Timing check, Generate 3, every pick
    recorded at the median formulation's results, so no Round is open."""
    import ui_batch
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("timing_flat"))
    try:
        with bo_views.fresh():
            opt = _install(demo.TIMING)
            ui_batch.ask_alone(opt, n_suggestions=3)
            r = opt.pending_batch_no
            order = sorted(range(len(opt.Y_history)), key=lambda i: opt.Y_history[i])
            results = dict(opt.results_history[order[len(order) // 2]])
            for row in list(opt.pending_batch):
                opt.tell(row["recipe"], results, formulation_no=row["formulation"], batch_no=r)
            opt.set_pending_batch(None)
            opt.save()
            got = _open(demo.TIMING), FoodOptimizer(opt.project_name), r
            # The flat Round brings skill to 0.16: the A3 gate (0.8.0 e2e
            # fixes, review I4) withholds the figure. Also opened with a model
            # whose skill lets it through, for final review I-1's check.
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(insights, "skill_gate", lambda o, c, fact=None: None)
                got = got + (_open(demo.TIMING),)
    finally:
        os.chdir(here)
    return got


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_round_recorded_flat_withholds_the_figure_under_the_a3_gate(timing_recorded_flat):
    """0.8.0 e2e fixes (review I4): skill 0.16 is not clearly better than the
    average, so the expected gain is withheld, saying why, and Next
    step does not offer "close to the top" from the model's predicted best."""
    at, opt, r, _ = timing_recorded_flat
    md, _ = _texts(at)
    fact = insights.skill(opt, ui_insights.checks_for(opt, "trust") or insights.check_rounds(opt))
    assert fact.state == "unclear"
    assert wording.gain_unreliable(wording.gain_reason_unclear(
        ui_insights.skill_value(fact.value), fact.n)) in md
    after = md[md.index(f"##### {wording.Q_NEXT}") + 1]
    assert not after.startswith("Within these")


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_round_recorded_flat_is_not_explored_again(timing_recorded_flat):
    """Final review I-1: with no Round open the expected gain is the
    next Round's, from today's rows (ask()'s own first-pick search on the
    model the next Generate fits), not Round 13's pre-results 1.08. It is
    under the noise threshold, so Next step is close to the top."""
    _, opt, r, at = timing_recorded_flat
    md, _ = _texts(at)
    [line] = [m for m in md if m.startswith("Expected gain")]
    assert line.startswith("Expected gain, next Round: ")
    assert "under the 1-point noise threshold" in line
    assert not any("still expects" in m or "model's uncertainty" in m for m in md)
    after = md[md.index(f"##### {wording.Q_NEXT}") + 1]
    assert after.startswith("Within these ingredients and limits, no predicted gain exceeds "
                            "the noise threshold.")
    # The figure is ask()'s own first-pick search on the model the next
    # Generate fits (botorch's default fit, the next seed: coordinator rulings).
    with bo_views.fresh():
        want = bo_views.next_optimum(opt, bo_views.next_model(opt))
    assert at.session_state[ui_insights.GAIN_KEY][1].gain == pytest.approx(want, abs=1e-9)
    assert line.startswith(f"Expected gain, next Round: {want:.2f} points")
    # Round 13's own figure stays as history, in Expected gain by Round.
    with insights.memo_in_use({}):
        history = bo_views.round_history(opt)
    assert history[-1].round == r and history[-1].gain > 1.0


def test_the_next_rounds_figure_is_worked_out_after_the_page(monkeypatch):
    """Final review I-1: the search (about 1.5 s at 60 x 4) is precomputed
    with the predicted best after the page is drawn, once per data, so no
    screen waits on it; never while a Round is open."""
    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui_insights, "memo", lambda o: {})
    calls = []
    real = bo_views.next_optimum
    monkeypatch.setattr(bo_views, "next_optimum", lambda *a: calls.append(1) or real(*a))
    opt = demo.load(demo.LINE_A)
    ui_insights.warm_gain(opt)                      # Round 6 open: nothing
    assert calls == [] and ui_insights.GAIN_KEY not in state
    opt.set_pending_batch(None)
    with bo_views.fresh():
        ui_insights.warm_gain(opt)
        ui_insights.warm_gain(opt)
    assert calls == [1] and state[ui_insights.GAIN_KEY][1].round is None


def test_a_machine_zero_reads_under_one_hundredth(monkeypatch):
    """T56 m3: the Timing check's 3.5e-21, not "0.00 points … beats … by
    about 0.00 points"."""
    opt = demo.load(demo.LINE_A)
    yard = insights.noise_yardstick(opt)
    # Line A has skill (0.36, n = 15); no checks are handed in here.
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)
    monkeypatch.setattr(ui_insights, "latest_gain", lambda *a, **k: bo_views.RoundGain(
        6, 90.45, None, 3.5e-21, True, None))
    got = ui_insights.gain_text(opt, {"checks": (), "fitted": {}}, yard)
    # Lines 1-2 above it say whether the threshold is assumed (review B2).
    assert got == wording.gain_line_tiny(6, ui_insights.threshold_text(yard, assumed=False))


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_layers_name_best_so_far_as_their_anchor():
    """T56 m7."""
    opt = demo.load(demo.LINE_A)
    best_no = int(opt.formulation_ids[opt.best_index()])
    assert ui_insights.anchor_text(opt, best_no) == wording.start_best(best_no)


def test_qucb_has_no_expected_improvement():
    opt = demo.load(demo.LINE_A)
    opt.bo_config = {"acquisition": "qucb"}
    yard = insights.noise_yardstick(opt)
    assert ui_insights.gain_text(opt, {"checks": (), "fitted": {}}, yard) == wording.GAIN_UCB


# ------------------------------------------------------------------ #
#  The noise guard (review I3, I4)
# ------------------------------------------------------------------ #

def _fold(at):
    return next(e for e in at.tabs[3].expander if e.key == "insights_try_one"
                or e.label.startswith("Try one"))


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_line_c_names_no_ingredient_effect(tabs):
    said = _all(_fold(tabs["C try one"]))
    assert wording.guard_line([wording.GUARD_FLOOR]) in said
    assert not any(t.startswith(("Largest local effect", "Local effect of")) for t in said)
    assert not [e for e in _fold(tabs["C try one"]).expander
                if e.label.startswith("Compare all")]


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_line_a_still_ranks_its_effects(tabs):
    said = _all(_fold(tabs["A try one"]))
    assert any(t.startswith(("Largest local effect", "Local effect of", "No ingredient"))
               for t in said)


def test_the_guard_says_setting_effects_for_a_process_only_study():
    guard = insights.Guard(["floor"])
    assert "setting effects" in ui_insights.guard_text(guard, settings_only=True)
    assert ui_insights.guard_text(insights.Guard(["skill"])) == wording.guard_skill_line()


# ------------------------------------------------------------------ #
#  Measured against predicted and the other lines with their basis
# ------------------------------------------------------------------ #

def test_method_and_details_is_drawn_only_while_open(tabs):
    """T56 I5: keyed, and worked out only while open."""
    fold = next(e for e in tabs[demo.LINE_A].tabs[3].expander
                if e.label == wording.FOLD_METHOD)
    assert not fold.markdown and not fold.caption and not fold.dataframe
    fold = next(e for e in tabs["A how"].tabs[3].expander if e.label == wording.FOLD_METHOD)
    # One-glance Review: one line per term, no table.
    assert [m.value for m in fold.markdown] == list(wording.METHOD_DEFINITIONS)
    assert not fold.dataframe


def test_measured_against_predicted_is_drawn_only_while_open(tabs):
    fold = next(e for e in tabs[demo.LINE_A].tabs[3].expander
                if e.label == wording.FOLD_MEASURED_CHART)
    assert not fold.caption
    at = _open(demo.LINE_A, ui_insights.EACH_KEY)
    fold = next(e for e in at.tabs[3].expander if e.label == wording.FOLD_MEASURED_CHART)
    assert [c.value for c in fold.caption] == [wording.MVP_CAPTION]


def test_a_disagreeing_repeat_names_its_interval():
    d = insights.Disagreement(21, 11, -10.5, 20, 7.4)
    assert ui_insights.disagreement_text(d) == (
        "Formulation 21, a Repeat of Formulation 11, scored 10.5 points lower. The 95 % "
        "interval for the difference of two results is ±7.4 points. If something changed "
        "(panel, supplier, equipment), check it before relying on the next suggestions.")


def test_the_correlated_pair_carries_its_r(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests.test_spread_fixes import _three
    opt = _three([(4.0, 2.0, 1.0, 150), (5.0, 3.1, 0.2, 200), (6.0, 3.9, 1.7, 160),
                  (7.0, 5.2, 0.9, 190)])
    a, b, r = insights.correlated_pair(opt)
    assert (a, b) == ("Pea", "Oil") == insights.moved_together(opt)
    assert r >= insights.TOGETHER
    # T0 review I-4: the pair is ranked by |r|, and the sentence says so.
    assert f"(|r| = {r:.2f})" in wording.moved_together(a, b, f"{r:.2f}")
