"""0.8.0 BO views, Task 6 (spec 2026-09-30-bo-views §2, §3.2, §4, §8;
review I1-I7, M6, M8, M11): Insights says its statistics by name with
their basis — the verdict's miss count, the Overall score's skill, the
Repeat-based noise threshold — under noun headings; says expected
improvement by Round in Progress; and stops stating noise as ingredient
effects when its model is fitting noise."""
import json
import os

import pytest
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
            yield out
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  Headings and the verdict (03 §4, §5.2; review I7)
# ------------------------------------------------------------------ #

def test_the_three_sections_are_real_noun_headings(tabs):
    md, _ = _texts(tabs[demo.LINE_A])
    heads = [m for m in md if m.startswith("##### ")]
    assert heads[:3] == [f"##### {q}" for q in (wording.Q_TRUST, wording.Q_STAND,
                                                wording.Q_NEXT)]


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

@pytest.mark.parametrize("name,text", [
    (demo.LINE_A, "Overall score, Rounds 2–5: skill 0.36 (n = 15), better than the average."),
    (demo.LINE_C, "Overall score, Rounds 2–5: skill −0.47 (n = 16), no better than the "
                  "average of earlier results."),
])
def test_the_skill_line_sits_under_the_verdict(tabs, name, text):
    md, _ = _texts(tabs[name])
    assert text in md
    assert md.index(text) < md.index(f"##### {wording.Q_STAND}")


def test_no_skill_is_the_next_step_on_line_c(tabs):
    md, _ = _texts(tabs[demo.LINE_C])
    opt = demo.load(demo.LINE_C)
    assert wording.next_no_skill("−0.47", 16, opt.pending_batch_no, 4) in md


def test_each_output_has_its_skill_in_the_details_table(tabs):
    how = next(e for e in tabs[demo.LINE_A].tabs[3].expander
               if e.label == wording.HOW_WORKED_OUT)
    [frame] = [d.value for d in how.dataframe if wording.SKILL_COLUMN in d.value.columns]
    cells = dict(zip(frame[wording.DETAILS_COLUMNS_2[0]], frame[wording.SKILL_COLUMN]))
    assert cells[wording.OVERALL_SCORE_COLUMN] == "0.36 (n = 15)"
    assert cells["Juiciness (/10)"] == "−0.04 (n = 16)"


def test_method_and_details_define_the_new_terms(tabs):
    how = next(e for e in tabs[demo.LINE_A].tabs[3].expander
               if e.label == wording.HOW_WORKED_OUT)
    said = _all(how)
    for text in (wording.HOW_VERDICT, wording.HOW_NOISE_THRESHOLD, wording.how_skill("0.00"),
                 wording.HOW_EXPECTED_IMPROVEMENT, wording.how_roles("3,000")):
        assert text in said, text
    assert wording.details_window([2, 3, 4, 5], 3, [3, 4, 5]) in said


# ------------------------------------------------------------------ #
#  §3.2 Expected improvement by Round (review I4, I5, M8)
# ------------------------------------------------------------------ #

def test_line_a_says_the_expected_improvement_of_its_open_round(tabs):
    md, _ = _texts(tabs[demo.LINE_A])
    [line] = [m for m in md if m.startswith("Expected improvement")]
    assert line.startswith("Expected improvement, Round 6: 0.12 points (model, one "
                           "formulation), under the 1-point noise threshold.")
    assert line.endswith("Recomputed from today's records.")
    assert md.index(line) < md.index(f"##### {wording.Q_NEXT}")


@pytest.mark.parametrize("name,reason", [(demo.LINE_B, wording.GAIN_REASON_VERDICT),
                                         (demo.LINE_C, wording.GAIN_REASON_SKILL)])
def test_the_number_is_withheld_when_the_model_is_not_trusted(tabs, name, reason):
    md, _ = _texts(tabs[name])
    assert wording.gain_unreliable(reason) in md


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
                or e.label.startswith("Effect of one"))


def test_line_c_names_no_ingredient_effect(tabs):
    said = _all(_fold(tabs["C try one"]))
    assert wording.guard_line([wording.GUARD_FLOOR]) in said
    assert not any(t.startswith(("Largest local effect", "Local effect of")) for t in said)
    assert wording.PATH_GUARD in said
    assert not [e for e in _fold(tabs["C try one"]).expander
                if e.label.startswith("Compare all")]


def test_line_a_still_ranks_its_effects(tabs):
    said = _all(_fold(tabs["A try one"]))
    assert any(t.startswith(("Largest local effect", "Local effect of", "No ingredient"))
               for t in said)
    assert wording.PATH_GUARD not in said and wording.PATH_GUARD_SKILL not in said


def test_the_guard_says_setting_effects_for_a_process_only_study():
    guard = insights.Guard(["floor"])
    assert "setting effects" in ui_insights.guard_text(guard, settings_only=True)
    assert ui_insights.guard_text(insights.Guard(["skill"])) == wording.guard_skill_line()


# ------------------------------------------------------------------ #
#  Measured against predicted and the other lines with their basis
# ------------------------------------------------------------------ #

def test_measured_against_predicted_is_drawn_only_while_open(tabs):
    fold = next(e for e in tabs[demo.LINE_A].tabs[3].expander
                if e.label == wording.SHOW_PREDICTIONS)
    assert not fold.caption
    at = _open(demo.LINE_A, ui_insights.EACH_KEY)
    fold = next(e for e in at.tabs[3].expander if e.label == wording.SHOW_PREDICTIONS)
    assert [c.value for c in fold.caption] == [wording.MVP_CAPTION]


def test_a_disagreeing_repeat_names_its_interval():
    d = insights.Disagreement(21, 11, -10.5, 20, 7.4)
    assert ui_insights.disagreement_text(d) == (
        "Formulation 21, a Repeat of Formulation 11, scored 10.5 points lower. The 95 % "
        "interval for the difference of two results is ±7.4. Check the panel, suppliers and "
        "equipment for a change.")


def test_the_correlated_pair_carries_its_r(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests.test_spread_fixes import _three
    opt = _three([(4.0, 2.0, 1.0, 150), (5.0, 3.1, 0.2, 200), (6.0, 3.9, 1.7, 160),
                  (7.0, 5.2, 0.9, 190)])
    a, b, r = insights.correlated_pair(opt)
    assert (a, b) == ("Pea", "Oil") == insights.moved_together(opt)
    assert r >= insights.TOGETHER
    assert f"(r = {r:.2f})" in wording.moved_together(a, b, f"{r:.2f}")
