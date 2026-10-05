"""0.8.0 review fixes, Task 8: "What happens if you change one ingredient?"
ranks by a chosen measurement too, names two when they are nearly tied,
says when two ingredients moved together, and greys the path where nothing
was tested near (spec 2026-09-29 §4, §9 I13, minor 16)."""
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import CLOSE_TO_THE_BEST
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_insights import _burger
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401
from tests.test_insights_wave_c_ui import _rerun_on


@pytest.fixture(autouse=True)
def _a_model_with_skill(monkeypatch):
    """These tests are about the effect lines, so they stand in a model
    whose Round-ahead skill lets them be said: the 0.8.0 e2e fixes' A3 gate
    (review I4) withholds them below 10 checks or skill 0.20, which these
    small projects are. tests/test_102_w1_insights.py covers the gate."""
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)



@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  §4.1 ranking by a measurement
# ------------------------------------------------------------------ #

def test_spreads_by_a_measurement_use_that_measurements_predictions():
    opt = _burger(rounds=4, noise=0.2)
    fitted = insights.fit_all(opt)
    start = insights.start_recipe(opt, opt.best_index())
    found = insights.spreads(opt, fitted, start, "Firmness")
    assert found.moved
    for s in found.moved:
        path = insights.build_path(opt, s.name, start)
        means = [p.mean for p in insights.predict(opt, fitted["Firmness"],
                                                  [opt._encode(r) for r in path.recipes])]
        assert s.spread == pytest.approx(max(means) - min(means))
    assert insights.spreads(opt, fitted, start) == insights.spreads(opt, fitted, start,
                                                                    insights.SCORE)


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_compare_all_ranks_by_the_chosen_measurement(insights_scored, monkeypatch):
    drawn = []
    real = ui_insights.change_chart
    monkeypatch.setattr(ui_insights, "change_chart",
                        lambda s, *a: drawn.append((s, a)) or real(s, *a))
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    box = at.tabs[3].selectbox(key=ui_insights.SPREAD_KEY)
    assert box.options[0] == wording.OVERALL_SCORE_COLUMN
    assert box.options[1:] == ["Firmness (N)", "Juiciness (/10)"]     # Results' order
    box.set_value("Firmness")
    _rerun_on(at, wording.TAB_INSIGHTS)
    changes, (axis, d) = drawn[-1]
    assert axis == "Effect on predicted Firmness (N)" and d == 1
    # Review plots: the full reading is the short caption's "?".
    caps = [c.proto.help for c in at.tabs[3].caption]
    assert wording.change_caption(measure="Firmness") in caps


# ------------------------------------------------------------------ #
#  §4.2 near-ties
# ------------------------------------------------------------------ #

def test_two_are_named_when_the_second_is_within_20_percent():
    S = insights.Spread
    assert insights.near_tie([S("Oil", 8.8), S("Beet", 7.9), S("Pea", 1.0)]) == [
        S("Oil", 8.8), S("Beet", 7.9)]
    assert insights.near_tie([S("Oil", 10.0), S("Beet", 7.9)]) is None
    assert insights.near_tie([S("Oil", 10.0), S("Beet", 8.0)]) is not None   # exactly 80 %
    assert insights.near_tie([S("Oil", 10.0)]) is None


# ------------------------------------------------------------------ #
#  §4.3 moved together
# ------------------------------------------------------------------ #

def _three(rows):
    """Pea 4-8, Oil 2-6, Salt 0-2, Water fills to 100, plus a process
    setting; `rows` are (Pea, Oil, Salt, Temp), all fully measured."""
    opt = _burger(rounds=0)
    opt.add_ingredient("Salt", 0, 2)
    opt.add_process_parameter("Temp", 150, 200, baseline=170)
    for no, (pea, oil, salt, temp) in enumerate(rows, start=1):
        opt.tell(opt.fill_formulas({"Pea": pea, "Oil": oil, "Salt": salt, "Temp": temp}),
                 {"Firmness": 5.0, "Juiciness": 5.0}, formulation_no=no, batch_no=1)
    return opt


def test_two_ingredients_moved_together_are_named():
    opt = _three([(4.0, 2.0, 1.0, 150), (5.0, 3.1, 0.2, 200), (6.0, 3.9, 1.7, 160),
                  (7.0, 5.2, 0.9, 190)])
    assert insights.moved_together(opt) == ("Pea", "Oil")
    assert wording.moved_together("Coconut oil", "Beet juice powder") == (
        "Coconut oil and Beet juice powder are correlated across the formulations with all "
        "results in, so their separate effects cannot be told apart.")


def test_the_fill_a_setting_and_a_row_that_never_moved_are_left_out():
    # Pea and Temp move together, Water (the fill) follows Pea; Salt never moves.
    opt = _three([(4.0, 5.0, 1.0, 150), (5.0, 2.0, 1.0, 160), (6.0, 6.0, 1.0, 170),
                  (7.0, 3.0, 1.0, 180)])
    assert insights.moved_together(opt) is None


def test_fewer_than_three_formulations_say_nothing():
    assert insights.moved_together(_three([(4.0, 2.0, 1.0, 150), (5.0, 3.0, 0.2, 200)])) is None
    assert insights.TOGETHER == 0.8


# ------------------------------------------------------------------ #
#  §9 I13 the path's grey
# ------------------------------------------------------------------ #

def test_a_path_point_is_tested_when_a_measured_formulation_is_near_on_every_row():
    opt = _burger(rounds=3)
    start = insights.start_recipe(opt, opt.best_index())
    path = insights.build_path(opt, "Oil", start)
    near = insights.tested_near(opt, path)
    assert len(near) == len(path.amounts)
    k = path.amounts.index(path.start_amount)
    assert near[k]                                   # the start itself was measured
    tested = [insights.start_recipe(opt, i) for i in range(len(opt.Y_history))]
    for r, flag in zip(path.recipes, near):
        d = min(insights.distance(opt, r, t) for t in tested)
        assert flag == (d <= CLOSE_TO_THE_BEST + 1e-9)


def test_untested_runs_cover_each_stretch_without_gaps():
    class P:
        amounts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    near = [False, True, True, False, False, True]
    assert insights.untested_runs(P, near) == [(0.0, 0.5), (2.5, 4.5)]
    assert insights.untested_runs(P, [True] * 6) == []
    assert insights.untested_runs(P, [False] * 6) == [(0.0, 5.0)]


def test_the_grey_caption_uses_the_near_rule():
    assert wording.path_key(9, grey=True) == (
        "Band: 95 % interval. Grey: not tested near here. Dotted: Formulation 9.")


def test_the_new_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.moved_together("A", "B"), wording.SPREAD_ON,
                 wording.change_axis("Firmness", "N"), wording.path_key(9, grey=True),
                 wording.change_caption(measure="Firmness")):
        assert not any(p.search(text) for p in _BANNED), text
