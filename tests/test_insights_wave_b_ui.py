"""Insights wave B on screen: the predictions on Make a round, the answer and
compare chart of question 2, and question 4. AppTest, like
tests/test_insights_ui.py."""
import io

import openpyxl
import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, ferment, scored  # noqa: F401
from tests.test_insights_ui import _open_how, _open_insights, _tab_text, insights_scored  # noqa: F401


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _counting(monkeypatch, name):
    calls = []
    real = getattr(insights, name)
    monkeypatch.setattr(insights, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _rerun_on(at, tab):
    """Run again with `tab` open. AppTest does not replay the tab bar: a
    plain at.run() after a tab was chosen draws the first tab again."""
    at.session_state["main_tab"] = tab
    return at.run()


def _batch_table(at):
    """v3: the Round table (ingredients down, the Round across)."""
    return next(t for t in at.table if "Pea protein" in t.value.index)


# ------------------------------------------------------------------ #
#  Task 2: predictions on Make a round (v3: gone from that screen; what
#  is left checks it fits nothing and leaves the workbook alone)
# ------------------------------------------------------------------ #

@pytest.fixture
def warm_round(insights_scored):
    """Nine formulations with all their results (the spread-out start is
    over) and Round 4 open with three formulations."""
    insights_scored.set_pending_batch([
        {"Pea protein": 12.0, "Methylcellulose": 1.0},
        {"Pea protein": 18.0, "Methylcellulose": 2.0},
        {"Pea protein": 22.0, "Methylcellulose": 2.5}])
    return insights_scored


@pytest.fixture
def warm_ferment(ferment):
    """Five scored trials (enough for a model) and one more trial open, on a
    project made of process settings alone: no ingredients, so every one of
    the Round-screen strings must read "trial", never "formulation"."""
    for no, (temp, hours) in enumerate(
            ((32.0, 5.0), (34.0, 7.0), (36.0, 9.0), (38.0, 11.0), (40.0, 13.0)), start=1):
        ferment.tell({"Incubation temperature": temp, "Incubation time": hours},
                     {"Acidity": 3.0 + 0.1 * temp}, formulation_no=no, batch_no=1)
    ferment.set_pending_batch([{"Incubation temperature": 37.0, "Incubation time": 8.0}])
    return ferment


def test_editing_the_round_hides_the_column(warm_round):
    at = _app()
    at.run()
    at.button(key=f"edit_round_burger_{warm_round.pending_batch_no}_start").click().run()
    assert not at.exception
    assert any(i.value == wording.EDIT_FORMULATIONS_ACTIVE for i in at.info)
    [grid] = [d for d in at.dataframe if "Pea protein (g)" in d.value.columns]   # the editor
    assert 'Predicted Overall score (95\xa0% range)' not in grid.value.columns
    assert 'Exploring suggestions can predict below Best so far.' not in [c.value for c in at.caption]
    assert 'Predicted measurements (95\xa0% range)' not in [e.label for e in at.expander]


def test_the_round_tab_closed_fits_nothing(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.session_state["_loaded_project"] = "burger"   # no landing on Make a round
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.FITTED_KEY not in at.session_state


def _cells(book):
    return [[list(r) for r in ws.iter_rows(values_only=True)] for ws in book.worksheets]


def test_the_workbook_is_unchanged(warm_round):
    opt = FoodOptimizer("burger")
    before = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    at = _app()
    at.run()
    assert _batch_table(at) is not None          # v3: the Round table, no forecast
    opt = at.session_state["optimizer"]
    after = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    assert after == before
    assert 'Predicted Overall score (95\xa0% range)' not in opt.batch_frame(opt.pending_batch).columns


def test_a_project_switch_drops_the_shared_models(warm_round):
    at = _open_insights(_app())
    # v3: the Review fits nothing, so the shared models are put in place
    # here, as Make a round's fit leaves them; the switch must drop them.
    at.session_state[ui_insights.FITTED_KEY] = {"stale": True}
    at.session_state[ui_insights.CACHE_KEY] = {"stale": True}
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.FITTED_KEY not in at.session_state
    assert ui_insights.CACHE_KEY not in at.session_state


@pytest.mark.parametrize("low, high, text", [(70.2, 85.4, "70–85"), (-1.25, 0.4, "-1.25 to 0.4"),
                                             (-3.0, -1.0, "-3 to -1")])
def test_range_text_uses_to_beside_a_minus_sign(low, high, text):
    fmt = (lambda v: f"{v:.0f}") if text in ("70–85", "-3 to -1") else None
    assert ui_insights.range_text(low, high, fmt) == text


# ------------------------------------------------------------------ #
#  Task 3: which ingredients matter most
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_a_failed_ranking_leaves_the_path(insights_scored, monkeypatch):
    """tested_near_changes failing drops the fold's first line; the path
    and its captions still draw (ui_insights._change's fallback)."""
    def boom(*a, **k):
        raise RuntimeError("no ranking")
    monkeypatch.setattr(insights, "tested_near_changes", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert not [m for m in md if m.startswith(("Near what you have tested",
                                                "Nothing has been tested near"))]
    assert [c for c in caps if c.startswith("More Pea protein")]


# ------------------------------------------------------------------ #
#  Task 5: which formulation is predicted to score highest
# ------------------------------------------------------------------ #

def _best_frames(at):
    return [d.value for d in at.tabs[3].dataframe
            if wording.FORMULATION_CAP in d.value.columns
            and wording.BEST_ROW in list(d.value[wording.FORMULATION_CAP])]


def _best(monkeypatch, close):
    def fake(opt, fitted, checks=()):
        return insights.PredictedBest(
            dict(opt.recipe_history[0]), insights.Prediction(88.2, 80.4, 95.1),
            close, close if close is not None else opt.best_index())
    monkeypatch.setattr(insights, "predicted_best", fake)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_section_4_shows_the_reason_a_round_could_not_be_made(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "predicted_best",
                        lambda opt, fitted, checks=(): wording.EVERYTHING_IS_FIXED)
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert wording.EVERYTHING_IS_FIXED in caps
    assert not [m for m in md if m.startswith("Predicted to score highest")]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_limit_edit_works_out_the_best_again(insights_scored, monkeypatch):
    calls = _counting(monkeypatch, "predicted_best")
    at = _open_insights(_app())
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert calls == [1]
    at.session_state["optimizer"].add_quantity_constraint(
        ["Pea protein", "Methylcellulose"], max_val=15.0)
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == [1, 1]
    assert wording.Q_STAND in " ".join(_tab_text(at)[0])


def test_a_project_switch_drops_the_best(warm_round):
    at = _open_insights(_app())
    # Review v3 fits nothing: the key as Make a round's work leaves it.
    at.session_state[ui_insights.BEST_KEY] = {"stale": True}
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.BEST_KEY not in at.session_state


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_models_are_shared_with_question_4(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()                                          # Make a round
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert wording.Q_STAND in " ".join(_tab_text(at)[0])
    assert calls == [1]


def test_how_models_names_both_places():
    """03 §5.7: the models, the kernel and the training rows named."""
    assert wording.HOW_MODELS == (
        "Models: one Gaussian process per measurement and one for the Overall score, each "
        "fitted to the formulations that have that value (default kernel: squared exponential, "
        "one correlation length per varying row; noise fitted). Suggestions use the "
        "Overall-score model only. It is fitted to the scores themselves, not built from the "
        "measurement models, so the two can disagree slightly.")


def test_every_wave_b_string_passes_the_vocabulary_guard():
    import re
    from tests.test_food_bo import _BANNED
    # "the model" is allowed from the BO-views wave on (review I5: the
    # caption says what the suggestions weigh).
    also = re.compile(r"\b(optimum|optimal|maximi[sz]e|confidence|probability)\b", re.I)
    for s in (wording.PREDICTIONS_WORKING,
              wording.compare_all(True), wording.compare_all(False),
              wording.BEST_ROW, wording.BEST_FAILED, wording.HOW_MODELS,
              wording.held_line("Best so far", ["Wheat gluten", "Sunflower oil"]),
              wording.held_all("Formulation 9"), wording.held_all("Best so far", True),
              wording.compare_all(False, False), ):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s


# ------------------------------------------------------------------ #
#  One vocabulary across Insights: trials, process settings, values
# ------------------------------------------------------------------ #

def _everything_said(tab):
    said = ([m.value for m in tab.markdown] + [c.value for c in tab.caption]
            + [e.label for e in tab.expander] + [s.label for s in tab.selectbox])
    for box in tab.selectbox:
        said += [str(o) for o in box.options]
    for d in tab.dataframe:
        said += [str(c) for c in d.value.columns]
        said += [str(v) for v in d.value.astype(str).to_numpy().ravel()]
    return said


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_insights_reads_trial_and_setting_on_a_process_only_project(warm_ferment):
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    said = _everything_said(tab)
    assert not [t for t in said if "ormulation" in t], [t for t in said if "ormulation" in t]
    assert not [t for t in said if "ngredient" in t], [t for t in said if "ngredient" in t]
    md, caps = _tab_text(at)
    # The three answers and the folds (spec 2026-09-30 §2, review I14).
    assert f"##### {wording.Q_STAND}" in md
    assert "Try one process setting" in [e.label for e in tab.expander]
    assert tab.selectbox(key=ui_insights.CHOICE_KEY).label == "Process setting"
    assert [c for c in caps if c.startswith("5 scored trials")]


def _best_frames_for(tab, first):
    return [d.value for d in tab.dataframe
            if first in d.value.columns and wording.BEST_ROW in list(d.value[first])]


@pytest.fixture
def mixed_scored(burger):
    """The burger with a process setting as well, nine formulations scored."""
    burger.add_process_parameter("Mixing time", 2, 10, unit="min")
    no = 0
    for r in (1, 2, 3):
        for pea, mc, mix in ((8.0, 0.5, 3.0), (14.0, 1.5, 6.0), (20.0, 2.5, 9.0)):
            no += 1
            pea += r * 0.7
            burger.tell({"Pea protein": pea, "Methylcellulose": mc, "Mixing time": mix},
                        {"Juiciness": 3 + 0.2 * pea, "Firmness": 1 + 0.25 * pea + mc},
                        formulation_no=no, batch_no=r)
    return burger


# ------------------------------------------------------------------ #
#  Final fix wave (waves B + C): carried from wave B's final review
# ------------------------------------------------------------------ #

def test_a_hidden_round_tab_predicts_nothing_even_when_warm(warm_round, monkeypatch):
    at = _app()
    at.run()
    calls = _counting(monkeypatch, "round_predictions")
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == []


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_question_2_keeps_ingredient_when_nothing_varies(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert wording.try_one_fold(False) in [e.label for e in at.tabs[3].expander]
    assert wording.CHANGE_NONE in caps


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_a_process_only_project_says_values_and_settings(warm_ferment, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_how(_open_insights(_app()))
    assert not at.exception
    md, caps = _tab_text(at)
    assert "Try one process setting" in [e.label for e in at.tabs[3].expander]
    assert "Nothing to change yet: every process setting is fixed." in caps
    [how] = [e for e in at.tabs[3].expander if e.label == wording.FOLD_METHOD]
    checks = next(m.value for m in how.markdown if m.value.startswith("Round-ahead check:"))
    assert "at the values it was run at" in checks and "amounts" not in checks
    assert "ngredient" not in " ".join(md + caps)


def test_the_process_only_lines_pass_the_word_guards():
    import re
    from tests.test_food_bo import _BANNED
    also = re.compile(r"\b(optimum|optimal|maximi[sz]e|confidence|probability|the model)\b", re.I)
    for s in (wording.how_checks(values=True), wording.change_none(settings_only=True)):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s
    assert wording.how_checks() == wording.HOW_CHECKS
    assert wording.change_none() == wording.CHANGE_NONE
