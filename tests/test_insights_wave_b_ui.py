"""Insights wave B on screen: the predictions on Make a round, the answer and
compare chart of question 2, and question 4. AppTest, like
tests/test_insights_ui.py."""
import io

import openpyxl
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_insights_ui import _open_insights, _tab_text, insights_scored  # noqa: F401


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
    return next(d for d in at.dataframe if wording.FORMULATION_CAP in d.value.columns
                and "Pea protein (g)" in d.value.columns)


# ------------------------------------------------------------------ #
#  Task 2: predictions on Make a round
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


def _expected(project):
    opt = FoodOptimizer(project)
    from ui_helpers import open_rows
    return insights.round_predictions(opt, insights.fit_all(opt), open_rows(opt))


def test_the_round_table_gains_the_predicted_score_column(warm_round):
    at = _app()
    at.run()                                   # opens on Make a round
    assert not at.exception
    table = _batch_table(at)
    cols = list(table.value.columns)
    assert cols == [wording.FORMULATION_CAP, "Pea protein (g)", "Methylcellulose (g)",
                    "Total (g)", wording.PREDICTED_SCORE_COLUMN,
                    wording.compared_with_column(warm_round.best_formulation_no())]
    expected = _expected("burger")
    shown = list(table.value[wording.PREDICTED_SCORE_COLUMN])
    numbers = list(table.value[wording.FORMULATION_CAP])
    assert shown == [ui_insights.score_text(expected[no][insights.SCORE]) for no in numbers]
    assert all(" (" in cell and "–" in cell for cell in shown)


def test_the_column_explains_itself_in_its_tooltip(warm_round):
    at = _app()
    at.run()
    assert wording.PREDICTED_SCORE_HELP in str(_batch_table(at).proto)


def test_one_caption_and_the_predicted_measurements(warm_round):
    at = _app()
    at.run()
    captions = [c.value for c in at.caption]
    assert captions.count(wording.PREDICTIONS_ROUND_CAPTION) == 1
    assert wording.PREDICTED_SCORE_HELP not in captions        # tooltip, not a caption
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    frame = exp.dataframe[0].value
    assert list(frame.columns) == [wording.FORMULATION_CAP, "Juiciness (/10)", "Firmness (N)"]
    expected = _expected("burger")
    for _, row in frame.iterrows():
        p = expected[int(row[wording.FORMULATION_CAP])]["Firmness"]
        assert row["Firmness (N)"] == ui_insights.value_text(p)


def test_a_measurement_with_too_few_values_shows_a_dash(warm_round, monkeypatch):
    real = insights.fit_all

    def fewer(opt):
        fitted = real(opt)
        fitted["Juiciness"] = insights.Fitted("Juiciness", None, None, 3, 0.0, 10.0)
        return fitted
    monkeypatch.setattr(insights, "fit_all", fewer)
    at = _app()
    at.run()
    assert not at.exception
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    frame = exp.dataframe[0].value
    assert set(frame["Juiciness (/10)"]) == {"—"}
    assert "—" not in set(frame["Firmness (N)"])


def test_the_cold_start_round_has_no_predictions(scored, monkeypatch):
    scored.set_pending_batch([{"Pea protein": 12.0, "Methylcellulose": 1.0}])
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS not in [e.label for e in at.expander]
    assert calls == []


def test_editing_the_round_hides_the_column(warm_round):
    at = _app()
    at.run()
    at.button(key=f"edit_round_burger_{warm_round.pending_batch_no}_start").click().run()
    assert not at.exception
    assert any(i.value == wording.EDIT_FORMULATIONS_ACTIVE for i in at.info)
    [grid] = [d for d in at.dataframe if "Pea protein (g)" in d.value.columns]   # the editor
    assert wording.PREDICTED_SCORE_COLUMN not in grid.value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS not in [e.label for e in at.expander]


def test_recorded_and_not_scored_rows_show_nothing(warm_round):
    a, b, c = (r['formulation'] for r in warm_round.pending_batch)
    warm_round.tell(warm_round.pending_batch[0]['recipe'], {"Juiciness": 6.0, "Firmness": 5.0},
                    formulation_no=a, batch_no=warm_round.pending_batch_no)
    warm_round.record_skipped(b, warm_round.pending_batch_no, warm_round.pending_batch[1]['recipe'])
    at = _app()
    at.run()
    assert not at.exception
    table = _batch_table(at).value
    cells = dict(zip(table[wording.FORMULATION_CAP], table[wording.PREDICTED_SCORE_COLUMN]))
    assert cells[a] == "" and cells[b] == "" and cells[c] != ""
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    assert list(exp.dataframe[0].value[wording.FORMULATION_CAP]) == [c]


def test_the_round_tab_closed_fits_nothing(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.session_state["_loaded_project"] = "burger"   # no landing on Make a round
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.FITTED_KEY not in at.session_state


def test_the_models_are_fitted_once_for_both_screens(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()                                          # Make a round
    assert calls == [1]
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == [1] and ui_insights.CACHE_KEY in at.session_state
    assert wording.Q_PREDICTIONS in " ".join(_tab_text(at)[0])
    _rerun_on(at, wording.TAB_BATCH)
    assert calls == [1]
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns


def test_the_round_screen_says_what_it_is_preparing(warm_round, monkeypatch):
    said = []
    real = ui_insights.fitted_models
    monkeypatch.setattr(ui_insights, "fitted_models",
                        lambda opt, working=wording.INSIGHTS_WORKING: said.append(working) or real(opt, working))
    at = _app()
    at.run()
    assert said == [wording.PREDICTIONS_WORKING]
    assert wording.PREDICTIONS_WORKING == "Preparing the predictions…"


def test_a_failed_fit_leaves_the_table_as_it_was(warm_round, monkeypatch):
    calls = []
    def boom(opt):
        calls.append(1)
        raise RuntimeError("no fit")
    monkeypatch.setattr(insights, "fit_all", boom)
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns
    _rerun_on(at, wording.TAB_BATCH)
    assert calls == [1]


def _cells(book):
    return [[list(r) for r in ws.iter_rows(values_only=True)] for ws in book.worksheets]


def test_the_workbook_is_unchanged(warm_round):
    opt = FoodOptimizer("burger")
    before = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    at = _app()
    at.run()
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    opt = at.session_state["optimizer"]
    after = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    assert after == before
    assert wording.PREDICTED_SCORE_COLUMN not in opt.batch_frame(opt.pending_batch).columns


def test_a_project_switch_drops_the_shared_models(warm_round):
    at = _open_insights(_app())
    assert ui_insights.FITTED_KEY in at.session_state
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


def test_score_and_value_text():
    p = insights.Prediction(78.4, 70.2, 85.49)
    assert ui_insights.score_text(p) == "78 (70–85)"
    assert ui_insights.value_text(insights.Prediction(5.81, 5.1, 6.54)) == "5.81 (5.1–6.54)"
