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
from tests.test_app_ui import APP_PATH, _submit_button, burger, ferment, scored  # noqa: F401
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


def _ferment_table(at):
    return next(d for d in at.dataframe
                if "Incubation temperature (°C)" in d.value.columns)


def test_the_round_screen_reads_trial_on_a_process_only_project(warm_ferment):
    at = _app()
    at.run()
    assert not at.exception
    table = _ferment_table(at)
    assert wording.PREDICTED_SCORE_COLUMN in table.value.columns
    proto = str(table.proto)
    trial_help = wording.for_project(warm_ferment, wording.PREDICTED_SCORE_HELP)
    assert "trial" in trial_help and "formulation" not in trial_help
    assert trial_help in proto and wording.PREDICTED_SCORE_HELP not in proto
    captions = [c.value for c in at.caption]
    trial_caption = wording.for_project(warm_ferment, wording.PREDICTIONS_ROUND_CAPTION)
    assert "trial" in trial_caption and "formulation" not in trial_caption
    assert captions.count(trial_caption) == 1
    assert wording.PREDICTIONS_ROUND_CAPTION not in captions
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    frame = exp.dataframe[0].value
    assert list(frame.columns) == [wording.TRIAL_CAP, "Acidity (pH)"]


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


# ------------------------------------------------------------------ #
#  Task 3: which ingredients matter most
# ------------------------------------------------------------------ #

def _spreads(monkeypatch, *pairs, held=()):
    monkeypatch.setattr(insights, "spreads", lambda opt, fitted, start: insights.Spreads(
        [insights.Spread(n, s) for n, s in pairs], list(held)))


def test_section_2_answer_line_and_compare_expander(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    opt = FoodOptimizer("burger")
    fitted = insights.fit_all(opt)
    [top, _] = insights.spreads(opt, fitted, insights.start_recipe(opt, opt.best_index())).moved
    line = wording.spread_line(top.name, f"{top.spread:.1f}", "Best so far")
    assert line in md
    # Above the two boxes: it is the answer, the boxes are the detail.
    assert md.index(line) == md.index(f"**{wording.q_change(False)}**") + 1
    assert wording.compare_all(False) in [e.label for e in at.tabs[3].expander]
    assert wording.SPREAD_CAPTION in caps


def test_the_box_opens_on_the_largest_spread_then_keeps_the_choice(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Methylcellulose", 9.0), ("Pea protein", 3.0))
    at = _open_insights(_app())
    box = at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY)
    assert box.value == "Methylcellulose"
    box.set_value("Pea protein")
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY).value == "Pea protein"


def test_the_answer_names_another_start(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Pea protein", 6.25), ("Methylcellulose", 1.0))
    at = _open_insights(_app())
    at.tabs[3].selectbox(key=ui_insights.START_KEY).set_value(9)
    _rerun_on(at, wording.TAB_INSIGHTS)
    md, _ = _tab_text(at)
    assert wording.spread_line("Pea protein", "6.2", "Formulation 9") in md


def test_no_spreads_no_answer_line(insights_scored, monkeypatch):
    _spreads(monkeypatch)
    at = _open_insights(_app())
    assert not at.exception
    md, _ = _tab_text(at)
    assert not [m for m in md if m.startswith("Across its allowed amounts")]
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]
    assert at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY).value == "Pea protein"


def test_one_choice_answer_line_and_no_expander(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Pea protein", 4.0))
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert wording.spread_line("Pea protein", "4.0", "Best so far", only=True) in md
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]


def test_a_flat_model_says_so(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Pea protein", 0.04), ("Methylcellulose", 0.0))
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert wording.SPREAD_FLAT in md
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]


def test_a_spread_under_a_tenth_is_flat(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Pea protein", 0.09), ("Methylcellulose", 0.0))
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert wording.SPREAD_FLAT in md
    assert not [m for m in md if m.startswith("Across its allowed")]


def test_a_spread_of_a_tenth_or_more_is_not_flat(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Pea protein", 0.12), ("Methylcellulose", 0.0))
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert wording.SPREAD_FLAT not in md
    assert wording.spread_line("Pea protein", "0.1", "Best so far") in md


def test_a_choice_boxed_in_by_the_limits_is_named_under_the_answer(insights_scored):
    """Best so far sits against a limit on Pea protein + Methylcellulose both
    ways: Pea protein's path covers under 2 % of its 0–25 g, so it is held in
    place, not read as having no effect; Methylcellulose still moves."""
    best = insights_scored.best_index()
    start = insights.start_recipe(insights_scored, best)
    names = ["Pea protein", "Methylcellulose"]
    insights_scored.add_quantity_constraint(names, exactly=sum(start[n] for n in names))
    opt = FoodOptimizer("burger")
    found = insights.spreads(opt, insights.fit_all(opt), insights.start_recipe(opt, best))
    assert found.held == ["Pea protein"] and [s.name for s in found.moved] == ["Methylcellulose"]
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    top = found.moved[0]
    line = wording.spread_line(top.name, f"{top.spread:.1f}", "Best so far", only=True)
    assert line in md
    held = wording.held_line("Best so far", ["Pea protein"])
    assert held == "Kept in place by your limits from Best so far: Pea protein."
    assert caps.count(held) == 1
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]
    assert at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY).value == "Methylcellulose"


def test_held_choices_stay_out_of_the_compare_chart(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Methylcellulose", 5.0), ("Pea protein", 2.0), held=["Salt"])
    drawn = []
    real = ui_insights.spread_chart
    monkeypatch.setattr(ui_insights, "spread_chart", lambda s: drawn.append(s) or real(s))
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert wording.spread_line("Methylcellulose", "5.0", "Best so far") in md
    assert wording.held_line("Best so far", ["Salt"]) in caps
    assert wording.compare_all(False) in [e.label for e in at.tabs[3].expander]
    assert [[s.name for s in one] for one in drawn] == [["Methylcellulose", "Pea protein"]]


def test_a_flat_answer_still_names_what_is_held(insights_scored, monkeypatch):
    _spreads(monkeypatch, ("Methylcellulose", 0.02), held=["Pea protein"])
    at = _open_insights(_app())
    at.tabs[3].selectbox(key=ui_insights.START_KEY).set_value(9)
    _rerun_on(at, wording.TAB_INSIGHTS)
    md, caps = _tab_text(at)
    assert wording.SPREAD_FLAT in md
    assert wording.held_line("Formulation 9", ["Pea protein"]) in caps


def test_every_choice_held_says_the_limits_hold_them(insights_scored, monkeypatch):
    _spreads(monkeypatch, held=["Pea protein", "Methylcellulose"])
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    line = wording.held_all("Best so far")
    assert line == ("From Best so far, your limits keep every ingredient and process setting "
                    "in place.")
    assert line in md
    assert wording.SPREAD_FLAT not in md
    assert not [m for m in md if m.startswith("Across its allowed")]
    assert not [c for c in caps if c.startswith("Kept in place")]
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]


def test_a_failed_spread_leaves_the_path(insights_scored, monkeypatch):
    def boom(*a):
        raise RuntimeError("no spread")
    monkeypatch.setattr(insights, "spreads", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert not [m for m in md if m.startswith("Across its allowed amounts")]
    assert [c for c in caps if c.startswith("More Pea protein")]


def test_spread_chart_is_horizontal_largest_first_without_error_bars():
    chart = ui_insights.spread_chart([insights.Spread("Oil", 6.2), insights.Spread("Pea", 2.0)])
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["y"]["sort"] == ["Oil", "Pea"]
    assert spec["encoding"]["x"]["title"] == wording.SPREAD_AXIS
    assert "errorbar" not in str(spec)
    assert ui_insights.spread_chart([]) is None


# ------------------------------------------------------------------ #
#  Task 5: which formulation is predicted to score highest
# ------------------------------------------------------------------ #

def _best_frames(at):
    return [d.value for d in at.tabs[3].dataframe
            if wording.FORMULATION_CAP in d.value.columns
            and wording.BEST_ROW in list(d.value[wording.FORMULATION_CAP])]


def test_section_4_answer_table_and_caption(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    q = f"**{wording.Q_BEST}**"
    assert md.index(f"**{wording.Q_IMPROVING}**") < md.index(q)
    answer = md[md.index(q) + 1]
    opt = FoodOptimizer("burger")
    best = insights.predicted_best(opt, insights.fit_all(opt))
    p = best.score
    assert answer.startswith(wording.best_answer(f"{p.mean:.0f}", f"{p.low:.0f}", f"{p.high:.0f}"))
    assert wording.BEST_CAPTION in caps
    [frame] = _best_frames(at)
    assert list(frame.columns) == [wording.FORMULATION_CAP, "Pea protein (g)",
                                   "Methylcellulose (g)", "Total (g)"]
    labels = list(frame[wording.FORMULATION_CAP])
    assert labels[0] == wording.BEST_ROW
    assert labels[1] in (wording.start_best(opt.best_formulation_no()),
                         *(wording.start_formulation(int(n)) for n in opt.formulation_ids))


def _best(monkeypatch, close):
    def fake(opt, fitted):
        return insights.PredictedBest(
            dict(opt.recipe_history[0]), insights.Prediction(88.2, 80.4, 95.1),
            close, close if close is not None else opt.best_index())
    monkeypatch.setattr(insights, "predicted_best", fake)


def test_section_4_close_to_a_tested_formulation(insights_scored, monkeypatch):
    _best(monkeypatch, close=4)
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    # Read back from disk: opening the project puts the shares on 100.
    opt = FoodOptimizer("burger")
    no, score = int(opt.formulation_ids[4]), opt.Y_history[4]
    assert (wording.best_answer("88", "80", "95") + " "
            + wording.best_close(no, f"{score:.1f}")) in md
    [frame] = _best_frames(at)
    assert list(frame[wording.FORMULATION_CAP]) == [wording.BEST_ROW, wording.start_formulation(no)]


def test_section_4_not_tested_near_here(insights_scored, monkeypatch):
    _best(monkeypatch, close=None)
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert wording.best_answer("88", "80", "95") + " " + wording.BEST_UNTESTED in md
    [frame] = _best_frames(at)
    assert list(frame[wording.FORMULATION_CAP]) == [
        wording.BEST_ROW, wording.start_best(insights_scored.best_formulation_no())]


def test_section_4_shows_the_reason_a_round_could_not_be_made(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "predicted_best", lambda opt, fitted: wording.EVERYTHING_IS_FIXED)
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert wording.EVERYTHING_IS_FIXED in caps
    assert not [m for m in md if m.startswith("Predicted to score highest")]


def test_section_4_failure_is_one_line_and_not_retried(insights_scored, monkeypatch):
    calls = []
    def boom(opt, fitted):
        calls.append(1)
        raise RuntimeError("no best")
    monkeypatch.setattr(insights, "predicted_best", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert caps.count(wording.BEST_FAILED) == 1
    assert any(m.startswith("Best so far:") for m in md)
    assert wording.Q_PREDICTIONS in " ".join(md)
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert caps.count(wording.BEST_FAILED) == 1 and calls == [1]


def test_section_4_gate(scored):          # 2 formulations with all their results
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert f"**{wording.Q_BEST}**" in md
    assert caps.count(wording.insights_gate(2)) == 3      # sections 1, 2 and 4


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
    assert wording.Q_BEST in " ".join(_tab_text(at)[0])


def test_a_project_switch_drops_the_best(warm_round):
    at = _open_insights(_app())
    assert ui_insights.BEST_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.BEST_KEY not in at.session_state


def test_the_models_are_shared_with_question_4(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()                                          # Make a round
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert wording.Q_BEST in " ".join(_tab_text(at)[0])
    assert calls == [1]


def test_how_models_names_both_places():
    assert wording.HOW_MODELS == (
        "Each measurement, and the Overall score, has its own prediction learned from the "
        "formulations scored so far (a Gaussian process for each). They are used on this tab "
        "and for the predictions on Make a round; suggestions on Make a round still come from "
        "the Overall score alone. Because they are learned separately, the Overall score curve "
        "can differ a little from what the measurement curves imply.")


def test_every_wave_b_string_passes_the_vocabulary_guard():
    import re
    from tests.test_food_bo import _BANNED
    also = re.compile(r"\b(optimum|optimal|maximi[sz]e|confidence|probability|the model)\b", re.I)
    for s in (wording.PREDICTIONS_WORKING, wording.PREDICTED_SCORE_COLUMN,
              wording.PREDICTED_SCORE_HELP, wording.PREDICTIONS_ROUND_CAPTION,
              wording.PREDICTED_MEASUREMENTS,
              wording.spread_line("Coconut oil", "6.2", wording.spread_from(True, 14)),
              wording.spread_line("Coconut oil", "6.2", wording.spread_from(False, 9), only=True),
              wording.SPREAD_FLAT, wording.compare_all(True), wording.compare_all(False),
              wording.SPREAD_CAPTION, wording.SPREAD_AXIS, wording.Q_BEST,
              wording.best_answer("88", "80", "95"), wording.best_close(14, "81.0"),
              wording.BEST_UNTESTED, wording.BEST_ROW, wording.BEST_CAPTION,
              wording.BEST_FAILED, wording.HOW_MODELS):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s


def test_exact_wave_b_strings():
    assert wording.PREDICTED_SCORE_COLUMN == "Predicted Overall score"
    assert wording.PREDICTED_SCORE_HELP == (
        "Predicted from the formulations scored so far; about 19 in 20 results land in the likely range.")
    assert wording.PREDICTIONS_ROUND_CAPTION == (
        "Some formulations are chosen to try amounts not tested yet, so their prediction can be "
        "lower than Best so far.")
    assert wording.spread_line("Coconut oil", "6.2", wording.spread_from(True, 14)) == (
        "Across its allowed amounts, Coconut oil changes the predicted Overall score the most: "
        "a spread of 6.2 points (from Best so far).")
    assert wording.spread_from(False, 9) == "Formulation 9"
    assert wording.SPREAD_CAPTION == (
        "Spread = highest minus lowest predicted Overall score along each path, from the same "
        "starting formulation, including amounts not tested yet.")
    assert wording.SPREAD_FLAT == (
        "No ingredient or process setting changes the predicted Overall score by more than 0.1 "
        "points yet.")
    assert wording.Q_BEST == "Which formulation is predicted to score highest?"
    assert wording.best_answer("88", "80", "95") == "Predicted to score highest: 88 (likely 80–95)."
    assert wording.best_close(14, "81.0") == "Close to Formulation 14, which scored 81.0."
    assert wording.BEST_UNTESTED == "Not tested near here yet — treat it as a guess."
    assert wording.BEST_CAPTION == (
        "Suggestions on Make a round also try untested areas; this is the single formulation "
        "predicted to score highest.")
