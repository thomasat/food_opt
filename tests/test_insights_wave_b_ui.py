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


def _scores(monkeypatch, below=()):
    """The real predictions, with each Overall score moved just above Best
    so far's measured one, or well below it for the formulations in
    `below`."""
    real = insights.round_predictions

    def fake(opt, fitted, rows):
        got = real(opt, fitted, rows)
        top = float(opt.Y_history[opt.best_index()])
        for no, per in got.items():
            m = top - 5.0 if no in below else top + 0.5
            per[insights.SCORE] = insights.Prediction(m, m - 3.0, m + 3.0)
        return got
    monkeypatch.setattr(insights, "round_predictions", fake)


def test_one_caption_and_the_predicted_measurements(warm_round, monkeypatch):
    _scores(monkeypatch, below={warm_round.pending_batch[0]['formulation']})
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


def test_no_round_caption_when_every_suggestion_is_predicted_above_best(warm_round, monkeypatch):
    _scores(monkeypatch)
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS in [e.label for e in at.expander]


def test_an_own_formulation_predicted_below_best_needs_no_caption(warm_round, monkeypatch):
    """The caption explains why a SUGGESTION can be predicted lower; a
    formulation of the user's own was not chosen to try anything."""
    own = warm_round.add_to_pending_batch({"Pea protein": 5.0, "Methylcellulose": 0.2},
                                          note="our standard")
    _scores(monkeypatch, below={own})
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    _scores(monkeypatch, below={own, warm_round.pending_batch[1]['formulation']})
    _rerun_on(at, wording.TAB_BATCH)
    assert [c.value for c in at.caption].count(wording.PREDICTIONS_ROUND_CAPTION) == 1


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
    trial_caption = wording.for_project(warm_ferment, wording.predictions_round_caption(values=True))
    assert trial_caption == ("Some trials are chosen to try values not tested yet, so their "
                             "prediction can be lower than Best so far.")
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


def test_the_round_screen_is_drawn_before_the_fit(warm_round, monkeypatch):
    """The table, the Save results button and the record and upload steps
    are all on screen before the models are fitted; the table is then drawn
    again in its place with the column."""
    import ui_batch
    order = []
    for name in ("_record_results", "_upload"):
        real = getattr(ui_batch, name)
        monkeypatch.setattr(ui_batch, name,
                            lambda *a, _real=real, _name=name, **k: order.append(_name) or _real(*a, **k))
    real_fit = insights.fit_all
    monkeypatch.setattr(insights, "fit_all", lambda opt: order.append("fit") or real_fit(opt))
    at = _app()
    at.run()
    assert not at.exception
    assert order == ["_record_results", "_upload", "fit"]
    assert wording.SAVE_RESULTS in [b.label for b in at.button]
    [table] = [d for d in at.dataframe if "Pea protein (g)" in d.value.columns
               and wording.FORMULATION_CAP in d.value.columns]      # drawn once, in place
    assert wording.PREDICTED_SCORE_COLUMN in table.value.columns
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]


def test_no_open_rows_no_fit(warm_round, monkeypatch):
    rows = warm_round.pending_batch
    for row in rows[:2]:
        warm_round.tell(row['recipe'], {"Juiciness": 6.0, "Firmness": 5.0},
                        formulation_no=row['formulation'], batch_no=warm_round.pending_batch_no)
    warm_round.record_skipped(rows[2]['formulation'], warm_round.pending_batch_no, rows[2]['recipe'])
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.FITTED_KEY not in at.session_state
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns


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


def test_spread_chart_never_truncates_an_ingredient_name():
    """A long name like "Textured pea protein" must not be cut short on the
    y axis: labelLimit=0 turns off Vega-Lite's default truncation."""
    chart = ui_insights.spread_chart([insights.Spread("Textured pea protein", 6.2),
                                      insights.Spread("Oil", 2.0)])
    axis = chart.to_dict()["encoding"]["y"]["axis"]
    assert axis["labelLimit"] == 0
    assert axis["titleLimit"] == 0


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
                                   "Methylcellulose (g)", "Total (g)",
                                   wording.PREDICTED_SCORE_COLUMN]
    labels = list(frame[wording.FORMULATION_CAP])
    assert labels[0] == wording.BEST_ROW
    assert labels[1] in (wording.start_best(opt.best_formulation_no()),
                         *(wording.start_formulation(int(n)) for n in opt.formulation_ids))
    # Each row carries its own prediction, beside the other's: the compared
    # formulation's own prediction is never above the predicted best's.
    fitted = insights.fit_all(opt)
    [other] = insights.predict(opt, fitted[insights.SCORE],
                               [opt._encode(insights.start_recipe(opt, best.compare))])
    assert list(frame[wording.PREDICTED_SCORE_COLUMN]) == [ui_insights.score_text(p),
                                                          ui_insights.score_text(other)]
    assert other.mean <= p.mean + 1e-6


def test_section_4_rows_show_their_own_predictions(insights_scored, monkeypatch):
    """Best so far's own prediction sits beside the predicted best's: the
    measured score of a lucky formulation can be above both."""
    _best(monkeypatch, close=None)
    at = _open_insights(_app())
    [frame] = _best_frames(at)
    opt = FoodOptimizer("burger")
    [own] = insights.predict(opt, insights.fit_all(opt)[insights.SCORE],
                             [opt._encode(insights.start_recipe(opt, opt.best_index()))])
    assert list(frame[wording.PREDICTED_SCORE_COLUMN]) == [
        ui_insights.score_text(insights.Prediction(88.2, 80.4, 95.1)), ui_insights.score_text(own)]


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
    assert f"**{wording.Q_BEST}**" not in md               # under 5: one line instead
    assert caps.count(wording.insights_gate(2)) == 1


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
              wording.BEST_FAILED, wording.HOW_MODELS,
              wording.held_line("Best so far", ["Wheat gluten", "Sunflower oil"]),
              wording.held_all("Formulation 9"), wording.held_all("Best so far", True),
              wording.spread_flat(True), wording.compare_all(False, False),
              wording.spread_caption(True), wording.predictions_round_caption(True),
              wording.path_grey(True), wording.q_change(True, False),
              wording.path_process("Mixing time", 3, False),
              wording.spread_line("Mixing time", "2.0", "Best so far", values=True)):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s


def test_exact_wave_b_strings():
    assert wording.PREDICTED_SCORE_COLUMN == "Predicted Overall score (likely range)"
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


def test_insights_reads_trial_and_setting_on_a_process_only_project(warm_ferment):
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    said = _everything_said(tab)
    assert not [t for t in said if "ormulation" in t], [t for t in said if "ormulation" in t]
    assert not [t for t in said if "ngredient" in t], [t for t in said if "ngredient" in t]
    md, caps = _tab_text(at)
    assert "**What happens if you change one process setting?**" in md
    assert "**Which trial is predicted to score highest?**" in md
    assert tab.selectbox(key=ui_insights.CHOICE_KEY).label == "Process setting"
    assert [c for c in caps if c.startswith("Based on 5 scored trials")]
    answer = [m for m in md if m.startswith("Across its allowed")]
    assert answer and all(m.startswith("Across its allowed values, ") for m in answer)
    [best] = _best_frames_for(tab, wording.TRIAL_CAP)
    assert list(best[wording.TRIAL_CAP])[0] == wording.BEST_ROW


def test_the_process_only_compare_expander_and_caption(warm_ferment, monkeypatch):
    _spreads(monkeypatch, ("Incubation temperature", 0.8), ("Incubation time", 0.3))
    at = _open_insights(_app())
    tab = at.tabs[3]
    md, caps = _tab_text(at)
    assert ("Across its allowed values, Incubation temperature changes the predicted Overall "
            "score the most: a spread of 0.8 points (from Best so far).") in md
    assert "Compare all process settings" in [e.label for e in tab.expander]
    assert ("Spread = highest minus lowest predicted Overall score along each path, from the "
            "same starting trial, including values not tested yet.") in caps


def test_a_process_only_project_held_and_flat_lines(warm_ferment, monkeypatch):
    _spreads(monkeypatch, held=["Incubation temperature", "Incubation time"])
    at = _open_insights(_app())
    md, _ = _tab_text(at)
    assert "From Best so far, your limits keep every process setting in place." in md
    _spreads(monkeypatch, ("Incubation time", 0.02), held=["Incubation temperature"])
    _rerun_on(at, wording.TAB_INSIGHTS)
    md, caps = _tab_text(at)
    assert "No process setting changes the predicted Overall score by more than 0.1 points yet." in md
    assert "Kept in place by your limits from Best so far: Incubation temperature." in caps


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


@pytest.mark.parametrize("top, what", [("Mixing time", "values"), ("Pea protein", "amounts")])
def test_the_answer_says_values_for_a_setting_and_amounts_for_an_ingredient(
        mixed_scored, monkeypatch, top, what):
    other = "Pea protein" if top == "Mixing time" else "Mixing time"
    _spreads(monkeypatch, (top, 5.0), (other, 2.0))
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert (f"Across its allowed {what}, {top} changes the predicted Overall score the most: "
            "a spread of 5.0 points (from Best so far).") in md
    assert wording.compare_all(True) in [e.label for e in at.tabs[3].expander]
    assert wording.SPREAD_CAPTION in caps           # amounts and values: says amounts


def test_process_setting_wording():
    assert wording.q_change(True, False) == "What happens if you change one process setting?"
    assert wording.q_change(True) == "What happens if you change one ingredient or process setting?"
    assert wording.compare_all(True, False) == "Compare all process settings"
    assert wording.path_grey(True) == "Grey: values not tested yet — treat the curve there as a guess."
    assert wording.PATH_GREY == "Grey: amounts not tested yet — treat the curve there as a guess."
    assert wording.path_process("Incubation time", 1, False) == (
        "Incubation time changes; everything else stays as in Formulation 1.")
    assert wording.path_process("Mixing time", 1) == (
        "Mixing time changes; the ingredients stay as in Formulation 1.")
    assert wording.PREDICTIONS_ROUND_CAPTION == wording.predictions_round_caption()
    assert wording.SPREAD_CAPTION == wording.spread_caption()
    assert wording.SPREAD_FLAT == wording.spread_flat()


# ------------------------------------------------------------------ #
#  Final fix wave (waves B + C): carried from wave B's final review
# ------------------------------------------------------------------ #

def test_a_warm_round_table_is_drawn_once_with_its_column(warm_round, monkeypatch):
    """With the models already fitted for this data, the table is drawn
    with the Predicted Overall score column on the first pass: no draw
    without it first, so the column does not blink on every rerun."""
    import ui_batch
    at = _app()
    at.run()                                   # cold: fits, then fills the column
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    drawn = []
    real = ui_batch._draw_table
    monkeypatch.setattr(ui_batch, "_draw_table",
                        lambda opt, slot, frame: drawn.append(list(frame.columns))
                        or real(opt, slot, frame))
    calls = _counting(monkeypatch, "fit_all")
    _rerun_on(at, wording.TAB_BATCH)
    assert not at.exception
    assert calls == []
    assert len(drawn) == 1 and wording.PREDICTED_SCORE_COLUMN in drawn[0]
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    assert wording.PREDICTED_MEASUREMENTS in [e.label for e in at.expander]


def test_a_cold_round_table_is_drawn_first_and_filled_after(warm_round, monkeypatch):
    import ui_batch
    drawn = []
    real = ui_batch._draw_table
    monkeypatch.setattr(ui_batch, "_draw_table",
                        lambda opt, slot, frame: drawn.append(list(frame.columns))
                        or real(opt, slot, frame))
    at = _app()
    at.run()
    assert not at.exception
    assert [wording.PREDICTED_SCORE_COLUMN in cols for cols in drawn] == [False, True]


def test_a_hidden_round_tab_predicts_nothing_even_when_warm(warm_round, monkeypatch):
    at = _app()
    at.run()
    calls = _counting(monkeypatch, "round_predictions")
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == []


def test_question_4_stands_when_its_own_prediction_fails(insights_scored, monkeypatch):
    import inspect
    _best(monkeypatch, close=None)
    real = insights.predict

    def boom(*a, **k):
        if inspect.stack()[1].function == "best_frame":
            raise RuntimeError("no prediction")
        return real(*a, **k)
    monkeypatch.setattr(insights, "predict", boom)
    at = _open_insights(_app())
    assert not at.exception
    [frame] = _best_frames(at)
    assert wording.PREDICTED_SCORE_COLUMN not in frame.columns
    assert list(frame[wording.FORMULATION_CAP])[0] == wording.BEST_ROW
    labels = [e.label for e in at.tabs[3].expander]
    assert wording.MORE_CHARTS in labels and wording.HOW_WORKED_OUT in labels


def test_question_2_keeps_ingredient_when_nothing_varies(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert "**What happens if you change one ingredient?**" in md
    assert wording.CHANGE_NONE in caps


def test_a_process_only_project_says_values_and_settings(warm_ferment, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert "**What happens if you change one process setting?**" in md
    assert "Nothing to change yet: every process setting is fixed." in caps
    [how] = [e for e in at.tabs[3].expander if e.label == wording.HOW_WORKED_OUT]
    checks = next(m.value for m in how.markdown if m.value.startswith("Checks:"))
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
